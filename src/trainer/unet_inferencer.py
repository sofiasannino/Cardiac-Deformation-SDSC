
import logging
import torch
import torch.nn as nn 
from pathlib import Path
import json
from time import time
import omegaconf
from tqdm import tqdm 
import numpy as np
from hydra.utils import instantiate

from src.datasets.inference_coord_dataset import nnUNetDataLoaderCoordTest
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from src.model.unet import PlainConvUNetCoord
import matplotlib.pyplot as plt
from hydra.core.hydra_config import HydraConfig

logger = logging.getLogger(__name__)



class UNetInferencerCoord():
    def __init__(self, config : omegaconf.DictConfig, device: torch.device): 
        
        self.device = device

        self.test_dataset = instantiate(config.dataset)
        self.batch_size = config.batch_size
        self.checkpoint_path = config.checkpoint_path
        self.loss = instantiate(config.loss)

        
        self.pool_size = config.pool_size
        self.hidden_coord = config.hidden_coord
        self.K = config.K

        self.network = self.load_checkpoint()
        self.network.to(self.device)


    def get_dataloader(self):
        dataset_test = self.test_dataset
        test_dataloader = nnUNetDataLoaderCoordTest(dataset_test, 
                                                self.batch_size)
        num_batches = len(test_dataloader)
        dl_test = SingleThreadedAugmenter(test_dataloader, None)
        return dl_test, num_batches
    
    def build_network_architecture(self) -> nn.Module:
        
        return PlainConvUNetCoord( # class used to perform 3d full res on run to segment intermediate frames, NOW HARDCODED THEN IMPROVE 
    input_channels=1,
    n_stages=6,
    features_per_stage=(32, 64, 128, 256, 320, 320),
    conv_op=nn.Conv3d,
    kernel_sizes=((1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)),
    strides=((1, 1, 1), (1, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2), (1, 2, 2)),
    n_conv_per_stage=(2, 2, 2, 2, 2, 2),
    num_classes=4,
    n_conv_per_stage_decoder=(2, 2, 2, 2, 2),
    pool_size=self.pool_size,
    hidden_coord=self.hidden_coord,
    K=self.K*3,  # number of control points
    conv_bias=True,
    norm_op=nn.InstanceNorm3d,
    norm_op_kwargs={"eps": 1e-5, "affine": True},
    dropout_op=None,
    dropout_op_kwargs=None,
    nonlin=nn.LeakyReLU,
    nonlin_kwargs={"negative_slope": 0.01, "inplace": True},
    deep_supervision=False,
    nonlin_first=False,
    final_activation="sigmoid",  # targets are normalized to [0, 1]
)
    def load_checkpoint(self) : 
        ckpt_path = Path(self.checkpoint_path)

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        if "network_weights" in ckpt:
            pretrained_state = ckpt["network_weights"]
        elif "state_dict" in ckpt:
            pretrained_state = ckpt["state_dict"]
        else:
            raise RuntimeError(
                f"Could not find 'network_weights' or 'state_dict' in checkpoint. "
                f"Available keys: {list(ckpt.keys())}"
            )
        
        # remove torch.compile prefix if present
        pretrained_state = {
            k.replace("_orig_mod.", "", 1): v
            for k, v in pretrained_state.items()
        }

        # load weights 
        model = self.build_network_architecture()
        model_state = model.state_dict()
        ckpt_state = ckpt["network_weights"]

        for k in ckpt_state:
            if k in model_state and ckpt_state[k].shape == model_state[k].shape:
                model_state[k] = ckpt_state[k]
        missing, unexpected = model.load_state_dict(model_state, strict=False)

        logger.warning("\nMissing keys after loading:")
        for k in missing:
                logger.info(k)

        logger.warning("\nUnexpected keys after loading:")
        for k in unexpected:
                logger.warning(k)

        return model
    def coords_norm_to_voxel(self, coords: torch.Tensor, spatial_shape):
        """
        coords: [B, K, 3] in normalized [z, y, x]
        spatial_shape: (D, H, W)
        """
        D, H, W = spatial_shape
        scale = torch.tensor(
            [D - 1, H - 1, W - 1],
            dtype=coords.dtype,
            device=coords.device,
        )
        return coords * scale 
    
    def test_step(self, batch: dict, batch_id : int) -> dict:

        data = batch["data"] # [B, C, D, H, W]
        coords_gt = batch["coords"]  # [B, K, 3]
        keys = batch["keys"]

        if isinstance(keys, str):
            keys = [keys]


        data = data.to(self.device, non_blocking=True)
        coords_gt = coords_gt.to(self.device, non_blocking=True)

        with torch.no_grad():
            
                coords_pred = self.network(data)

                if coords_pred.shape != coords_gt.shape:
                    raise RuntimeError(
                        f"Shape mismatch in inference: "
                        f"coords_pred={coords_pred.shape}, coords_gt={coords_gt.shape}"
                    )

                loss_raw = self.loss(coords_pred, coords_gt) #[B, K, 3]
                
                # reducing wrt landmarks 
                per_sample_loss = loss_raw.sum(dim=-1).mean(dim=1)  # [B]
                loss = per_sample_loss.mean()

                # Euclidean error per point in normalized coordinate space
                point_errors_norm = torch.linalg.norm( coords_pred - coords_gt, dim=-1, )  # [B, K]

                coords_pred_vox = self.coords_norm_to_voxel(coords_pred, data.shape[-3:])
                coords_gt_vox = self.coords_norm_to_voxel(coords_gt, data.shape[-3:])

                point_errors_vox = torch.linalg.norm( coords_pred_vox - coords_gt_vox, dim=-1, )  # [B, K]

                output_dir = Path(HydraConfig.get().runtime.output_dir)
                plot_dir = output_dir / "inference_plots"
                plot_dir.mkdir(parents=True, exist_ok=True)

                samples = []

                for i, key in enumerate(keys):
                    key = str(key)

                    plot_file = plot_dir / f"{key}.png"

                    self.plot_control_points_on_mri_slice(   
                        img=data[i].detach().cpu().numpy(),              # [C, D, H, W]
                        coords=coords_pred_vox[i].detach().cpu().numpy(), # [K, 3]
                        key=key,
                        out_file=plot_file,
                    )

                    samples.append({
                        "key": key,
                        "loss": float(per_sample_loss[i].detach().cpu().item()),
                        "mean_point_error_normalized": float(point_errors_norm[i].mean().detach().cpu().item()),
                        "mean_point_error_voxel": float(point_errors_vox[i].mean().detach().cpu().item()),

                    })

                return {
                "loss": float(loss.detach().cpu().item()),
                "mean_point_error_normalized": float(point_errors_norm.mean().detach().cpu().item()),
                "mean_point_error_voxel": float(point_errors_vox.mean().detach().cpu().item()),
                "samples": samples,
            }
    
    def plot_control_points_on_mri_slice(
        self,
        img: np.ndarray,
        coords: np.ndarray,
        key: str,
        out_file: Path,
        z_slice=None,
        tol=1.0,
    ):
        """
        img: [C, D, H, W] or [D, H, W]
        coords: [K, 3] voxel coordinates in [z, y, x]
        """

        img = np.asarray(img)
        coords = np.asarray(coords)

        if img.ndim == 4:
            volume = img[0]
        elif img.ndim == 3:
            volume = img
        else:
            raise ValueError(f"Expected img with shape [C,D,H,W] or [D,H,W], got {img.shape}")

        D, H, W = volume.shape

        if coords.shape[0] == 0:
            logger.warning("No control points to plot for %s", key)
            return

        z = coords[:, 0]
        y = coords[:, 1]
        x = coords[:, 2]

        if z_slice is None:
            z_slice = int(np.round(np.mean(z)))

        z_slice = int(np.clip(z_slice, 0, D - 1))

        #keep = np.abs(z - z_slice) <= tol

        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(7, 7))
        plt.imshow(volume[z_slice], cmap="gray")
        #plt.scatter(x[keep], y[keep], s=6, c="red", alpha=0.7)
        plt.scatter(x, y, s=6, c="red", alpha=0.7)

        #plt.title(f"Test on {key} | z={z_slice} | points={int(keep.sum())}")
        plt.title(f"Test on {key} | z={z_slice} | all points projected")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_file, dpi=200)
        plt.close()


    def run_inference(self):
        logger.info("Start running inference with U-Net coordinate regression model")
        self.network.eval() 
        
        dl_test, num_it = self.get_dataloader()

        logger.info(f"Number of inference frames: {num_it}")

        output_dir = Path(HydraConfig.get().runtime.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_file = output_dir / "inference_outputs.json"

        losses = []
        errors_norm = []
        errors_vox = []
        all_samples = []

        start_time = time()

        with torch.no_grad():
            for batch_id in tqdm(range(num_it), desc="Test iterations", colour="red"):
                batch = next(dl_test)

                outputs = self.test_step(batch, batch_id=batch_id)

                losses.append(outputs["loss"])
                errors_norm.append(outputs["mean_point_error_normalized"])
                errors_vox.append(outputs["mean_point_error_voxel"])
                all_samples.extend(outputs["samples"])

        results = {
            "checkpoint_path": str(self.checkpoint_path),
            "num_batches": int(num_it),
            "num_samples": int(len(all_samples)),
            "runtime_seconds": float(time() - start_time),

            "summary": {
                "mean_loss": float(np.mean(losses)) if len(losses) > 0 else None,
                "mean_point_error_normalized": float(np.mean(errors_norm)) if len(errors_norm) > 0 else None,
                "mean_point_error_voxel": float(np.mean(errors_vox)) if len(errors_vox) > 0 else None,
            },

            "samples": all_samples,
        }

        with open(json_file, "w") as f:
            json.dump(results, f, indent=4)

        logger.info(f"Saved inference outputs to: {json_file}")

        return results



         
         
         

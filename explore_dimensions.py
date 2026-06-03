import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from omegaconf import OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import logging 

from src.trainer.unet_trainer import nnUNetTrainerCoord
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="src/configs/model", config_name="debug")
def main(config):

    # define trainer
    unet_trainer = nnUNetTrainerCoord(config, device="cuda")

    # get dataloader
    mt_gen_train, _ = unet_trainer.get_dataloaders()

    Ds = []
    Hs = []
    Ws = []

    for i in range(config.num_patients):

       
        batch = next(mt_gen_train)
        data = batch["data"]
        if isinstance(data, torch.Tensor):
            shape = tuple(data.shape)
        else:
            shape = data.shape

        if len(shape) == 4:
            C, D, H, W = shape
        elif len(shape) == 5:
            B, C, D, H, W = shape
        else:
            raise ValueError(f"Unexpected data shape: {shape}")

        print(f"Sample {i}: C = {C}, D = {D}, H = {H}, W = {W}")

        Ds.append(D)
        Hs.append(H)
        Ws.append(W)
        for _ in range(config.number_frames - 1):
            next(mt_gen_train)

    Ds = np.asarray(Ds, dtype=np.float32)
    Hs = np.asarray(Hs, dtype=np.float32)
    Ws = np.asarray(Ws, dtype=np.float32)

    # plot histograms
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].hist(Ds, bins=50)
    axes[0].set_title("Distribution of D")
    axes[0].set_xlabel("D")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(Hs, bins=50)
    axes[1].set_title("Distribution of H")
    axes[1].set_xlabel("H")
    axes[1].set_ylabel("Frequency")

    axes[2].hist(Ws, bins=50)
    axes[2].set_title("Distribution of W")
    axes[2].set_xlabel("W")
    axes[2].set_ylabel("Frequency")

    plt.tight_layout()

    # save in Hydra run directory
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    out_file = output_dir / "data_shape_histograms.png"

    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"Saved histogram plot to: {out_file}")

    # show plot
    plt.show()


if __name__ == "__main__":
    main()
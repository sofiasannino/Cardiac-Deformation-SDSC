import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
import SimpleITK as sitk

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging
from src.model import ControlPoints

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import SimpleITK as sitk


def test_control_points_2d_3d(control_points, mask: sitk.Image):
    """
    Produce:
    1) plot 2D dei control points
    2) plot 2D della mask
    3) plot 3D mask + control points

    Assunzioni:
    - mask è 3D
    - control_points.points[label] contiene physical points (x,y,z)
    """

    mask_arr = sitk.GetArrayFromImage(mask)   # [z, y, x]

    # stessi colori per labels e punti
    label_colors = {
        1: "red",
        2: "lime",
        3: "deepskyblue"
    }

    # -----------------------------
    # 1) PLOT 2D SOLO CONTROL POINTS
    # proiezione XY
    # -----------------------------
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.set_title("Control points - proiezione XY")
    ax1.set_facecolor("black")

    for label, pts_phys in control_points.points.items():
        pts_idx = []

        for p in pts_phys:
            idx = mask.TransformPhysicalPointToIndex(tuple(map(float, p)))  # (x,y,z)
            pts_idx.append(idx)

        if len(pts_idx) > 0:
            pts_idx = np.array(pts_idx)
            x = pts_idx[:, 0]
            y = pts_idx[:, 1]

            ax1.scatter(
                x, y,
                s=50,
                c=label_colors.get(label, "white"),
                label=f"label {label}"
            )

    ax1.invert_yaxis()
    ax1.set_aspect("equal")
    ax1.legend()
    plt.tight_layout()

    # -----------------------------
    # 2) PLOT 2D SOLO MASK
    # proiezione XY con max lungo z
    # -----------------------------
    mask_xy = np.max(mask_arr, axis=0)  # [y, x]

    cmap = mcolors.ListedColormap(["black", "red", "lime", "deepskyblue"])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.set_title("Mask labels - proiezione XY")
    ax2.imshow(mask_xy, cmap=cmap, norm=norm, origin="upper")
    ax2.set_aspect("equal")
    plt.tight_layout()

    # -----------------------------
    # 3) PLOT 3D: MASK + CONTROL POINTS
    # per non appesantire troppo il plot,
    # mostro i voxel della mask con uno stride
    # -----------------------------
    fig3 = plt.figure(figsize=(9, 8))
    ax3 = fig3.add_subplot(111, projection="3d")
    ax3.set_title("Verifica 3D: mask + control points")

    # plot dei voxel della mask per ogni label
    for label in sorted(control_points.points.keys()):
        region = np.argwhere(mask_arr == label)   # [N, 3] in ordine (z,y,x)

        if len(region) > 0:
            # sottocampionamento per alleggerire il plot
            step = max(1, len(region) // 4000)
            region_sub = region[::step]

            z = region_sub[:, 0]
            y = region_sub[:, 1]
            x = region_sub[:, 2]

            ax3.scatter(
                x, y, z,
                s=1,
                c=label_colors.get(label, "white"),
                alpha=0.15
            )

    # plot dei control points
    for label, pts_phys in control_points.points.items():
        pts_idx = []

        for p in pts_phys:
            idx = mask.TransformPhysicalPointToIndex(tuple(map(float, p)))  # (x,y,z)
            pts_idx.append(idx)

        if len(pts_idx) > 0:
            pts_idx = np.array(pts_idx)
            x = pts_idx[:, 0]
            y = pts_idx[:, 1]
            z = pts_idx[:, 2]

            ax3.scatter(
                x, y, z,
                s=60,
                c=label_colors.get(label, "white"),
                edgecolors="black",
                depthshade=True,
                label=f"CP label {label}"
            )

    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    ax3.legend()

    plt.tight_layout()
    plt.show()


@hydra.main(version_base=None, config_path="src/configs", config_name="cardiodeform_model")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    #set_random_seed(config.trainer.seed)

    #project_config = OmegaConf.to_container(config)
    #logger = setup_saving_and_logging(config)
    #writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    control_points = ControlPoints(config.controlpoints)
    toy_mask = sitk.ReadImage("/home/renku/work/cardiac_deformation_final_dataset/patient001/labels/patient001_iframe0001.nii.gz", sitk.sitkUInt8)
    control_points.InitializeFromMask(toy_mask, num_points_per_label=20)
    test_control_points_2d_3d(control_points, toy_mask)

    

    # setup data_loader instances
    # batch_transforms should be put on device
    # dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    #model = instantiate(config.model).to(device)
    #logger.info(model)

    # get function handles of loss and metrics
    #loss_function = instantiate(config.loss_function).to(device)
    #metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    #trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    #optimizer = instantiate(config.optimizer, params=trainable_params)
    #lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    #epoch_len = config.trainer.get("epoch_len")

    #trainer = Trainer(
    #    model=model,
    #    criterion=loss_function,
    #    metrics=metrics,
    #    optimizer=optimizer,
    #    lr_scheduler=lr_scheduler,
    #    config=config,
    #    device=device,
     #   dataloaders=dataloaders,
      #  epoch_len=epoch_len,
       # logger=logger,
        #writer=writer,
        #batch_transforms=batch_transforms,
        #skip_oom=config.trainer.get("skip_oom", True),
    #)

    #trainer.train()


if __name__ == "__main__":
    main()

import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
import SimpleITK as sitk

from src.model import ControlPoints
from src.trainer.train_utils import test_control_points_2d_3d

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="cardiodeform_model")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    
    control_points = ControlPoints(config.model)
    toy_mask = sitk.ReadImage("/home/renku/work/s3-bucket/ACDC/training/patient001/patient001_frame01_gt.nii.gz", sitk.sitkUInt8)
    control_points.ExtractPoints(toy_mask)
    test_control_points_2d_3d(control_points, toy_mask)

    

   

if __name__ == "__main__":
    main()

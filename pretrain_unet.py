from omegaconf import OmegaConf
import hydra 

from src.trainer.unet_trainer import nnUNetTrainerCoord
from src.trainer.plot import plot_losses



@hydra.main(version_base = None, config_path = "src/configs/model" , config_name = "unet_coord.yaml" )
def main(config) : 

    # define trainer 
    unet_trainer = nnUNetTrainerCoord(config, device = "cuda")

    # run training 
    unet_trainer.run_training()

    # plot losses 
    plot_losses(config)
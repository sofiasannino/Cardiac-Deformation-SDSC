from omegaconf import OmegaConf
import hydra 

from src.trainer.unet_inferencer import  UNetInferencerCoord
from src.trainer.plot import plot_losses_test

@hydra.main(version_base = None, config_path = "src/configs/model" , config_name = "inferencer" )
def main(config) : 

    # define inferencer
    unet_trainer = UNetInferencerCoord(config, device = "cuda")

    # run training 
    results = unet_trainer.run_inference()

    # plot losses 
    plot_losses_test(results)



if __name__ == "__main__":
    main()

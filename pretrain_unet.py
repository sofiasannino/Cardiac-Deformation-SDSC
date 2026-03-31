import hydra
from omegaconf import DictConfig
from hydra.utils import call

from src import convert_acdc_to_nnunet


@hydra.main(version_base=None, config_path="configs/model", config_name="unet")
def main(cfg: DictConfig):

    # set up dataset according to unet requirements 
    convert_acdc_to_nnunet(
        acdc_training_root=cfg.data.acdc_training_root,
        out_dataset_root=cfg.data.out_dataset_root,
        acdc_info_json=cfg.data.acdc_info_json,
        copy_intermediate_to_imagesTs=cfg.data.copy_intermediate_to_imagesTs
    )
    

if __name__ == "__main__":
    main()

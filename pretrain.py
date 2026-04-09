import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
import logging
from hydra.utils import call

from src.datasets import convert_acdc_to_nnunet
from src.logger.logger import setup_logging


log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="src/configs/model", config_name="pretrain")
def main(cfg: DictConfig):


    # reproducibility
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    setup_logging(save_dir=output_dir)
    log.info("Logger set up successfully.")
    log.info(f"Output directory: {output_dir}")
    


    # set up dataset according to U-Net requirements
    if cfg.set_up_dataset:
        log.info("Setting up dataset...")
        convert_acdc_to_nnunet(
        acdc_training_root=cfg.acdc_training_root,
        out_dataset_root=cfg.out_dataset_root,
        acdc_info_json=cfg.acdc_info_json,
        copy_intermediate_to_imagesTs=cfg.copy_intermediate_to_imagesTs
    )
    

if __name__ == "__main__":
    main()

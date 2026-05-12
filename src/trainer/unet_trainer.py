import json
import logging
import multiprocessing
import os
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from typing import Tuple, Union, List
from sklearn.model_selection import KFold
from pathlib import Path
import json
from time import time
import omegaconf
import tqdm


import numpy as np
import torch
from torch import autocast
import torch.nn as nn
import torch.distributed as dist
#from torch.nn.parallel import DistributedDataParallel as DDP
from torch._dynamo import OptimizedModule
from contextlib import nullcontext as dummy_context
from hydra.core.hydra_config import HydraConfig
from batchgenerators.utilities.file_and_folder_operations import join, isfile
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter

from src.datasets.coord_dataset import nnUNetDatasetCoord, nnUNetDataLoaderCoord
from src.model.unet import PlainConvUNetCoord
from src.trainer.polylr import PolyLRScheduler

logger = logging.getLogger(__name__)





class nnUNetTrainerCoord():
    def __init__(self, config : omegaconf.DictConfig, device: torch.device):

        #super().__init__(plans, configuration, fold, dataset_json, device) # inherits all nnUnettrainer arguments

        # coordinate regression settings
        self.enable_deep_supervision = config.enable_deep_supervision
        self.max_num_patients = config.max_num_patients # set to a number to limit the number of patients for faster debugging
        self.batch_size = config.batch_size
        self.K = config.K
        self.pool_size = config.pool_size
        self.hidden_coord = config.hidden_coord
       

        
        self.is_ddp = config.is_ddp # we cannot distribute GPUs right now 
        self.local_rank = 0 if not self.is_ddp else dist.get_rank()

        self.device = device
        

        # print what device we are using
        #if self.is_ddp:  # implicitly it's clear that we use cuda in this case
        #     print(f"I am local rank {self.local_rank}. {device_count()} GPUs are available. The world size is "
        #          f"{dist.get_world_size()}."
        #          f"Setting device to {self.device}")
        #    self.device = torch.device(type='cuda', index=self.local_rank)
        #else:
        if self.device == "cuda":
            # we might want to let the user pick this but for now please pick the correct GPU with CUDA_VISIBLE_DEVICES=X
            self.device = torch.device(type='cuda', index=config.cuda_device_index)
        print(f"Using device: {self.device}")

        # loading and saving this class for continuing from checkpoint should not happen based on pickling. This
        # would also pickle the network etc. Bad, bad. Instead we just reinstantiate and then load the checkpoint we
        # need. So let's save the init args
        #self.my_init_kwargs = {}
        #for k in inspect.signature(self.__init__).parameters.keys():
        #    self.my_init_kwargs[k] = locals()[k]

        ###  Saving all the init args into class variables for later access
        #continue_training = plans.pop("continue_training")
        #logger_config = {"plans": plans, "configuration": configuration, "fold": fold, "dataset": dataset_json}
        #self.plans_manager = PlansManager(plans)
        #self.configuration_manager = self.plans_manager.get_configuration(configuration)
        #self.configuration_name = configuration
        #self.dataset_json = dataset_json
        self.fold = config.fold

        ### Setting all the folder names. We need to make sure things don't crash in case we are just running
        # inference and some of the folders may not be defined!
        self.preprocessed_dataset_folder = config.dataset_folder
        self.output_folder = Path(HydraConfig.get().runtime.output_dir)

        #self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
        #                                        self.configuration_manager.data_identifier) \
        #    if self.preprocessed_dataset_folder_base is not None else None
        #self.dataset_class = None  # -> initialize
        # unlike the previous nnunet folder_with_segs_from_previous_stage is now part of the plans. For now it has to
        # be a different configuration in the same plans
        # IMPORTANT! the mapping must be bijective, so lowres must point to fullres and vice versa (using
        # "previous_stage" and "next_stage"). Otherwise it won't work!
        #self.is_cascaded = self.configuration_manager.previous_stage_name is not None
        #self.folder_with_segs_from_previous_stage = \
        #    join(nnUNet_results, self.plans_manager.dataset_name,
        #         self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" +
        #         self.configuration_manager.previous_stage_name, 'predicted_next_stage', self.configuration_name) \
        #        if self.is_cascaded else None

        ### Dataset
        self.dataset_class = nnUNetDatasetCoord

        ### Some hyperparameters for you to fiddle with ---- > change this with config 
        self.initial_lr = config.initial_lr
        self.weight_decay = config.weight_decay
        self.oversample_foreground_percent = config.oversample_foreground_percent
        self.probabilistic_oversampling = config.probabilistic_oversampling
        self.num_train_iterations_per_epoch = config.num_train_iterations_per_epoch
        self.num_val_iterations_per_epoch = config.num_val_iterations_per_epoch
        self.num_epochs = config.num_epochs
        self.current_epoch = config.current_epoch
        self.enable_deep_supervision = config.enable_deep_supervision 

        ### Dealing with labels/regions
        #self.label_manager = self.plans_manager.get_label_manager(dataset_json)
        # labels can either be a list of int (regular training) or a list of tuples of int (region-based training)
        # needed for predictions. We do sigmoid in case of (overlapping) regions

        self.num_input_channels = None  # -> self.initialize()
        self.network = None  # -> self.build_network_architecture()
        self.optimizer = self.lr_scheduler = None  # -> self.initialize
        self.grad_scaler = None #(GradScaler("cuda") if not TORCH_HAS_OLD_GRADSCALER else GradScaler()) if self.device.type == 'cuda' else None ADD THIS LATER IF WE WANT MIXED PRECISION
        self.loss = None  # -> self.initialize

        ### Simple logging. Don't take that away from me!
        # initialize log file. This is just our log for the print statements etc. Not to be confused with lightning
        # logging --> removed since I use hydra 
        #timestamp = datetime.now()
        #maybe_mkdir_p(self.output_folder)
        #self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                             #(timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                              #timestamp.second))
        #self.logger = MetaLogger(self.output_folder, continue_training)
        #self.logger.update_config(logger_config)

        ### placeholders
        self.dataloader_train = self.dataloader_val = None  # see on_train_start

        ### initializing stuff for remembering things and such
        self._best_ema = None

        ### inference things
        #self.inference_allowed_mirroring_axes = None  # this variable is set in
        # self.configure_rotation_dummyDA_mirroring_and_inital_patch_size and will be saved in checkpoints

        ### checkpoint saving stuff and history saving 
        self.save_every = config.save_every
        self.disable_checkpointing = config.disable_checkpointing
        self.checkpoint_path = config.checkpoint_path # weights to load for encoder 
        self.history_json = Path(self.output_folder) / config.history_json

        self.was_initialized = False
    def load_pretrained_encoder_from_nnunet(self,  model, checkpoint_path: str, device: torch.device | str = "cpu", verbose: bool = True,):
        """
        Load only encoder.* weights from an old nnU-Net checkpoint into PlainConvUNetCoord.
        Expected old checkpoint:
            checkpoint["network_weights"]
        Loads:
            encoder.*

        Skips:
            decoder.*
            coord head
            anything with incompatible shape
        """

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if "network_weights" in ckpt:
            pretrained_state = ckpt["network_weights"]
        elif "state_dict" in ckpt:
            pretrained_state = ckpt["state_dict"]
        else:
            raise RuntimeError(
                f"Could not find 'network_weights' or 'state_dict' in checkpoint. "
                f"Available keys: {list(ckpt.keys())}"
            )

        # Remove torch.compile prefix if present
        pretrained_state = {
            k.replace("_orig_mod.", "", 1): v
            for k, v in pretrained_state.items()
        }

        model_state = model.state_dict()

        encoder_weights = {}
        skipped_shape_mismatch = []
        skipped_not_in_model = []
        skipped_not_encoder = []

        for k, v in pretrained_state.items():
            if not k.startswith("encoder."):
                skipped_not_encoder.append(k)
                continue

            if k not in model_state:
                skipped_not_in_model.append(k)
                continue

            if model_state[k].shape != v.shape:
                skipped_shape_mismatch.append((k, tuple(v.shape), tuple(model_state[k].shape)))
                continue

            encoder_weights[k] = v

        # Update only matching encoder weights
        model_state.update(encoder_weights)
        missing, unexpected = model.load_state_dict(model_state, strict=False)

        if verbose:
            logger.info(f"Loaded encoder tensors: {len(encoder_weights)}")
            logger.info(f"Skipped non-encoder tensors: {len(skipped_not_encoder)}")
            logger.info(f"Skipped encoder tensors not in model: {len(skipped_not_in_model)}")
            logger.info(f"Skipped encoder tensors with shape mismatch: {len(skipped_shape_mismatch)}")

            if len(skipped_shape_mismatch) > 0:
                logger.info("\nShape mismatches:")
                for item in skipped_shape_mismatch:
                    logger.info(item)

            logger.warning("\nMissing keys after loading:")
            for k in missing:
                logger.info(k)

            logger.warning("\nUnexpected keys after loading:")
            for k in unexpected:
                logger.warning(k)

        return model


    def initialize(self):
        if not self.was_initialized:
            ## DDP batch size and oversampling can differ between workers and needs adaptation
            # we need to change the batch size in DDP because we don't use any of those distributed samplers
            # self._set_batch_size_and_oversample()
            self.batch_size = 1

            self.num_input_channels = 1 # check 
            '''
            sig = inspect.signature(self.build_network_architecture)
            if 'plans_manager' in sig.parameters:
                self.network = self.build_network_architecture(
                    self.plans_manager,
                    self.configuration_manager,
                    self.num_input_channels,
                    self.label_manager.num_segmentation_heads,
                    self.enable_deep_supervision
                ).to(self.device)
            else:
                warnings.warn(
                    f"Trainer {self.__class__.__name__} uses the old build_network_architecture signature. "
                    "Please update to the new signature: "
                    "build_network_architecture(plans_manager, configuration_manager, "
                    "num_input_channels, num_output_channels, enable_deep_supervision). "
                    "The old signature will be removed in a future version.",
                    DeprecationWarning, stacklevel=2,
                )
            '''
            self.network = self.build_network_architecture().to(self.device)

            # initalize network weights : He initialiation for coord head, while load pretrained weights for the rest of the network if pretrained weights are provided
            if self.checkpoint_path is not None:
                logger.info(f"Initializing encoder weights from checkpoint: {self.checkpoint_path}")
                self.network.apply(self.network.initialize)
                self.network = self.load_pretrained_encoder_from_nnunet(self.network, self.checkpoint_path, device=self.device)
            else: 
                raise RuntimeError("No checkpoint path provided for encoder initialization. Please provide a valid checkpoint path or set checkpoint_path to None to disable encoder initialization.")

            # compile network for free speedup # CHECK 
            #if self._do_i_compile():
            #    self.print_to_log_file('Using torch.compile...')
            #    self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()

            # if ddp, wrap in DDP wrapper
            #if self.is_ddp:
            #    self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
            #    self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()


            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True

            logger_config_hparas = {
                "initial_lr": self.initial_lr,
                "weight_decay": self.weight_decay,
                #"oversample_foreground_percent": self.oversample_foreground_percent,
                #"probabilistic_oversampling": self.probabilistic_oversampling,
                "num_iterations_per_epoch": self.num_train_iterations_per_epoch,
                "num_val_iterations_per_epoch": self.num_val_iterations_per_epoch,
                "num_epochs": self.num_epochs,
                #"enable_deep_supervision": self.enable_deep_supervision,
                "batch_size": self.batch_size,
                }
            logger.info("Training hyperparameters:")
            for k, v in logger_config_hparas.items():
                logger.info(f"  {k}: {v}")
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")
    '''
    def _do_i_compile(self):
        # new default: compile is enabled!

        # compile does not work on mps
        if self.device == torch.device('mps'):
            if 'nnUNet_compile' in os.environ.keys() and os.environ['nnUNet_compile'].lower() in ('true', '1', 't'):
                self.print_to_log_file("INFO: torch.compile disabled because of unsupported mps device")
            return False

        # CPU compile crashes for 2D models. Not sure if we even want to support CPU compile!? Better disable
        if self.device == torch.device('cpu'):
            if 'nnUNet_compile' in os.environ.keys() and os.environ['nnUNet_compile'].lower() in ('true', '1', 't'):
                self.print_to_log_file("INFO: torch.compile disabled because device is CPU")
            return False

        # default torch.compile doesn't work on windows because there are apparently no triton wheels for it
        # https://discuss.pytorch.org/t/windows-support-timeline-for-torch-compile/182268/2
        if os.name == 'nt':
            if 'nnUNet_compile' in os.environ.keys() and os.environ['nnUNet_compile'].lower() in ('true', '1', 't'):
                self.print_to_log_file("INFO: torch.compile disabled because Windows is not natively supported. If "
                                       "you know what you are doing, check https://discuss.pytorch.org/t/windows-support-timeline-for-torch-compile/182268/2")
            return False

        if 'nnUNet_compile' not in os.environ.keys():
            return True
        else:
            return os.environ['nnUNet_compile'].lower() in ('true', '1', 't')
    
    def _save_debug_information(self):
        # saving some debug information
        if self.local_rank == 0:
            dct = {}
            for k in self.__dir__():
                if not k.startswith("__"):
                    if not callable(getattr(self, k)) or k in ['loss', ]:
                        dct[k] = str(getattr(self, k))
                    elif k in ['network', ]:
                        dct[k] = str(getattr(self, k).__class__.__name__)
                    else:
                        # print(k)
                        pass
                if k in ['dataloader_train', 'dataloader_val']:
                    dl = getattr(self, k)
                    if hasattr(dl, 'generator'):
                        dct[k + '.generator'] = str(dl.generator)
                        if hasattr(dl.generator, 'transforms'):
                            try:
                                dct[k + '.generator.transforms'] = str(dl.generator.transforms)
                            except Exception as e:
                                dct[k + '.generator.transforms'] = f"Could not stringify generator.transforms: {type(e).__name__}: {e}"
                    if hasattr(dl, 'num_processes'):
                        dct[k + '.num_processes'] = str(dl.num_processes)
                    if hasattr(dl, 'transform'):
                        dct[k + '.transform'] = str(dl.transform)
            import subprocess
            hostname = subprocess.getoutput(['hostname'])
            dct['hostname'] = hostname
            torch_version = torch.__version__
            if self.device.type == 'cuda':
                gpu_name = torch.cuda.get_device_name()
                dct['gpu_name'] = gpu_name
                cudnn_version = torch.backends.cudnn.version()
            else:
                cudnn_version = 'None'
            dct['device'] = str(self.device)
            dct['torch_version'] = torch_version
            dct['cudnn_version'] = cudnn_version
            save_json(dct, join(self.output_folder, "debug.json"))
    '''
    def build_network_architecture(self) -> nn.Module:
        """
        This is where you build the architecture according to the plans. There is no obligation to use
        get_network_from_plans, this is just a utility we use for the nnU-Net default architectures. You can do what
        you want. Even ignore the plans and just return something static (as long as it can process the requested
        patch size)
        """
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
    K=self.K,  # number of control points
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

    #def _get_deep_supervision_scales(self):
        #pass # check 

    #def _set_batch_size_and_oversample(self):
        #pass # check 

    def _build_loss(self) : 
        return nn.HuberLoss(reduction="mean", delta = 0.02) # delta adjusted to normalized coordinates 
    '''
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """
        This function overrides the original , to implement coordinate regression 
        """
        patch_size = self.configuration_manager.patch_size
        

        do_dummy_2d_data_aug = False
        initial_patch_size = patch_size
        mirror_axes = ()

        # Important: disable test-time mirroring.
        # For coordinate regression, mirrored predictions would need to be un-mirrored.
        self.inference_allowed_mirroring_axes = None

        self.print_to_log_file("Coordinate regression: disabled rotation, scaling, dummy 2D DA, and mirroring.")

        rotation_for_DA = (0.0, 0.0)

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    #def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        #pass

    #def print_plans(self): the original will be used and will be likely give wrong info by now
        #pass
    '''
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    #def plot_network_architecture(self):
       # pass
    def generate_crossval_split(self, train_identifiers: List[str], seed=12345, n_splits=5) -> List[dict[str, List[str]]]:
        splits = []
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for i, (train_idx, test_idx) in enumerate(kfold.split(train_identifiers)):
            train_keys = np.array(train_identifiers)[train_idx]
            test_keys = np.array(train_identifiers)[test_idx]
            splits.append({})
            splits[-1]['train'] = list(train_keys)
            splits[-1]['val'] = list(test_keys)
        return splits

    def do_split(self):
        """
        Patient-level split for coordinate regression.
        The dataset identifiers are frame-level keys, for example:
            patient096_00042
        But each identifier contains:
            dataset.identifiers[key]["patient"] = "patient096"
        split by patient to avoid leakage between train and validation.
        """

        dataset = self.dataset_class(
            self.preprocessed_dataset_folder,
            identifiers=None, max_num_patients=self.max_num_patients, num_frames_per_patient=20
        )

        all_frame_keys = sorted(list(dataset.identifiers.keys()))

        if self.fold == "all":
            tr_keys = all_frame_keys
            val_keys = all_frame_keys

            logger.info(
                f"Using fold='all': {len(tr_keys)} training and {len(val_keys)} validation frames."
            )

            return tr_keys, val_keys

        # build patient -> list of frame keys
        patient_to_frames = {}

        for key in all_frame_keys:
            patient = dataset.identifiers[key]["patient"]
            patient_to_frames.setdefault(patient, []).append(key)

        patients_sorted = sorted(list(patient_to_frames.keys()))

        logger.info(
            f"Found {len(patients_sorted)} patients and {len(all_frame_keys)} frames."
        )

        # create deterministic patient-level 5-fold split 
        patient_splits = self.generate_crossval_split(
            patients_sorted,
            seed=12345,
            n_splits=5,
        )

        logger.info(f"Desired fold for training: {self.fold}")

        if self.fold < len(patient_splits):
            train_patients = patient_splits[self.fold]["train"]
            val_patients = patient_splits[self.fold]["val"]

        else:
            logger.info(
                f"You requested fold {self.fold}, but only {len(patient_splits)} folds exist. "
                "Creating deterministic 80:20 patient-level split."
                % (self.fold, len(patient_splits))
            )

            rnd = np.random.RandomState(seed=12345 + self.fold)

            idx_tr = rnd.choice(
                len(patients_sorted),
                int(len(patients_sorted) * 0.8),
                replace=False,
            )

            train_patients = [patients_sorted[i] for i in idx_tr]
            val_patients = [
                patients_sorted[i]
                for i in range(len(patients_sorted))
                if i not in idx_tr
            ]

        # expand patient-level split to frame-level keys
        tr_keys = []
        val_keys = []

        for patient in train_patients:
            tr_keys.extend(patient_to_frames[patient])

        for patient in val_patients:
            val_keys.extend(patient_to_frames[patient])

        tr_keys = sorted(tr_keys)
        val_keys = sorted(val_keys)

        logger.info(
            "This split has %d training patients and %d validation patients."
            % (len(train_patients), len(val_patients))
        )

        logger.info(
            "This split has %d training frames and %d validation frames."
            % (len(tr_keys), len(val_keys))
        )

        # safety check: no frame overlap
        frame_overlap = set(tr_keys).intersection(set(val_keys))
        if len(frame_overlap) > 0:
            self.print_to_log_file(
                f"WARNING: Frame leakage detected! Overlapping frames: {sorted(list(frame_overlap))}"
            )

        # safety check: no patient overlap
        train_patients_check = set(dataset.identifiers[k]["patient"] for k in tr_keys)
        val_patients_check = set(dataset.identifiers[k]["patient"] for k in val_keys)

        patient_overlap = train_patients_check.intersection(val_patients_check)

        if len(patient_overlap) > 0:
            self.print_to_log_file(
                f"WARNING: Patient leakage detected! Patients in both train and val: "
                f"{sorted(list(patient_overlap))}"
            )

        return tr_keys, val_keys

    def get_tr_and_val_datasets(self):
        # do split 
        tr_keys, val_keys = self.do_split()

        split_out = Path(self.output_folder) / f"coord_split_fold_{self.fold}.json"
        split_out.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving train/val split to {split_out}")
        with open(split_out, "w", encoding="utf-8") as f:
            json.dump(
        {
            "fold": self.fold,
            "num_train": len(tr_keys),
            "num_val": len(val_keys),
            "train": list(tr_keys),
            "val": list(val_keys),
        },
        f,
        indent=2,
    )

        all_identifiers = self.dataset_class._get_identifiers(self.preprocessed_dataset_folder)
        tr_identifiers = { k: all_identifiers[k] for k in tr_keys}
        val_identifiers = {k: all_identifiers[k] for k in val_keys }
        dataset_tr = self.dataset_class( self.preprocessed_dataset_folder, identifiers=tr_identifiers,)

        dataset_val = self.dataset_class(self.preprocessed_dataset_folder, identifiers=val_identifiers,)

        self.num_train_samples = len({v["patient"] for v in dataset_tr.identifiers.values()})
        self.num_val_samples = len({v["patient"] for v in dataset_val.identifiers.values()})

        return dataset_tr, dataset_val

    def get_dataloaders(self):
        #if self.dataset_class is None:
        #    self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        #patch_size = self.configuration_manager.patch_size

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        #deep_supervision_scales = None

        #(rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        #tr_transforms = self.get_training_transforms(
        #    patch_size , rotation_for_DA , deep_supervision_scales , mirror_axes, do_dummy_2d_data_aug,
        #    use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
        #    is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
        #    regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
        #    ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        #val_transforms = None #self.get_validation_transforms(deep_supervision_scales,
                                                        #is_cascaded=self.is_cascaded,
                                                        #foreground_labels=self.label_manager.foreground_labels,
                                                        #regions=self.label_manager.foreground_regions if
                                                        #self.label_manager.has_regions else None,
                                                        #ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        dl_tr = nnUNetDataLoaderCoord(dataset_tr,
                 self.batch_size, # 1 
                 sampling_probabilities=None,
                 transforms=None)
        dl_val = nnUNetDataLoaderCoord(dataset_val,
                 self.batch_size, # 1 
                 sampling_probabilities=None,
                 transforms=None)

        allowed_num_processes = 0 # get_allowed_n_proc_DA(), for first run not implemented 
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)
        # # let's get this party started
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val
    '''

    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        # canceled things here
        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=None,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))
        
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )
       
        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))
       
        transforms.append(
            RemoveLabelTansform(-1, 0)
        )
       
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=0.5
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=0.5
                    ), apply_probability=0.2
                )
            )
        
        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )
       
        #if deep_supervision_scales is not None:
            #transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)
    

    @staticmethod
    def get_validation_transforms(
            deep_supervision_scales: Union[List, Tuple, None],
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        transforms.append(
            RemoveLabelTansform(-1, 0)
        )
       
        if is_cascaded:
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
       
        return None #ComposeTransforms(transforms)
        '''

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well! ---> for coordinate regression we don't need deep supervision by now 
        """
        pass  

    def on_train_start(self):
        if not self.was_initialized:
            self.initialize()

        # dataloaders must be instantiated here (instead of __init__) because they need access to the training data
        # which may not be present  when doing inference
        self.dataloader_train, self.dataloader_val = self.get_dataloaders()


        

        logger.info(f"Output folder: {self.output_folder}")

        # make sure deep supervision is on in the network
        #self.set_deep_supervision_enabled(self.enable_deep_supervision)

        logger.info(f"Using device: {self.device}")
        logger.info(f"Using network architecture:\n{self.network}")
        logger.info(f"Using loss:\n{self.loss}")
        logger.info(f"Using optimizer:\n{self.optimizer}")
        logger.info(f"Number of patients in training set: {self.num_train_samples}")
        logger.info(f"Number of patients in validation set: {self.num_val_samples}")
        logger.info(f"Number of frames in train : {self.num_train_samples * 20}")
        logger.info(f"Number of frames in val : {self.num_val_samples * 20}")
        self.empty_cache(self.device)

        # maybe unpack
        #if self.local_rank == 0:
        #    self.dataset_class.unpack_dataset(
        #        self.preprocessed_dataset_folder,
        #        overwrite_existing=False,
        #     num_processes=max(1, round(get_allowed_n_proc_DA() // 2)),
        #         verify=True)

        #if self.is_ddp:
        #    dist.barrier() 

        # copy plans and dataset.json so that they can be used for restoring everything we need for inference
        #save_json(self.plans_manager.plans, join(self.output_folder_base, 'plans.json'), sort_keys=False)
        #save_json(self.dataset_json, join(self.output_folder_base, 'dataset.json'), sort_keys=False)

        # we don't really need the fingerprint but its still handy to have it with the others
        #shutil.copyfile(join(self.preprocessed_dataset_folder_base, 'dataset_fingerprint.json'),
        #                join(self.output_folder_base, 'dataset_fingerprint.json'))

        # produces a pdf in output folder
        #self.plot_network_architecture()

        #self._save_debug_information()

        logger.info(f"batch size: {self.batch_size}")

        # save epochs informations
        self.training_history = {
                "epochs": [],
            }
        # print(f"oversample: {self.oversample_foreground_percent}")

    def on_train_end(self): # a little customized from nnunet
        # dirty hack because on_epoch_end increments the epoch counter
        # and this is executed afterwards
        self.current_epoch -= 1
        self.save_checkpoint(str(Path(self.output_folder) /"checkpoint_final.pth"))
        self.current_epoch += 1

        # delete latest checkpoint
        latest_checkpoint = str(Path(self.output_folder)/ "checkpoint_latest.pth")
        if self.local_rank == 0 and isfile(latest_checkpoint):
            os.remove(latest_checkpoint)

        # shut down dataloaders if they use multiprocessing
        old_stdout = sys.stdout
        try:
            with open(os.devnull, "w") as f:
                sys.stdout = f

                if (
                    self.dataloader_train is not None
                    and isinstance(
                        self.dataloader_train,
                        (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter),
                    )
                ):
                    self.dataloader_train._finish()

                if (
                    self.dataloader_val is not None
                    and isinstance(
                        self.dataloader_val,
                        (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter),
                    )
                ):
                    self.dataloader_val._finish()

        finally:
            sys.stdout = old_stdout

        self.empty_cache(self.device)
        logger.info("Training done.")


    def empty_cache(self, device: torch.device):
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            from torch import mps
            mps.empty_cache()
        else:
            pass


    def on_train_epoch_start(self): # customized 
        self.network.train()

        # update learning rate
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(self.current_epoch)

        current_lr = self.optimizer.param_groups[0]["lr"]

        logger.info("")
        logger.info(f"Epoch {self.current_epoch}")
        logger.info(f"Current learning rate: {np.round(current_lr, decimals=5)}")

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]   # [B, 1, D, H, W]
        target = batch["coords"]    # [B, K, 3] for K control points and 3 coordinates (z, y, x)

        data = data.to(self.device, non_blocking=True)
        #if isinstance(target, list):
        #   target = [i.to(self.device, non_blocking=True) for i in target]
        #else:
        target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        #with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context(): #REINTRODUCE THIS LATER 
        output = self.network(data)
        if output.shape != target.shape:
                raise RuntimeError(f"Output and target shapes do not match. "
                                  f"output={output.shape}, target={target.shape}")
        # del data ---> DO TAB LATER 
        l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}
    
    def collate_outputs(self, outputs: List[dict]):
        """
        used to collate default train_step and validation_step outputs. If you want something different then you gotta
        extend this

        we expect outputs to be a list of dictionaries where each of the dict has the same set of keys
        """
        collated = {}
        for k in outputs[0].keys():
            if np.isscalar(outputs[0][k]):
                collated[k] = [o[k] for o in outputs]
            elif isinstance(outputs[0][k], np.ndarray):
                collated[k] = np.vstack([o[k][None] for o in outputs])
            elif isinstance(outputs[0][k], list):
                collated[k] = [item for o in outputs for item in o[k]]
            else:
                raise ValueError(f'Cannot collate input of type {type(outputs[0][k])}. '
                                f'Modify collate_outputs to add this functionality')
        return collated

    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = self.collate_outputs(train_outputs)

        if "loss" not in outputs:
            raise RuntimeError(
            f"Expected key 'loss' in train_outputs, got keys: {list(outputs.keys())}")

        losses = np.asarray(outputs["loss"], dtype=np.float32).reshape(-1)

        #if self.is_ddp:
        #    world_size = dist.get_world_size()

        #     gathered_losses = [None for _ in range(world_size)]
        #    dist.all_gather_object(gathered_losses, losses)

        #    losses = np.concatenate(
        #        [np.asarray(x).reshape(-1) for x in gathered_losses]
        #    )

        self.current_train_loss = float(np.mean(losses))

        logger.info( f"Train loss epoch {self.current_epoch}: {self.current_train_loss:.6f}")

    def on_validation_epoch_start(self):
        self.network.eval()

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        coords_gt = batch["coords"]

        data = data.to(self.device, non_blocking=True)
        coords_gt = coords_gt.to(self.device, non_blocking=True)

        with torch.no_grad():
            #with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                coords_pred = self.network(data)

                if coords_pred.shape != coords_gt.shape:
                    raise RuntimeError(
                        f"Shape mismatch in validation: "
                        f"coords_pred={coords_pred.shape}, coords_gt={coords_gt.shape}"
                    )

                loss = self.loss(coords_pred, coords_gt)

                # Euclidean error per point in normalized coordinate space
                point_errors = torch.linalg.norm(coords_pred - coords_gt, dim=-1)  # [B, K]
                mean_point_error = point_errors.mean()

        return {"loss": loss.detach().cpu().item(), "mean_point_error": mean_point_error.detach().cpu().item()}

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs = self.collate_outputs(val_outputs)

        if "loss" not in outputs:
            raise RuntimeError(
                f"Expected key 'loss' in val_outputs, got keys: {list(outputs.keys())}"
            )

        if "mean_point_error" not in outputs:
            raise RuntimeError(
                f"Expected key 'mean_point_error' in val_outputs, got keys: {list(outputs.keys())}"
            )

        losses = np.asarray(outputs["loss"], dtype=np.float32).reshape(-1)
        mean_errors = np.asarray(outputs["mean_point_error"], dtype=np.float32).reshape(-1)

        #if self.is_ddp:
           # world_size = dist.get_world_size()

            # = [None for _ in range(world_size)]
           # dist.all_gather_object(gathered_losses, losses)
           # losses = np.concatenate(
            #    [np.asarray(x).reshape(-1) for x in gathered_losses]
           # )

          #  gathered_errors = [None for _ in range(world_size)]
          #  dist.all_gather_object(gathered_errors, mean_errors)
          #  mean_errors = np.concatenate(
           #     [np.asarray(x).reshape(-1) for x in gathered_errors]
           # )

        self.current_val_loss = float(np.mean(losses))
        self.current_val_mean_point_error = float(np.mean(mean_errors))

        logger.info(
            f"Validation loss epoch {self.current_epoch}: {self.current_val_loss:.6f}"
        )
        logger.info(
            f"Validation mean point error epoch {self.current_epoch}: "
            f"{self.current_val_mean_point_error:.6f}"
        )


    def on_epoch_start(self):
        self.epoch_start_time = time()
        logger.info(f"Epoch {self.current_epoch} started at {self.epoch_start_time}")

    def on_epoch_end(self):
        epoch_end_time = time()
        epoch_time = epoch_end_time - self.epoch_start_time

        current_lr = self.optimizer.param_groups[0]["lr"]

        logger.info(f"train_loss: {self.current_train_loss:.6f}")
        logger.info(f"val_loss: {self.current_val_loss:.6f}")
        logger.info(f"val_mean_point_error: {self.current_val_mean_point_error:.6f}")
        logger.info(f"Epoch time: {epoch_time:.2f} s")

        epoch_record = {
            "epoch": int(self.current_epoch),
            "train_loss": float(self.current_train_loss),
            "val_loss": float(self.current_val_loss),
            "val_mean_point_error": float(self.current_val_mean_point_error),
            "epoch_time": float(epoch_time),
            "lr": float(current_lr),
        }

        # update history 
        self.training_history["epochs"].append(epoch_record)

        # save evry epoch in same file JSON
        if self.local_rank == 0:
            self.history_json.parent.mkdir(parents=True, exist_ok=True)

            with open(self.history_json, "w", encoding="utf-8") as f:
                json.dump(self.training_history, f, indent=2)

            logger.info(f"Saved training history to {self.history_json}")

        # periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(str(Path(self.output_folder) / "checkpoint_latest.pth"))

        # best checkpointing based on validation loss
        if self._best_ema is None or self.current_val_loss < self._best_ema:
            self._best_ema = self.current_val_loss
            logger.info(
                f"New best validation loss: {self._best_ema:.6f}. Saving checkpoint_best.pth"
            )
            self.save_checkpoint(str(Path(self.output_folder) / "checkpoint_best.pth"))

        self.current_epoch += 1
    def save_checkpoint(self, filename: str) -> None: # customized for coord regression
        if self.local_rank != 0:
            return

        if self.disable_checkpointing:
            logger.info("No checkpoint written, checkpointing is disabled.")
            return

        # unwrap DDP
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network

        # unwrap torch.compile
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod

        checkpoint = {
            "network_weights": mod.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "lr_scheduler_state": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            "grad_scaler_state": self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
            "_best_ema": self._best_ema,
            "current_epoch": self.current_epoch + 1,
            "trainer_name": self.__class__.__name__,
            "checkpoint_path_pretrained_encoder": self.checkpoint_path,
            "model_type": "PlainConvUNetCoord",
            "training_history" : self.training_history
        }

        torch.save(checkpoint, filename)
        logger.info(f"Saved checkpoint to {filename}")

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(
                filename_or_checkpoint,
                map_location=self.device,
                weights_only=False,
            )
        else:
            checkpoint = filename_or_checkpoint

        new_state_dict = {}

        for k, value in checkpoint["network_weights"].items():
            key = k

            # remove DataParallel/DDP prefix if needed
            if key not in self.network.state_dict().keys() and key.startswith("module."):
                key = key[7:]

            new_state_dict[key] = value

        # load network
        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)

        # load optimizer
        if "optimizer_state" in checkpoint and checkpoint["optimizer_state"] is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        # load scheduler
        if (
            self.lr_scheduler is not None
            and "lr_scheduler_state" in checkpoint
            and checkpoint["lr_scheduler_state"] is not None
        ):
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state"])

        # load grad scaler
        if (
            self.grad_scaler is not None
            and "grad_scaler_state" in checkpoint
            and checkpoint["grad_scaler_state"] is not None
        ):
            self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state"])

        self.current_epoch = checkpoint.get("current_epoch", 0)
        self._best_ema = checkpoint.get("_best_ema", None)

        self.training_history = checkpoint.get(
            "training_history",
            {
                "epoch": [],
                "train_loss": [],
                "val_loss": [],
                "val_mean_point_error": [],
                "epoch_time": [],
                "lr": [],
            },
        )

        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

    def perform_actual_validation(self, save_probabilities: bool = False):
        "not implemented for coordinate regression, as we do not save probabilities but coordinates"
        pass

    def run_training(self):
        self.on_train_start()

        logging.info(f"Start running U-NET FOR COORDINATE REGRESSION MODEL ")

        for epoch in tqdm(range(self.current_epoch, self.num_epochs), desc = "Epochs", colour = "green" ):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in tqdm(range(self.num_train_iterations_per_epoch), desc= "Training iterations", colour = "red"):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in tqdm(range(self.num_val_iterations_per_epoch), desc = "Validation iterations", colour = "blue"):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()
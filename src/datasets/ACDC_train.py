import os
import random
import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig, ListConfig
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import Dataset

import pydicom
import SimpleITK as sitk
from PIL import Image
import torchvision.transforms as transforms
import datasets.transforms as transform
from datasets.transforms import resample





class ACDC_Dataset(Dataset) : 
    def __init__(self, infos, cfg_ACDCA_train : DictConfig, is_train=True):
        super().__init__()
        self.target_size = tuple(cfg_ACDCA_train.target_size)      # (H, W, D)
        self.target_spacing = tuple(cfg_ACDCA_train.target_spacing)  # (sx, sy, sz)           
        self.crop_length = cfg_ACDCA_train.n_frame # Tmax 
        self.train_dict = infos["train"]
        self.test_dict = infos["test"]
        self.all_dict = self.preprocess(is_train)
        self.file_list = list(self.all_dict.keys())
        self.transform = transforms.Compose([
                                transform.ToTensorVideo(),
                                transform.CenterCropVideo((self.target_size[0], self.target_size[1])),
                                transform.NormalizeVideo()
                                ]) 

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        case = self.all_dict[self.file_list[index]]
        patient_dir = case["patient_dir"]
        pid = case.get("patient_id") 
        data_path = os.path.join(patient_dir, f"patient{pid}_4d.nii.gz")
        data=sitk.ReadImage(data_path)  # [T, D, H, W]
       
        ### Resample to target spacing 
        current_video = resample(data, self.target_spacing) # [T, D, H, W]
        length, depth, height, width = current_video.shape

        ### Crop/pad to target size and normalize to [0,1] 

            # time cropping
        # ES_index = case["ES"]
        # current_video = transform.ResizedVideo( cfg_ACDCA_train.n_frames,  ES_index, cfg_ACDCA_train.interpolation_mode)
        if length < self.crop_length:
            comp_length = self.crop_length - length
            comp_frames = np.flip(current_video[-1-comp_length:-1, ...], axis=0)
            current_video = np.concatenate((current_video, comp_frames), axis=0)
        
        elif length > self.crop_length:
            start_idx = random.randint(0, length-self.crop_length-1)
            current_video = current_video[start_idx:start_idx+self.crop_length, ...] # shape [T=n_frame, D, H, W]

            # depth cropping
        if depth >= self.target_size[-1]:
            sd = int((depth - self.target_size[-1]) / 2)
            current_video = current_video[:, sd:sd + self.target_size[-1], ...]
        else:
            sd = int((self.target_size[-1] - depth) / 2)
            current_video_ = np.zeros([length, self.target_size[-1], height, width], dtype=current_video.dtype)
            current_video_[:, sd:sd + depth, :, :] = current_video
            current_video = current_video_

            #  apply transforms 
        current_video = self.transform(current_video) # [T=n_frame, D, H, W]


        return current_video
    
    def preprocess(self, is_train):
        all_dict = dict()
        count = 0
        if is_train: 
            for key in list(self.train_dict.keys()):
                all_dict[count] = (self.train_dict[key])
                count += 1
        else:
            for key in list(self.test_dict.keys()):
                all_dict[count] = (self.test_dict[key])
                count += 1
        return all_dict
    
    def get_pid(self, index) :
        case = self.all_dict[self.file_list[index]] 
        return case["patient_id"]
   
                 



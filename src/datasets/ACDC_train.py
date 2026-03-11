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

from data_utils import load_acdc_info



class ACDC_Dataset(Dataset) : 
    def __init__(self, infos, cfg_ACDCA_train : DictConfig, is_train=True):
        super().__init__()
        self.target_size = tuple(cfg_ACDCA_train.target_size)      # (H, W, D)
        self.target_spacing = tuple(cfg_ACDCA_train.target_spacing)  # (sx, sy, sz)
        self.min_max = tuple(cfg_ACDCA_train.intensity)            

        self.train_dict = infos["train"]
        self.test_dict = infos["test"]
        self.all_dict = self.preprocess(is_train)
        self.file_list = list(self.all_dict.keys())
        self.transform = transforms.Compose([
                                transform.ToTensorVideo(),
                                transform.CenterCropVideo((self.target_size[0], self.target_size[1])),]) 

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        case = self.all_dict[self.file_list[index]]
        patient_dir = case["patient_dir"]
        pid = case.get("patient_id") 
        data_path = os.path.join(patient_dir, f"patient{pid}_4d.nii.gz")
        data=sitk.ReadImage(data_path)  # [T, D, H, W]
       
        # resample to target spacing 
        current_video = resample(data, self.target_spacing) # [T, D, H, W]
        length, depth, height, width = current_video.shape

        # crop/pad to target size  

        if depth >= self.target_size[-1]:
            sd = int((depth - self.target_size[-1]) / 2)
            current_video = current_video[:, sd:sd + self.target_size[-1], ...]
        else:
            sd = int((self.target_size[-1] - depth) / 2)
            current_video_ = np.zeros([length, self.target_size[-1], height, width], dtype=current_video.dtype)
            current_video_[:, sd:sd + depth, :, :] = current_video
            current_video = current_video_

        current_video = self.transform(current_video) # [T, D, H, W]

        # normalize to [0,1] 

        current_video = current_video - current_video.min()
        current_video = current_video / current_video.std()
        current_video = current_video / current_video.max()

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
    '''
    def get_ES(self, index) :
        
       # get ES time frame for evaluation
         
        case = self.all_dict[self.file_list[index]]
        patient_dir = case["patient_dir"]
        info_path = os.path.join(patient_dir, f"Info.cfg")
        for line in Path(info_path).read_text().splitlines():
            k, v = line.split(":", 1)
            if k.strip() == "ES" : 
                return int(v.strip())
        raise KeyError(f"ES not found in {info_path}")
    '''
                 



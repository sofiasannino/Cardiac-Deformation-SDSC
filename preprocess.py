import os
import shutil
from pathlib import Path
from typing import List


from dataclasses import dataclass

import hydra
from omegaconf import DictConfig

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, save_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed

from batchgenerators.utilities.file_and_folder_operations import nifti_files, join, maybe_mkdir_p, save_json
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
import numpy as np
import nibabel as nib


def make_out_dirs(dataset_id: int, task_name="ACDC"):
    dataset_name = f"Dataset{dataset_id:03d}_{task_name}"

    out_dir = Path(nnUNet_raw.replace('"', "")) / dataset_name
    out_train_dir = out_dir / "imagesTr"
    out_labels_dir = out_dir / "labelsTr"
    out_test_dir = out_dir / "imagesTs"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_train_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)
    os.makedirs(out_test_dir, exist_ok=True)

    return out_dir, out_train_dir, out_labels_dir, out_test_dir


def create_ACDC_split(labelsTr_folder: str, seed: int = 1234) -> List[dict[str, List]]:
    # labelsTr_folder = '/home/isensee/drives/gpu_data_root/OE0441/isensee/nnUNet_raw/nnUNet_raw_remake/Dataset027_ACDC/labelsTr'
    nii_files = nifti_files(labelsTr_folder, join=False)
    patients = np.unique([i[:len('patient000')] for i in nii_files])
    rs = np.random.RandomState(seed)
    rs.shuffle(patients)
    splits = []
    for fold in range(5):
        val_patients = patients[fold::5]
        train_patients = [i for i in patients if i not in val_patients]
        val_cases = [i[:-7] for i in nii_files for j in val_patients if i.startswith(j)]
        train_cases = [i[:-7] for i in nii_files for j in train_patients if i.startswith(j)]
        splits.append({'train': train_cases, 'val': val_cases})
    return splits


def copy_files(src_data_folder: Path, train_dir: Path, labels_dir: Path, test_dir: Path):
    """Copy files from the ACDC dataset to the nnUNet dataset folder. Returns the number of training cases."""
    patients_train = sorted([f for f in (src_data_folder / "training").iterdir() if f.is_dir()])
    patients_test = sorted([f for f in (src_data_folder / "testing").iterdir() if f.is_dir()])

    num_training_cases = 0
    # Copy training files and corresponding labels.
    for patient_dir in patients_train:
        for file in patient_dir.iterdir():
            if file.suffix == ".gz" and "_gt" not in file.name and "_4d" not in file.name:
                # The stem is 'patient.nii', and the suffix is '.gz'.
                # We split the stem and append _0000 to the patient part.
                shutil.copy(file, train_dir / f"{file.stem.split('.')[0]}_0000.nii.gz")
                num_training_cases += 1
            elif file.suffix == ".gz" and "_gt" in file.name:
                shutil.copy(file, labels_dir / file.name.replace("_gt", ""))

    # Copy test files.
    for patient_dir in patients_test:
        for file in patient_dir.iterdir():
            if file.suffix == ".gz" and "_gt" not in file.name and "_4d" not in file.name:
                shutil.copy(file, test_dir / f"{file.stem.split('.')[0]}_0000.nii.gz")

    return num_training_cases

'''
def copy_files(src_data_folder: Path, train_dir: Path, labels_dir: Path, test_dir: Path):
    """Copy files from the ACDC dataset to the nnUNet dataset folder. Returns the number of training cases."""
    patients_train = sorted([f for f in (src_data_folder / "training").iterdir() if f.is_dir()])
    patients_test = sorted([f for f in (src_data_folder / "testing").iterdir() if f.is_dir()])

    num_training_cases = 0
    # Copy training files and corresponding labels.
    for patient_dir in patients_train:
        for file in patient_dir.iterdir():
            if file.suffix == ".gz" and "_gt" not in file.name and "_4d" not in file.name:
                # The stem is 'patient.nii', and the suffix is '.gz'.
                # We split the stem and append _0000 to the patient part.
                shutil.copy(file, train_dir / f"{file.stem.split('.')[0]}_0000.nii.gz")
                num_training_cases += 1
            elif file.suffix == ".gz" and "_gt" in file.name:
                shutil.copy(file, labels_dir / file.name.replace("_gt", ""))

    # Copy test files.
    create_test_cases(src_data_folder, test_dir)

    return num_training_cases
'''
'''
def create_test_cases(src_data_folder: Path, test_dir: Path,  target_length: int = 20):
    # For every 4d volume in the train set, create T frames from it and copy them to the test set for inference. 
    patients_train = sorted([f for f in (src_data_folder / "training").iterdir() if f.is_dir()])
    for patient_dir in patients_train:
        for file in patient_dir.iterdir():
            if file.suffix=="cfg":
                # parse the cfg file to get the ES frame index
                with open(file, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith("ES"):
                            ES_index = int(line.split("=")[1].strip())
                            break
            if file.suffix == ".gz" and "_4d" in file.name: 
                # The stem is 'patient_4d.nii', and the suffix is '.gz'.
                # We split the stem and append _0000 to the patient part.
                patient_name = file.stem.split('.')[0].replace('_4d', '')

                # Load the 4D volume 
                img = nib.load(file)
                
                # Get the interpolated frames 
                frames = interpolate_4d_volumes(img, target_length, ES=ES_index)  # List of Nifti images, each is a frame
                
                # Save each frame as a separate file in the test_dir
                for t, frame_img in enumerate(frames):
                    nib.save(frame_img, test_dir / f"{patient_name}_frame{t:04d}_0000.nii.gz")
'''
'''
def interpolate_4d_volumes(patient_img : nib.Nifti1Image, target_length:int, ES:int) -> List[nib.Nifti1Image]:
    # Interpolate the 4D volume to have target_length frames. 
    data = patient_img.get_fdata()  # shape [X, Y, Z, T]
    current_length = data.shape[3]
    if current_length == target_length:
        return [nib.Nifti1Image(data[:, :, :, t], patient_img.affine, patient_img.header) for t in range(current_length)]
    
    # Interpolate along the time dimension, keeping the ES frame fixed.


    interpolated_imgs = []
    new_ES_index = int(round((target_length - 1) * ES / (current_length - 1)))
    new_clip = np.zeros((data.shape[0], data.shape[1], data.shape[2],target_length), dtype=data.dtype)  # [X, Y, Z, T]
    T = current_length

    # left part: before ES
    if new_ES_index > 0:
        new_grid_1 = np.linspace(0, ES - 1, new_ES_index)
        for i in range(new_ES_index):
            m = new_grid_1[i]
            if float(m).is_integer():
                new_clip[..., i] = data[ ..., int(m)]
            else:
                i_prev = int(np.floor(m))
                i_prox = int(np.ceil(m))
                alpha = m - i_prev
                new_clip[..., i] = data[..., i_prev] + alpha * (data[..., i_prox] - data[..., i_prev])

    # ES exactly preserved
    new_clip[..., new_ES_index] = data[..., ES]

    # right part: from ES onward
    right_len = target_length - new_ES_index
    if right_len > 1:
        new_grid_2 = np.linspace(ES, T - 1, right_len)
        for j in range(1, right_len):   # start from 1 because j=0 is ES itself
            m = new_grid_2[j]
            i = new_ES_index + j
            if float(m).is_integer():
                new_clip[..., i] = data[..., int(m)]
            else:
                i_prev = int(np.floor(m))
                i_prox = int(np.ceil(m))
                alpha = m - i_prev
                new_clip[..., i] = data[..., i_prev] + alpha * (data[..., i_prox] - data[..., i_prev])

    # ES exactly preserved
    new_clip[..., new_ES_index] = data[..., ES]

    # right part: from ES onward
    right_len = target_length - new_ES_index
    if right_len > 1:
        new_grid_2 = np.linspace(ES, T - 1, right_len)
        for j in range(1, right_len):   # start from 1 because j=0 is ES itself
            m = new_grid_2[j]
            i = new_ES_index + j
            if float(m).is_integer():
                new_clip[..., i] = data[..., int(m)]
            else:
                i_prev = int(np.floor(m))
                i_prox = int(np.ceil(m))
                alpha = m - i_prev
                new_clip[..., i] = data[..., i_prev] + alpha * (data[..., i_prox] - data[..., i_prev])

    # Convert the interpolated clip back to a list of Nifti images, one for each frame.          
    for t in range(target_length):
        interpolated_imgs.append(nib.Nifti1Image(new_clip[..., t], patient_img.affine, patient_img.header))
    
    
    return interpolated_imgs

'''

def convert_acdc(src_data_folder: str, dataset_id=27):
    out_dir, train_dir, labels_dir, test_dir = make_out_dirs(dataset_id=dataset_id)
    num_training_cases = copy_files(Path(src_data_folder), train_dir, labels_dir, test_dir)

    generate_dataset_json(
        str(out_dir),
        channel_names={
            0: "cineMRI",
        },
        labels={
            "background": 0,
            "RV": 1,
            "MLV": 2,
            "LVC": 3,
        },
        file_ending=".nii.gz",
        num_training_cases=num_training_cases,
    )

@hydra.main(version_base=None, config_path="src/configs/datasets", config_name="preprocess")
def main(cfg: DictConfig) -> None:
    print("Converting...")

    convert_acdc(cfg.input_folder, cfg.dataset_id)

    dataset_name = f"Dataset{cfg.dataset_id:03d}_{cfg.task_name}"
    labelsTr = join(nnUNet_raw, dataset_name, "labelsTr")
    preprocessed_folder = join(nnUNet_preprocessed, dataset_name)

    maybe_mkdir_p(preprocessed_folder)

    split = create_ACDC_split(labelsTr, seed=cfg.split_seed)
    save_json(split, join(preprocessed_folder, "splits_final.json"), sort_keys=False)

    print("Done!")


if __name__ == "__main__":
    main()
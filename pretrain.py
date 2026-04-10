import hydra
from omegaconf import DictConfig
import subprocess

from pathlib import Path
from typing import List, Tuple, Dict
import json
import shutil

import nibabel as nib
import numpy as np


def parse_es_from_info_cfg(cfg_path: Path) -> int:
    """
    Parse ACDC Info.cfg and return ES as a 1-based frame index.
    Supports both 'ES: 12' and 'ES=12'.
    Assumes ED is always 1.
    """
    es = None

    with open(cfg_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if ":" in line:
                key, value = [x.strip() for x in line.split(":", 1)]
            elif "=" in line:
                key, value = [x.strip() for x in line.split("=", 1)]
            else:
                continue

            if key == "ES":
                es = int(value)
                break

    if es is None:
        raise ValueError(f"Could not parse ES from {cfg_path}")

    return es


def interpolate_4d_volumes_keep_ed_es(
    patient_img: nib.Nifti1Image,
    target_length: int,
    es_index_1based: int
) -> Tuple[List[nib.Nifti1Image], int, int]:
    """
    Interpolate a 4D cine volume to target_length while preserving:
      - ED exactly at index 1
      - ES exactly at its mapped position in the new timeline

    Assumption:
      - ED is always the first frame in ACDC, i.e. original ED index = 1.

    Args:
        patient_img: 4D nib image with shape [X, Y, Z, T]
        target_length: desired number of frames after interpolation
        es_index_1based: original ES frame index, 1-based

    Returns:
        interpolated_imgs: list of 3D Nifti images of length target_length
        new_ed_index_1based: always 1
        new_es_index_1based: ES location in the interpolated timeline
    """
    data = patient_img.get_fdata()  # shape [X, Y, Z, T]
    if data.ndim != 4:
        raise ValueError(f"Expected 4D image, got shape {data.shape}")

    T = data.shape[3]
    if T < 2:
        raise ValueError("Cannot interpolate a clip with fewer than 2 frames")

    if target_length < 2:
        raise ValueError("target_length must be at least 2")

    ed0 = 0  # ED is always first frame
    es0 = es_index_1based - 1

    if not (0 <= es0 < T):
        raise ValueError(f"Invalid ES index {es_index_1based} for clip with {T} frames")

    if target_length == T:
        frames = []
        for t in range(T):
            frame_data = data[..., t]
            header = patient_img.header.copy()
            header.set_data_shape(frame_data.shape)
            frames.append(nib.Nifti1Image(frame_data, patient_img.affine, header))
        return frames, 1, es_index_1based

    new_clip = np.zeros((*data.shape[:3], target_length), dtype=data.dtype)

    # ED remains fixed at the first frame
    new_ed0 = 0

    # Map ES into the new timeline
    new_es0 = int(round((target_length - 1) * es0 / (T - 1)))

    if new_es0 <= new_ed0:
        raise ValueError(
            f"Mapped ES index ({new_es0 + 1}) is not after ED. "
            f"Check target_length={target_length} and ES={es_index_1based}."
        )

    # Segment 1: ED -> ES
    left_grid = np.linspace(ed0, es0, new_es0 + 1)
    for i, m in enumerate(left_grid):
        if float(m).is_integer():
            new_clip[..., i] = data[..., int(m)]
        else:
            i_prev = int(np.floor(m))
            i_next = int(np.ceil(m))
            alpha = m - i_prev
            new_clip[..., i] = data[..., i_prev] + alpha * (data[..., i_next] - data[..., i_prev])

    # Enforce exact ED and ES
    new_clip[..., new_ed0] = data[..., ed0]
    new_clip[..., new_es0] = data[..., es0]

    # Segment 2: ES -> end of cycle
    right_len = target_length - new_es0
    if right_len > 1:
        right_grid = np.linspace(es0, T - 1, right_len)
        for j in range(1, right_len):  # j=0 is ES, already fixed
            m = right_grid[j]
            i = new_es0 + j
            if float(m).is_integer():
                new_clip[..., i] = data[..., int(m)]
            else:
                i_prev = int(np.floor(m))
                i_next = int(np.ceil(m))
                alpha = m - i_prev
                new_clip[..., i] = data[..., i_prev] + alpha * (data[..., i_next] - data[..., i_prev])

    interpolated_imgs = []
    for t in range(target_length):
        frame_data = new_clip[..., t]
        header = patient_img.header.copy()
        header.set_data_shape(frame_data.shape)
        interpolated_imgs.append(nib.Nifti1Image(frame_data, patient_img.affine, header))

    return interpolated_imgs, 1, new_es0 + 1

def reconfigure_acdc(
    final_out_dir: Path,
    test_dir_pp: Path,
    interp_frames_dir: Path,
    input_dir: Path,
    num_patients: int | None = None,
):
    """
    Create a final directory with one folder per patient:
        patientXXX/
            frames/
            labels/

    The sequence is the full interpolated cardiac cycle for each patient.

    frames/: full temporal sequence in order
    labels/: labels in the exact same temporal order
    """
    final_out_dir.mkdir(parents=True, exist_ok=True)

    training_dir = input_dir / "training"
    if not training_dir.exists():
        raise FileNotFoundError(f"Training folder not found: {training_dir}")

    metadata_path = interp_frames_dir / "interpolation_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    patients_train = sorted([p for p in training_dir.iterdir() if p.is_dir()])
    if num_patients is not None:
        patients_train = patients_train[:num_patients]

    for patient_path in patients_train:
        patient_name = patient_path.name  # e.g. patient001

        if patient_name not in metadata:
            raise KeyError(f"{patient_name} not found in interpolation metadata")

        patient_meta = metadata[patient_name]

        original_ed_idx = int(patient_meta["original_ed_index_1based"])
        original_es_idx = int(patient_meta["original_es_index_1based"])
        new_ed_idx = int(patient_meta["new_ed_index_1based"])
        new_es_idx = int(patient_meta["new_es_index_1based"])
        target_length = int(patient_meta["target_length"])

        # temporary workaround for patient090
        effective_original_ed_idx = 4 if patient_name == "patient090" else original_ed_idx

        patient_dir = final_out_dir / patient_name
        frames_dir = patient_dir / "frames"
        labels_dir = patient_dir / "labels"
        frames_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for t in range(1, target_length + 1):
            out_frame = frames_dir / f"{patient_name}_iframe{t:04d}.nii.gz"
            out_label = labels_dir / f"{patient_name}_iframe{t:04d}.nii.gz"

            if t == new_ed_idx:
                frame_src = patient_path / f"{patient_name}_frame{effective_original_ed_idx:02d}.nii.gz"
                label_src = patient_path / f"{patient_name}_frame{effective_original_ed_idx:02d}_gt.nii.gz"

            elif t == new_es_idx:
                frame_src = patient_path / f"{patient_name}_frame{original_es_idx:02d}.nii.gz"
                label_src = patient_path / f"{patient_name}_frame{original_es_idx:02d}_gt.nii.gz"

            else:
                frame_src = interp_frames_dir / f"{patient_name}_iframe{t:04d}_0000.nii.gz"
                label_src = test_dir_pp / f"{patient_name}_iframe{t:04d}.nii.gz"

            if not frame_src.exists():
                raise FileNotFoundError(f"Missing frame source: {frame_src}")
            if not label_src.exists():
                raise FileNotFoundError(f"Missing label source: {label_src}")

            shutil.copy(frame_src, out_frame)
            shutil.copy(label_src, out_label)


def create_test_files(
    src_data_folder: Path,
    test_dir: Path,
    target_length: int = 20,
    clear_output_dir: bool = False,
    save_metadata: bool = True
) -> None:
    """
    Create nnU-Net inference inputs from ACDC training 4D cine volumes.

    For each patient:
      - read ES from Info.cfg
      - assume ED = 1
      - interpolate the full cycle to target_length while preserving ED and ES
      - save only non-ED/non-ES interpolated frames as separate 3D files:
            patientXXX_iframe0002_0000.nii.gz

    The output naming follows nnU-Net single-channel inference convention.
    """
    training_dir = src_data_folder / "training"
    if not training_dir.exists():
        raise FileNotFoundError(f"Training folder not found: {training_dir}")

    if clear_output_dir and test_dir.exists():
        shutil.rmtree(test_dir)

    test_dir.mkdir(parents=True, exist_ok=True)

    metadata: Dict[str, Dict] = {}
    patients_train = sorted([f for f in training_dir.iterdir() if f.is_dir()])

    for patient_dir in patients_train:
        info_cfg = patient_dir / "Info.cfg"
        if not info_cfg.exists():
            raise FileNotFoundError(f"Missing Info.cfg in {patient_dir}")

        es_index = parse_es_from_info_cfg(info_cfg)

        fourd_files = list(patient_dir.glob("*_4d.nii.gz"))
        if len(fourd_files) != 1:
            raise ValueError(
                f"Expected exactly one *_4d.nii.gz in {patient_dir}, found {len(fourd_files)}"
            )

        fourd_path = fourd_files[0]
        patient_name = fourd_path.name.replace("_4d.nii.gz", "")

        img = nib.load(str(fourd_path))
        original_num_frames = img.shape[3]

        frames, new_ed_index, new_es_index = interpolate_4d_volumes_keep_ed_es(
            patient_img=img,
            target_length=target_length,
            es_index_1based=es_index
        )

        patient_meta = {
            "original_num_frames": int(original_num_frames),
            "target_length": int(target_length),
            "original_ed_index_1based": 1,
            "original_es_index_1based": int(es_index),
            "new_ed_index_1based": int(new_ed_index),
            "new_es_index_1based": int(new_es_index),
            "exported_frames": []
        }

        for t, frame_img in enumerate(frames, start=1):
            if t in {new_ed_index, new_es_index}:
                continue

            out_name = f"{patient_name}_iframe{t:04d}_0000.nii.gz"
            out_path = test_dir / out_name
            nib.save(frame_img, str(out_path))

            patient_meta["exported_frames"].append({
                "file_name": out_name,
                "interpolated_frame_index_1based": int(t)
            })

        metadata[patient_name] = patient_meta

    if save_metadata:
        metadata_path = test_dir / "interpolation_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"Created nnU-Net inference inputs in: {test_dir}")


@hydra.main(version_base=None, config_path="src/configs/model", config_name="pretrain")
def main(cfg: DictConfig):
    """
    Expected config fields:
      cfg.acdc_training_root
      cfg.inference_input_dir
      cfg.target_length
      cfg.clear_inference_input_dir   (optional, bool)
      cfg.save_interpolation_metadata (optional, bool)
      cfg.script.path 
      cfg.set_test (bool)
    """

    # consider all the 4d volumes in the training samples, and for each volume extract the intermediated frames (no ED/ES)
    # interpolate intermediate frames so that for each patient there is a fixed number of frames 
    # adapt these frames as test case according to inference U-Net format
    if(cfg.set_test) : 
        create_test_files(
        src_data_folder=Path(cfg.acdc_training_root),
        test_dir=Path(cfg.test_dir),
        target_length=int(cfg.target_length),
        clear_output_dir=bool(getattr(cfg, "clear_inference_input_dir", True)),
        save_metadata=bool(getattr(cfg, "save_interpolation_metadata", True)),
    )

    # run inference on intermediate frames 

    #script_path = Path(cfg.script.path).resolve()

    #subprocess.run(
    #    ["bash", str(script_path)],
    #    check=True
    #)

    # reconfigure dataset for training
    reconfigure_acdc(Path(cfg.final_dataset_directory), Path(cfg.test_dir_pp), Path(cfg.test_dir), Path(cfg.acdc_training_root), int(cfg.num_patients)) 




if __name__ == "__main__":
    main()
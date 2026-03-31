import os
import glob
import re
import numpy as np
import json 
import shutil
from pathlib import Path

def load_acdc_info(acdc_training_root, out_path, seed=0, train_ratio=0.9):

    ''' 
    Splits randomly the training dataset in train and validation set, create a dict containing
    train ids and val ids
    infos = load_acdc_info(
    "/home/renku/work/s3-bucket/ACDC/training",
    "/home/renku/work/s3-bucket/ACDC/ACDC_info.npy",
    seed=0
    ) '''
    # patient folders: .../training/patient001 ... patient100
    patient_dirs = sorted(glob.glob(os.path.join(acdc_training_root, "patient[0-9][0-9][0-9]")))
    if len(patient_dirs) == 0:
        raise FileNotFoundError(f"No patientXXX folders found under {acdc_training_root}")
    # json 
    with open("ACDC_info.json", "r") as f : 
        data = json.load(f)
    # patient ids
    cases = []
    for d in patient_dirs:
        m = re.search(r"patient(\d+)$", os.path.basename(d))
        if m:
            pid = m.group(1)  # "001"
            cases.append((pid, os.path.abspath(d)))

    rng = np.random.RandomState(seed)
    idx = np.arange(len(cases))
    rng.shuffle(idx)

    n_train = int(train_ratio * len(cases))
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    def build_dict(indices):
        d = {}
        for i in indices:
            pid, pdir = cases[i]
            d[pid] = {
                "patient_id": pid,
                "patient_dir": pdir,
                "ED" : data[pid]["ED"],
                "ES" : data[pid]["ES"],
                "group" : data[pid]["Group"], 
                "n_frames" : data[pid]["NbFrame"]
            }
        return d

    infos = {"train": build_dict(train_idx), "test": build_dict(test_idx)}
    np.save(out_path, infos, allow_pickle=True)
    print(f"Saved {out_path}: train={len(infos['train'])}, test={len(infos['test'])}")
    return infos


def convert_acdc_to_nnunet( acdc_training_root, out_dataset_root, acdc_info_json, copy_intermediate_to_imagesTs=False):
    """
    Convert ACDC training set into nnU-Net raw dataset format.
    
    Creates:
      out_dataset_root/
        dataset.json
        imagesTr/
        labelsTr/
        imagesTs/   (optional)

    Assumes:
      patientXXX/
        patientXXX_frameYY.nii.gz
        patientXXX_frameYY_gt.nii.gz
        Info.cfg or equivalent info already parsed into acdc_info_json
    """
    out_dataset_root = Path(out_dataset_root)
    imagesTr = out_dataset_root / "imagesTr"
    labelsTr = out_dataset_root / "labelsTr"
    imagesTs = out_dataset_root / "imagesTs"

    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)
    if copy_intermediate_to_imagesTs:
        imagesTs.mkdir(parents=True, exist_ok=True)

    with open(acdc_info_json, "r") as f:
        info = json.load(f)

    patient_dirs = sorted(glob.glob(os.path.join(acdc_training_root, "patient[0-9][0-9][0-9]")))
    if not patient_dirs:
        raise FileNotFoundError(f"No patientXXX folders found under {acdc_training_root}")

    n_training_cases = 0

    for pdir in patient_dirs:
        pid_match = re.search(r"patient(\d+)$", os.path.basename(pdir))
        if pid_match is None:
            continue
        pid = pid_match.group(1)  # "001"
        patient_name = f"patient{pid}"

        ed = int(info[pid]["ED"])
        es = int(info[pid]["ES"])

        # ED case - copy image and label to imagesTr and labelsTr
        ed_img_src = Path(pdir) / f"{patient_name}_frame{ed:02d}.nii.gz"
        ed_lbl_src = Path(pdir) / f"{patient_name}_frame{ed:02d}_gt.nii.gz"

        ed_case_id = f"{patient_name}_ED"
        ed_img_dst = imagesTr / f"{ed_case_id}_0000.nii.gz"
        ed_lbl_dst = labelsTr / f"{ed_case_id}.nii.gz"

        shutil.copy2(ed_img_src, ed_img_dst)
        shutil.copy2(ed_lbl_src, ed_lbl_dst)
        n_training_cases += 1

        # ES case - copy image and label to imagesTr and labelsTr
        es_img_src = Path(pdir) / f"{patient_name}_frame{es:02d}.nii.gz"
        es_lbl_src = Path(pdir) / f"{patient_name}_frame{es:02d}_gt.nii.gz"

        es_case_id = f"{patient_name}_ES"
        es_img_dst = imagesTr / f"{es_case_id}_0000.nii.gz"
        es_lbl_dst = labelsTr / f"{es_case_id}.nii.gz"

        shutil.copy2(es_img_src, es_img_dst)
        shutil.copy2(es_lbl_src, es_lbl_dst)
        n_training_cases += 1

        # Optional: intermediate frames for later inference 
        if copy_intermediate_to_imagesTs:
            n_frames = int(info[pid]["NbFrame"])
            for fr in range(1, n_frames + 1):
                if fr in (ed, es):
                    continue
                src = Path(pdir) / f"{patient_name}_frame{fr:02d}.nii.gz"
                if src.exists():
                    case_id = f"{patient_name}_frame{fr:02d}"
                    dst = imagesTs / f"{case_id}_0000.nii.gz"
                    shutil.copy2(src, dst)

    dataset_json = {
        "channel_names": {
            "0": "MRI"
        },
        "labels": {
            "background": 0,
            "RV": 1,
            "MYO": 2,
            "LV": 3
        },
        "numTraining": n_training_cases,
        "file_ending": ".nii.gz"
    }

    with open(out_dataset_root / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"Done. Wrote dataset to: {out_dataset_root}")
    print(f"Training cases: {n_training_cases}")



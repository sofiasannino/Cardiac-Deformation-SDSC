import os
import glob
import re
import numpy as np

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
            }
        return d

    infos = {"train": build_dict(train_idx), "test": build_dict(test_idx)}
    np.save(out_path, infos, allow_pickle=True)
    print(f"Saved {out_path}: train={len(infos['train'])}, test={len(infos['test'])}")
    return infos



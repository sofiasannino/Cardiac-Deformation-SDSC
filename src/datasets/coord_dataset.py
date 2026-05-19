from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Union, Tuple, Optional

import json
import warnings

import SimpleITK as sitk
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from batchgenerators.dataloading.data_loader import DataLoader


class nnUNetBaseDataset(ABC):
    """
    Defines the interface
    """
    def __init__(self, folder: str, identifiers: List[str] = None,):
        super().__init__()
        
        self.source_folder = folder
        if identifiers is None:
            identifiers = self.get_identifiers(folder)
        self.identifiers = identifiers

    def __getitem__(self, identifier):
        return self.load_case(identifier)

    @abstractmethod
    def load_case(self, identifier):
        pass

    @staticmethod
    @abstractmethod
    def save_case(
            data: np.ndarray,
            seg: np.ndarray,
            properties: dict,
            output_filename_truncated: str
            ):
        pass

    @staticmethod
    @abstractmethod
    def get_identifiers(folder: str) -> List[str]:
        pass

    @staticmethod
    def unpack_dataset(folder: str, overwrite_existing: bool = False,
                       num_processes: int = 1,
                       verify: bool = True):
        pass





class nnUNetDatasetCoord(nnUNetBaseDataset):
    def __init__(self, folder: str, identifiers: Optional[Dict[str, Dict[str, str]]] = None, labels : Tuple[int, ...] = (1, 2, 3), points_per_label : int = 1562,
                 max_num_patients: Optional[int] = None, num_frames_per_patient: int = 20):
        super().__init__(folder, identifiers)
        self.labels = labels # labels 
        self.K = points_per_label*3 # total number of control points
        if max_num_patients is not None:
            self.identifiers = dict(list(self.identifiers.items())[:(max_num_patients*num_frames_per_patient)])
        
    def __getitem__(self, identifier):
        return self.load_case(identifier)

    def load_case(self, identifier):
        entry = self.identifiers[identifier]

        # load frame 
        frame_file = entry["frame"]
        frame = sitk.ReadImage(frame_file)
        data = sitk.GetArrayFromImage(frame).astype(np.float32)  # [D, H, W]
        data = data[None, ...]  # [1, D, H, W]

        # z-score normalization
        mean = data.mean()
        std = data.std()
        if std > 0:
            data = (data - mean) / std


        # load coordinates
        coords = self.load_coords(entry["coords"]) # [K, 3] # still [x, y, z]

        
        if coords.shape[0] != self.K: 
            raise RuntimeError("not right number of coords")
        
        
        properties = {
        "spacing": frame.GetSpacing(),
        "origin": frame.GetOrigin(),
        "direction": frame.GetDirection(),
        "size": frame.GetSize(),
    }

        return data, coords, properties

    def load_coords(self, coords_file):
        coord_list = [] 
        with open(file=coords_file, mode='r') as f: 
                coords_dict = json.load(f)
        
        for label in self.labels:
            label_key = str(label)

            points_per_label = coords_dict[label_key]
            for item in points_per_label :  
                point = np.asarray(item["point"], dtype=np.float32) # directions already create the order
                if len(point) != 3: 
                    RuntimeError("point has not 3 coordinates")
                coord_list.append(point)
        coords = np.stack(coord_list, axis=0).astype(np.float32) 
        return coords
    

    @staticmethod
    def save_case(data, seg, properties, output_filename_truncated):
        raise NotImplementedError("Coordinate dataset does not save cases for now")
   

    @staticmethod
    def get_identifiers(folder: str) -> Dict[str, Dict[str, str]]:
        data_path = Path(folder)

        # load patients with bad segmentations
        bad_seg_path = data_path / "bad_seg.json"
        if bad_seg_path.exists():
            with open(bad_seg_path, "r", encoding="utf-8") as f:
                bad_seg = json.load(f)

            # assumes structure like {"patient096": [...], "patient012": [...]}
            bad_patients = set(bad_seg.keys())
        else:
            bad_patients = set()

        identifiers: Dict[str, Dict[str, str]] = {}

        # this id counts only valid, non-skipped entries
        current_id = 0

        patients_train = sorted([p for p in data_path.iterdir() if p.is_dir()])

        for patient_path in patients_train:
            patient_name = patient_path.name

            # skip patients listed in bad_seg.json
            if patient_name in bad_patients:
                print(f"Skipping {patient_name}: listed in bad_seg.json")
                continue

            frames_path = patient_path / "frames"
            coords_path = patient_path / "coords"

            if not frames_path.exists():
                print(f"Skipping {patient_name}: missing frames folder")
                continue

            if not coords_path.exists():
                print(f"Skipping {patient_name}: missing coords folder")
                continue

            frames = sorted(
                list(frames_path.glob("*.nii.gz")) +
                list(frames_path.glob("*.nii"))
            )

            for frame_path in frames:
                if frame_path.name.endswith(".nii.gz"):
                    frame_id = frame_path.name.replace(".nii.gz", "")
                else:
                    frame_id = frame_path.stem

                coord_path = coords_path / f"coords_{frame_id}.json"

                if not coord_path.exists():
                    print(f"Missing coords for frame {frame_path.name}: expected {coord_path}")
                    continue

                # key includes patient name + progressive id after skipped patients
                key = f"{patient_name}_{current_id:05d}"

                identifiers[key] = {
                    "patient": patient_name,
                    "frame_id": frame_id,
                    "frame": str(frame_path),
                    "coords": str(coord_path),
                }

                current_id += 1

        return identifiers

    
### Dataloader 

class nnUNetDataLoaderCoord(DataLoader):
    def __init__(self,
                 data: nnUNetDatasetCoord,
                 batch_size: int,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 transforms=None):

        super().__init__(data, 1 , 1, None, True,
                         False , True , sampling_probabilities) # batch_size = 1


        # this is used by DataLoader for sampling train cases!
        self.indices = sorted(list(data.identifiers.keys()))
        self.transforms = transforms


    def generate_train_batch(self):

        selected_keys = self.get_indices()
        # preallocate output tensors in final patch size and write transformed samples directly
        data_all = None
        coords_all = None

        with torch.no_grad():
            with threadpool_limits(limits=1, user_api=None):
                for j, key in enumerate(selected_keys):

                    data, coords, properties = self._data.load_case(key)
                    # data must already be numpy 
                    data = np.asarray(data, dtype=np.float32) # [C, D, H, W]

                    
                    coords = np.asarray(coords, dtype=np.float32) # [x, y, z]

                    if coords.ndim != 2 or coords.shape[1] != 3:
                        raise RuntimeError(f"Expected coords shape [K, 3], got {coords.shape} for {key}")

                    # convert [x, y, z] to  [z, y, x]
                    coords = coords[:, [2, 1, 0]]

                    # normalize coords in [0, 1] using data shape [C, D, H, W]
                    C, D, H, W = data.shape

                    coords[:, 0] = coords[:, 0] / (D - 1)
                    coords[:, 1] = coords[:, 1] / (H - 1)
                    coords[:, 2] = coords[:, 2] / (W - 1)

                    data_sample = torch.from_numpy(data).float()
                    coords_sample = torch.from_numpy(coords).float()

                    if self.transforms is not None:
                        # Only safe for intensity-only transforms unless coords are also transformed.
                        transformed = self.transforms(**{"image": data_sample})
                        data_sample = transformed["image"]

                    if data_all is None:
                        data_all = torch.empty(
                            (self.batch_size, *data_sample.shape),
                            dtype=torch.float32,
                        )

                        coords_all = torch.empty(
                            (self.batch_size, *coords_sample.shape),
                            dtype=torch.float32,
                        )

                    #if data_sample.shape != data_all.shape[1:]:
                        #raise RuntimeError(
                        #    f"Shape mismatch in batch. First sample shape={data_all.shape[1:]}, "
                        #    f"current sample {key} shape={data_sample.shape}. "
                        #    f"Use batch_size=1, padding, cropping, or resampling.")

                    data_all[j] = data_sample
                    coords_all[j] = coords_sample
                    
        return {'data': data_all, 'coords': coords_all, 'keys': selected_keys}

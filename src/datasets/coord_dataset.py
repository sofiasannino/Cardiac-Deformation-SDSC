import os
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Union, Type, Tuple
import SimpleITK as sitk 
import numpy as np
import math
from pathlib import Path
import json


class nnUNetBaseDataset(ABC):
    """
    Defines the interface
    """
    def __init__(self, folder: str, identifiers: List[str] = None,):
        super().__init__()
        # print('loading dataset')
        if identifiers is None:
            identifiers = self.get_identifiers(folder)
        identifiers.sort()

        self.source_folder = folder
        self.folder_with_segs_from_previous_stage = folder_with_segs_from_previous_stage
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
                       num_processes: int = default_num_processes,
                       verify: bool = True):
        pass





class nnUNetDatasetCoord(nnUNetBaseDataset):
    def __init__(self, folder: str, identifiers: Dict[str, Dict[str, str]] = None):
        super().__init__(folder, identifiers)
        self.labels = [1, 2, 3] # labels 
        self.K = 1562*3 # total number of control points
        if identifiers is not None : 
            self.identifiers = identifiers
        else :
            self.identifiers = self.get_identifiers(folder=folder)
        

    def __getitem__(self, identifier):
        return self.load_case(identifier)

    def load_case(self, identifier):
        entry = self.identifiers[identifier]

        # load frame 
        frame_file = entry["frame"]
        frame = sitk.ReadImage(frame_file)

        # load coordinates
        coords_list = self.load_coords(entry["coords"])
        coords = np.stack(coord_list, axis=0).astype(np.float32)  # [K, 3] # still [x, y, z]

        if coords.shape[0] != self.K: 
            RuntimeError("not right number of coords")

        return frame, coords
    def load_coords(coords_file):
        coord_list = [] 
        with open(file=coords_file, mode='r') as f: 
                coords_dict = json.load(f)
        
        for label in self.labels:
            label_key = str(label)

            points_per_label = coord_dict[label_key]
            for item in self.points_per_label :  
                point = np.asarray(item["point"], dtype=np.float32)
                if len(point) != 3: 
                    RuntimeError("point has not 3 coordinates")
                coord_list.append(point)
        return coord_list

    @staticmethod
    def save_case(data, seg, properties, output_filename_truncated):
        raise NotImplementedError("Coordinate dataset does not save cases.")

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

    @staticmethod
    def unpack_dataset(folder: str, overwrite_existing: bool = False, num_processes: int = 1, verify: bool = True):
        pass

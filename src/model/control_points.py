from tracemalloc import start

from sympy import centroid

import SimpleITK as sitk
import numpy as np
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import hydra
from omegaconf import DictConfig
import warnings
from tqdm import tqdm
import os
import logging
from copy import deepcopy
import matplotlib.pyplot as plt

from control_points_utils import test_control_points_2d_3d
from control_points_utils import _serialize_points,  _frame_name, plot_anchor_first_alignment

Point3D = Tuple[float, float, float]
Direction = Tuple[float, float, float]

logger = logging.getLogger(__name__)


class ControlPoints:
    def __init__(self, cfg: DictConfig):
        self.labels = cfg.labels
        self.theta_num = cfg.theta_num
        self.phi_num = cfg.phi_num
        self.max_iter = cfg.iter
        self.step = cfg.step
        self.extra_iter = cfg.extra_iter 

        self.directions: List[Direction] = self.GetSphericalDirections()

        # points[label] = [(point, direction), ...]
        self.points: Dict[int, List[Tuple[Point3D, Direction]]] = {
            label: [] for label in self.labels
        }
        logger.info(
            "Initialized ControlPoints with labels=%s, theta_num=%s, phi_num=%s, max_iter=%s, step=%s, num_directions=%s",
            self.labels, self.theta_num, self.phi_num, self.max_iter, self.step, len(self.directions)
        )

    def GetPoints(self) -> Dict[int, List[Tuple[Point3D, Direction]]]:
        """Return list of points"""
        logger.debug("Returning points dictionary with labels=%s", list(self.points.keys()))
        return self.points

    def GetLabels(self): 
        return self.labels

    def TransformPointsBack(self, transform: sitk.Transform) -> None:
        """
        Transform all stored points from the anchor reference system
        back to the original image system, modifying self.points in place.

        Assumes:
        - self.points[label] = [(point, direction), ...]
        - `transform` maps ANCHOR -> ORIGINAL
        """

        logger.info("Transforming points back to original reference system")
        inverse_transform = transform # BECAUSE SITK LEARNS A TRANSFORM P_MOVING = T(P_ALIGNED)--> ALIGNED[P_ALIGNED] = MOVING[P_MOVING]

        for label, points_with_dirs in tqdm(
            self.points.items(),
            desc="Transforming control points back",
            leave=False
        ):
            transformed_points = []

            for p, d in points_with_dirs:
                transformed_p = inverse_transform.TransformPoint(tuple(map(float, p)))
                transformed_points.append((transformed_p, d))

            self.points[label] = transformed_points
            logger.debug("Label %s: transformed %s points back", label, len(transformed_points))
    def ConvertPointsToContinuousIndex(self, reference_img: sitk.Image):
        """
        Convert physical control points to continuous index coordinates.
        The points must already be expressed in the physical reference system
        of reference_img
        """

        converted_points = {}

        for label in self.labels:
            converted_points[label] = []

            for point_phys, direction in self.points[label]:
                point_phys = tuple(float(x) for x in point_phys)

                # convert physical point to continuous index
                point_idx = reference_img.TransformPhysicalPointToContinuousIndex(point_phys)

                # keep SimpleITK index order: (x, y, z)
                converted_points[label].append(
                    (
                        tuple(float(x) for x in point_idx),
                        tuple(float(x) for x in direction),
                    )
                )

        self.points = converted_points

    def GetSphericalDirections(self) -> List[Direction]:
        """Sample directions on the sphere."""
        logger.info("Sampling spherical directions")

        theta = np.linspace(0, np.pi, num=self.theta_num)
        phi = np.linspace(0, 2 * np.pi, num=self.phi_num, endpoint=False)

        directions = []

        # north pole
        directions.append((0.0, 0.0, 1.0))

        # intermediate latitudes
        for t in theta[1:-1]:
            for p in phi:
                x = np.sin(t) * np.cos(p)
                y = np.sin(t) * np.sin(p)
                z = np.cos(t)
                directions.append((float(x), float(y), float(z)))

        # south pole
        directions.append((0.0, 0.0, -1.0))

        logger.info("Sampled %s spherical directions", len(directions))
        return directions

    def ExtractPointsWithGrid(self, mask: sitk.Image):
        """
        For each label:
        - compute centroid from the largest connected component
        - cast rays along spherical directions
        - find the first point entering the region of that label
        - store as (point, direction) inside points[label]

        Output format:
        self.points[label] = [(point, direction), ...]
        """

        extra_iter = 2500
        old_patient_points = self.points

        logger.info("Extracting control points with ray + fallback method")
        # reset dictionary with the new format
        self.points = {label: [] for label in self.labels}

        for label in tqdm(self.labels, desc="Processing labels in ExtractPoints"):
            logger.info("Processing label %s", label)
            # full binary mask for the current label
            class_mask = sitk.Cast(mask == label, sitk.sitkUInt8)

            # connected components
            cc = sitk.ConnectedComponent(class_mask)
            relabeled = sitk.RelabelComponent(cc)
            largest_cc = sitk.Cast(relabeled == 1, sitk.sitkUInt8)

            shape_stats = sitk.LabelShapeStatisticsImageFilter()
            shape_stats.Execute(largest_cc)

            logger.debug("Computed connected components and shape stats for label %s", label)

            # check if label exists
            if not shape_stats.HasLabel(1):
                logger.warning("This patient frame is not well segmented for label %s", label)
                logger.warning("Using previous frame control points")
                self.points = old_patient_points
                return False

            # centroid of the largest connected component
            centroid = np.array(shape_stats.GetCentroid(1), dtype=float)

            size = class_mask.GetSize()

            # contour of the full label mask, for fallback
            contour = sitk.BinaryContour(class_mask)
            contour_arr = sitk.GetArrayFromImage(contour)   # [z, y, x]
            contour_idx_zyx = np.argwhere(contour_arr > 0)
            logger.debug("Label %s: contour has %s voxels", label, len(contour_idx_zyx))

            extended_search_count = 0
            fallback_count = 0
            failed_fallback_count = 0
            for d in tqdm(self.directions, desc = f"Extracting points from label {label} in all directions", leave=False):
                direction = np.array(d, dtype=float)

                # starting rule far away from the centroid 
                point = centroid + direction * 2 * shape_stats.GetEquivalentSphericalRadius(1)

                found = False
                for _ in range(self.max_iter):
                    point_idx = tuple(int(x) for x in class_mask.TransformPhysicalPointToIndex(tuple(point)))

                    inside_image = all(0 <= point_idx[i] < size[i] for i in range(len(size)))

                # check against the full label mask
                    if inside_image and class_mask.GetPixel(point_idx) != 0:
                        self.points[label].append((tuple(float(x) for x in point), tuple(float(x) for x in direction),))
                        found = True
                        break

                    # update rule 
                    point = point - direction * self.step

                if not found: # fallback 1
                    extended_search_count += 1
                    for _ in range(extra_iter):
                        point_idx = tuple(int(x) for x in class_mask.TransformPhysicalPointToIndex(tuple(point)))

                        inside_image = all(0 <= point_idx[i] < size[i] for i in range(len(size)))

                        if inside_image and class_mask.GetPixel(point_idx) != 0:
                            self.points[label].append((tuple(float(x) for x in point), tuple(float(x) for x in direction),))
                            found = True
                            break

                        point = point - direction * self.step

                if not found: # fallback 2
                    fallback_count += 1
                    best_point = None
                    best_score = -np.inf

                    for idx_zyx in contour_idx_zyx:
                        z, y, x = idx_zyx
                        idx_xyz = (int(x), int(y), int(z))

                        p_phys = np.array(class_mask.TransformIndexToPhysicalPoint(idx_xyz),dtype=float)

                        score = np.dot(p_phys - centroid, direction)

                        if score > best_score:
                            best_score = score
                            best_point = p_phys

                    if best_point is not None:
                        self.points[label].append((tuple(float(x) for x in best_point),tuple(float(x) for x in direction),))
                    else:
                        failed_fallback_count += 1

            if extended_search_count > 0 or fallback_count > 0:
                logger.warning(
                "Label %s: used extended ray search for %s directions, contour fallback for %s directions, failed fallback for %s directions",
                label, extended_search_count, fallback_count, failed_fallback_count
            )

            logger.info(
            "Finished label %s: stored_points=%s unique_points=%s",
            label,
            len(self.points[label]),
            len(set(p for p, _ in self.points[label])) if len(self.points[label]) > 0 else 0,
        )
        return True
    def GridSearch(self, label, point, direction, class_mask, size, extra_iter):
        found = False

        point = np.array(point, dtype=float)
        direction = np.array(direction, dtype=float)

        # main search
        for _ in range(self.max_iter):
            point_idx = tuple(int(x) for x in class_mask.TransformPhysicalPointToIndex(tuple(point)))

            inside_image = all(0 <= point_idx[i] < size[i] for i in range(len(size)))

            if inside_image and class_mask.GetPixel(point_idx) != 0:
                self.points[label].append((
                    tuple(float(x) for x in point),
                    tuple(float(x) for x in direction),))
                found = True
                return found, point

            point = point - direction * self.step
        # extra search
        for _ in range(extra_iter):
            point_idx = tuple(int(x) for x in class_mask.TransformPhysicalPointToIndex(tuple(point)))

            inside_image = all(0 <= point_idx[i] < size[i] for i in range(len(size)))

            if inside_image and class_mask.GetPixel(point_idx) != 0:
                self.points[label].append((
                    tuple(float(x) for x in point),
                    tuple(float(x) for x in direction),
                ))
                found = True
                return found, point

            point = point - direction * self.step

        return found, point

    def BinarySearch(self, label, point, direction, class_mask, centroid, size, extra_iter):

        found = False

        start = np.array(centroid, dtype=float)   # should be inside the mask
        end = np.array(point, dtype=float)        # should be outside the mask
        direction = np.array(direction, dtype=float)

        ### bring end point inside the image if it is outside, moving inward along the ray direction
        max_inside_iter = self.max_iter + extra_iter
        for _ in range(max_inside_iter):
            point_idx = tuple(int(x) for x in class_mask.TransformPhysicalPointToIndex(tuple(end)))
            inside_image = all(0 <= point_idx[i] < size[i] for i in range(len(size)))
            if inside_image:
                break
            end = end - direction * self.step
        else:
        # could not bring end inside the image
            return False, end

    
        ### check that start is inside the image and mask
    
        start_idx = tuple(int(x) for x in class_mask.TransformPhysicalPointToIndex(tuple(start)))
        start_inside_image = all(0 <= start_idx[i] < size[i] for i in range(len(size)))
        if not start_inside_image:
            logger.warning("Start point is outside the image for label %s. Cannot perform binary search.", label)
            return False, end

        if class_mask.GetPixel(start_idx) == 0:
            logger.warning("Start point is outside the mask for label %s. Cannot perform binary search.", label)
            return False, end

        ### check that end is outside the mask
        end_idx = tuple(int(x) for x in class_mask.TransformPhysicalPointToIndex(tuple(end)))

        if class_mask.GetPixel(end_idx) != 0:
            # binary search needs start inside and end outside
            logger.warning("End point is inside the mask for label %s. Cannot perform binary search.", label)
            return False, end
    
        ### binary search
        for _ in range(self.max_iter):
            mid = (start + end) / 2.0

            point_idx = tuple(int(x) for x in class_mask.TransformPhysicalPointToIndex(tuple(mid)))

            if class_mask.GetPixel(point_idx) != 0:
                # mid is inside the mask, so move start outward
                start = mid
            else:
                # mid is outside the mask, so move end inward
                end = mid
        '''
        ### optional extra refinement
    
        if extra_iter > 0:
            # extended_search_count = 1

            for _ in range(extra_iter):
                mid = (start + end) / 2.0

                point_idx = tuple(int(x) for x in class_mask.TransformPhysicalPointToIndex(tuple(mid)))

                if class_mask.GetPixel(point_idx) != 0:
                    # mid is inside the mask, so move start outward
                    start = mid
                else:
                    # mid is outside the mask, so move end inward
                    end = mid
        '''
        point = start
        found = True

        self.points[label].append((tuple(float(x) for x in point),tuple(float(x) for x in direction),))

        return found, point

    def BinarySearch2(self, label, point, direction, class_mask, centroid, size, extra_iter):
        """
        Binary-search-like ray refinement using the logic:

        - Start from a point far from the centroid along `direction`.
        - If the point is outside the image, move it inward along -direction.
        - At each iteration:
            - halve the distance
            - if current point is inside the mask, move outward
            - if current point is outside the mask, move inward
        - If the final point is inside the mask, store it.
        - Otherwise, run extra refinement iterations.
        """

        found = False
        

        point = np.array(point, dtype=float)
        centroid = np.array(centroid, dtype=float)
        direction = np.array(direction, dtype=float)

        distance = np.linalg.norm(point - centroid)

        def get_index_and_inside_image(p):
            idx = tuple(
                int(x) for x in class_mask.TransformPhysicalPointToIndex(tuple(p))
            )
            inside_image = all(0 <= idx[i] < size[i] for i in range(len(size)))
            return idx, inside_image

        def is_inside_mask(p):
            idx, inside_image = get_index_and_inside_image(p)
            if not inside_image:
                return False, idx, inside_image
            return class_mask.GetPixel(idx) != 0, idx, inside_image

        # Bring starting point inside the image
    
        max_inside_iter = self.max_iter + extra_iter

        for _ in range(max_inside_iter):
            point_idx, inside_image = get_index_and_inside_image(point)

            if inside_image:
                break

            point = point - direction * self.step

        else:
            # Could not bring point inside the image
            return False, point

    
        # main binary-search-like refinement
        for _ in range(self.max_iter):
            inside_mask, point_idx, inside_image = is_inside_mask(point)

            distance = distance / 2.0

            if inside_mask:
                # if inside the mask, move outward
                point = point + direction * distance
            else:
                # if outside the mask 
                point = point - direction * distance

        inside_mask, point_idx, inside_image = is_inside_mask(point)

        if inside_mask:
            found = True
            self.points[label].append(
            (
                tuple(float(x) for x in point),
                tuple(float(x) for x in direction),
            )
        )
            return found, point


        # extra refinement if main search failed
        if extra_iter > 0:

            for _ in range(extra_iter):
                inside_mask, point_idx, inside_image = is_inside_mask(point)

                distance = distance / 2.0

                if inside_mask:
                    point = point + direction * distance
                else:
                    point = point - direction * distance

            # Recompute after the last update
            inside_mask, point_idx, inside_image = is_inside_mask(point)

            if inside_mask:
                found = True
                self.points[label].append(
                (
                    tuple(float(x) for x in point),
                    tuple(float(x) for x in direction),
                )
            )

        return found, point

    def ExtractPoints(self, mask: sitk.Image):
        """
        For each label:
        - compute centroid from the largest connected component
        - cast rays along spherical directions
        - find the first point entering the region of that label
        - store as (point, direction) inside points[label]

        Output format:
        self.points[label] = [(point, direction), ...]
        """

        extra_iter = self.extra_iter 
        old_patient_points = self.points

        logger.info("Extracting control points with ray + fallback method")
        # reset dictionary with the new format
        self.points = {label: [] for label in self.labels}

        for label in tqdm(self.labels, desc="Processing labels in ExtractPoints"):
            logger.info("Processing label %s", label)
            # full binary mask for the current label
            class_mask = sitk.Cast(mask == label, sitk.sitkUInt8)

            # connected components
            cc = sitk.ConnectedComponent(class_mask)
            relabeled = sitk.RelabelComponent(cc)
            largest_cc = sitk.Cast(relabeled == 1, sitk.sitkUInt8)

            shape_stats = sitk.LabelShapeStatisticsImageFilter()
            shape_stats.Execute(largest_cc)

            logger.debug("Computed connected components and shape stats for label %s", label)

            # check if label exists
            if not shape_stats.HasLabel(1):
                logger.warning("This patient frame is not well segmented for label %s", label)
                logger.warning("Using previous frame control points")
                self.points = old_patient_points
                return False
    
            # centroid of the largest connected component
            centroid = np.array(shape_stats.GetCentroid(1), dtype=float)
            
            # decide which method 
            centroid_idx = tuple(int(x) for x in class_mask.TransformPhysicalPointToIndex(tuple(centroid)))
            if class_mask.GetPixel(centroid_idx)!=0: 
                method = "binary"
            else :
                logger.warning("Using grid search since centroid not in mask :(") 
                method = "grid"

            size = class_mask.GetSize()
            # contour of the full label mask for fallback
            contour = sitk.BinaryContour(class_mask)
            contour_arr = sitk.GetArrayFromImage(contour)   # [z, y, x]
            contour_idx_zyx = np.argwhere(contour_arr > 0)
            logger.debug("Label %s: contour has %s voxels", label, len(contour_idx_zyx))

            grid_search_count = 0
            fallback_count = 0
            failed_fallback_count = 0
            for d in tqdm(self.directions, desc = f"Extracting points from label {label} in all directions", leave=False):
                direction = np.array(d, dtype=float)

                # starting rule: far away from the centroid 
                point = centroid + direction * 2 * shape_stats.GetEquivalentSphericalRadius(1)
                if method == "binary": # hoping for binary
                    found, best_point = self.BinarySearch(label, point, direction, class_mask, centroid, size, extra_iter)
                else:
                    found, best_point = self.GridSearch(label, point, direction, class_mask, size, extra_iter)

                if not found and method == "binary" : # fallback 1: grid search
                    grid_search_count += 1
                    point = centroid + direction * 2 * shape_stats.GetEquivalentSphericalRadius(1)
                    found, best_point = self.GridSearch(label, point, direction, class_mask, size, extra_iter)

                if not found: # fallback 2: contour 
                    fallback_count += 1
                    best_point = None
                    best_score = -np.inf

                    for idx_zyx in contour_idx_zyx:
                        z, y, x = idx_zyx
                        idx_xyz = (int(x), int(y), int(z))

                        p_phys = np.array(class_mask.TransformIndexToPhysicalPoint(idx_xyz),dtype=float)

                        score = np.dot(p_phys - centroid, direction)

                        if score > best_score:
                            best_score = score
                            best_point = p_phys

                    if best_point is not None:
                        self.points[label].append((tuple(float(x) for x in best_point),tuple(float(x) for x in direction),))
                    else:
                        failed_fallback_count += 1

            if grid_search_count > 0 or fallback_count > 0:
                logger.warning(
                "Label %s: used grid ray search for %s directions, contour fallback for %s directions, failed fallback for %s directions",
                label, grid_search_count, fallback_count, failed_fallback_count
            )

            logger.info(
            "Finished label %s: stored_points=%s unique_points=%s",
            label,
            len(self.points[label]),
            len(set(p for p, _ in self.points[label])) if len(self.points[label]) > 0 else 0,
        )
        return True

    def DefineAnchor(self, patient_path: Path):
        
        """Define the anchor patient for registration as the first frame mask."""
        masks_dir = patient_path / "labels"
        
        # choose the first frame as the anchor
        anchor_frame = sorted(masks_dir.glob("*.nii.gz"))[0]
        logger.info("Selected anchor frame: %s", anchor_frame)
        return anchor_frame
        

    def TransformToAnchor(self, fixed_mask_path : Path, moving_mask_path : Path):
        """Compute the transform used to resample the moving mask onto the fixed/anchor mask."""
        logger.info("Computing transform from moving=%s to fixed=%s", moving_mask_path, fixed_mask_path)
        # load masks as SimpleITK images
        fixed_orig = sitk.ReadImage(fixed_mask_path, sitk.sitkFloat32)
        moving_orig = sitk.ReadImage(moving_mask_path, sitk.sitkFloat32)

        # binarize segmentation mask
        fixed_arr = (sitk.GetArrayFromImage(fixed_orig) > 0).astype(np.uint8)
        moving_arr = (sitk.GetArrayFromImage(moving_orig) > 0).astype(np.uint8)

        fixed_bin = sitk.GetImageFromArray(fixed_arr)
        moving_bin = sitk.GetImageFromArray(moving_arr)
        fixed_bin.CopyInformation(fixed_orig)
        moving_bin.CopyInformation(moving_orig)

        fixed_reg = sitk.SignedMaurerDistanceMap(
        fixed_bin, insideIsPositive=False, squaredDistance=False, useImageSpacing=True) # anisotropic distance map to account for different spacing in the images
        moving_reg = sitk.SignedMaurerDistanceMap(
            moving_bin, insideIsPositive=False, squaredDistance=False, useImageSpacing=True)


        # initialize transform using the center of mass of the masks
        initial_transform = sitk.CenteredTransformInitializer(fixed_reg, moving_reg, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)

        # Registration Framework
        
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMeanSquares()
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0,
            minStep=1e-4,
            numberOfIterations=600)
        registration_method.SetInitialTransform(initial_transform, inPlace=False)

        # Execute Registration
        final_transform = registration_method.Execute(fixed_reg, moving_reg)

        print(f"Final Metric Value: {registration_method.GetMetricValue()}")
        print(f"Optimizer stop condition: {registration_method.GetOptimizerStopConditionDescription()}")
        logger.info(
            "Finished registration: metric=%s stop_condition=%s",
            registration_method.GetMetricValue(),
            registration_method.GetOptimizerStopConditionDescription()
        )

        return final_transform


    def AlignToAnchor(self, moving_mri_path, fixed_mri_reference, transform, IsMask = True):
        """Apply the computed transform to the a MRI image to align it to the reference MRI image."""
        logger.info(
            "Aligning moving image %s to reference %s (IsMask=%s)",
            moving_mri_path, fixed_mri_reference, IsMask
        )
        moving_mri = sitk.ReadImage(moving_mri_path)
        fixed_mri = sitk.ReadImage(fixed_mri_reference)
    
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_mri)
        resampler.SetDefaultPixelValue(0)
        if IsMask:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(transform)
    
        aligned = resampler.Execute(moving_mri)
        logger.debug("Finished alignment for %s", moving_mri_path)
        return aligned
    
def _find_corresponding_mri_frame(patient_dir: Path, mask_frame_path: Path):
    """
    Find the original MRI frame corresponding to a label/mask frame.

    Expected structure:
    patient_dir/
        labels/
        frames/
    """
    frames_dir = patient_dir / "frames"

    if not frames_dir.exists():
        return None

    candidate = frames_dir / mask_frame_path.name
    if candidate.exists():
        return candidate

    # fallback for .nii / .nii.gz ambiguity
    candidates = list(frames_dir.glob(mask_frame_path.stem + "*"))
    if len(candidates) > 0:
        return candidates[0]

    return None

def plot_control_points_on_mri_slice(
    mri_img: sitk.Image,
    coords,
    key: str,
    out_file: Path,
    z_slice=None,
    tol=1.0,
        ):
    """
    Plot one 2D z-slice of a 3D MRI frame with control points overlaid.

    mri_img: SimpleITK image of the original MRI frame, not the mask
    coords:  [K, 3] continuous voxel indexes in [x, y, z]
    """

    img = sitk.GetArrayFromImage(mri_img)  # [D, H, W]
    D, H, W = img.shape

    coords = np.asarray(coords, dtype=np.float32)

    if coords.shape[0] == 0:
        logger.warning("No control points to plot for %s", key)
        return

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    if z_slice is None:
        z_slice = int(np.round(np.mean(z)))

    z_slice = int(np.clip(z_slice, 0, D - 1))

    keep = np.abs(z - z_slice) <= tol

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 7))
    plt.imshow(img[z_slice], cmap="gray")
    plt.scatter(x[keep], y[keep], s=6, c="red", alpha=0.7)

    plt.title(f"{key} | z={z_slice} | points={int(keep.sum())}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()
    
def _flatten_control_points(points_dict):
    """
    Convert control_points.points into a single [K, 3] array.

    Assumes:
    - points_dict[label] = [(point, direction), ...]
    - point is already in continuous voxel index coordinates [x, y, z]
    """
    coords = []

    for label, points_with_dirs in points_dict.items():
        for p, d in points_with_dirs:
            coords.append(p)

    if len(coords) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    return np.asarray(coords, dtype=np.float32)



@hydra.main(version_base=None, config_path="../configs/model", config_name="controlpoints.yaml")
def main(config):
    """
    Main script for defining control points on masks 
    Args:
        config (DictConfig): hydra experiment config.
    """
    logging.basicConfig(
        level=getattr(logging, str(config.get("log_level", "INFO")).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    logger.info("Starting control point extraction pipeline")
    logger.info("Config: %s", config)

    control_points = ControlPoints(config)

    test = bool(config.get("test", False))

    if test: 
        logger.info("Running test mode")
        toy_mask = sitk.ReadImage(
            "/home/renku/work/cardiac_deformation_final_dataset/patient021/labels/patient021_iframe0004.nii.gz",
            sitk.sitkUInt8
        )
        control_points.ExtractPoints(toy_mask)
        test_control_points_2d_3d(control_points, toy_mask)

    else: 

        # define anchor frame (reference)
        data_path = Path(config.data_path)
        anchor_frame = control_points.DefineAnchor(data_path / "patient001")

        # define output paths
        out_json_path = Path(config.out_json_path)
        final_output_json = out_json_path / "total_coords.json"
        bad_seg_json = out_json_path / "bad_seg.json"

        # folder for alignment plots
        alignment_plot_dir = out_json_path / "alignment_plots"
        alignment_plot_dir.mkdir(parents=True, exist_ok=True)

        # folder for control points alignment check
        control_points_plot_dir = out_json_path / "control_points_plot"
        control_points_plot_dir.mkdir(parents=True, exist_ok=True)

        # loop over patients and frames to find control points
        patients_dir = sorted([f for f in data_path.iterdir() if f.is_dir()])
        logger.info("Found %s patient directories", len(patients_dir))

        results = {}
        bad_segmented = {}

        for pat in tqdm(patients_dir[18:], desc="Processing patients"):
            logger.info("Processing patient %s", pat.name)

            labels_dir = pat / "labels"
            if not labels_dir.exists():
                logger.warning("Skipping patient %s because labels directory does not exist", pat.name)
                continue

            coords_dir = pat / "coords"
            coords_dir.mkdir(parents=True, exist_ok=True)

            frames = sorted(list(labels_dir.glob("*.nii.gz")) + list(labels_dir.glob("*.nii")))
            logger.info("Patient %s has %s frames", pat.name, len(frames))

            if len(frames) == 0:
                logger.warning("Skipping patient %s because no frames were found", pat.name)
                continue

            results[pat.name] = {}

            # choose first frame
            first_frame = frames[0]

            # learn transform only between anchor and first frame
            old = sitk.ProcessObject_GetGlobalWarningDisplay()
            sitk.ProcessObject_SetGlobalWarningDisplay(False)
            try:
                transform = control_points.TransformToAnchor(anchor_frame, first_frame)
            finally:
                sitk.ProcessObject_SetGlobalWarningDisplay(old)

            logger.info(
                "Computed transform for patient %s using first frame %s",
                pat.name,
                first_frame.name
            )

            # align first frame to anchor for visualization
            aligned_first_frame = control_points.AlignToAnchor(
                first_frame,
                anchor_frame,
                transform,
                IsMask=True
            )

            # plot anchor and first mask alignment
            plot_anchor_first_alignment(
                anchor_mask_path=anchor_frame,
                aligned_first_mask=aligned_first_frame,
                patient_name=pat.name,
                out_dir=alignment_plot_dir
            )
            logger.info("Saved alignment plot for patient %s", pat.name)

            # process all frames using the same transform
            for fr in tqdm(frames, desc=f"Processing frames for {pat.name}", leave=False):
                logger.info("Processing frame %s for patient %s", fr.name, pat.name)

                # read original frame
                fr_img = sitk.ReadImage(str(fr), sitk.sitkUInt8)

                # align frame to anchor using first-frame transform
                aligned_fr = control_points.AlignToAnchor(
                    fr,
                    anchor_frame,
                    transform,
                    IsMask=True
                )

                # extract control points
                check = control_points.ExtractPoints(aligned_fr)
                if not check:
                    logger.warning(
                        "Bad segmentation for patient=%s frame=%s. Skipping coords.",
                        pat.name,
                        fr.name
                    )
                    bad_segmented.setdefault(pat.name, []).append(fr.name)
                    continue

                # map points back to original patient reference system
                control_points.TransformPointsBack(transform)

                # convert physical coordinates to continuous index coordinates
                control_points.ConvertPointsToContinuousIndex(fr_img)

                # plot control points on the corresponding original MRI frame, not on the mask
                frame_key = _frame_name(fr)

                mri_frame_path = _find_corresponding_mri_frame(pat, fr)

                if mri_frame_path is None:
                    logger.warning(
                        "Could not find corresponding MRI frame for patient=%s frame=%s. Skipping control point plot.",
                        pat.name,
                        fr.name,
                    )
                else:
                    mri_img = sitk.ReadImage(str(mri_frame_path))

                    coords_for_plot = _flatten_control_points(control_points.points)

                    plot_file = control_points_plot_dir / f"{pat.name}_{frame_key}_control_points.png"

                    plot_control_points_on_mri_slice(
                        mri_img=mri_img,
                        coords=coords_for_plot,
                        key=f"{pat.name} | {frame_key}",
                        out_file=plot_file,
                        z_slice=None,
                        tol=1.0,
                    )

                    logger.info("Saved control point plot to %s", plot_file)

                # save 
                frame_coords = _serialize_points(
                    control_points.points,
                    control_points.labels,
                )

                results[pat.name][frame_key] = frame_coords
                logger.debug(
                    "Saved serialized points for patient=%s frame=%s",
                    pat.name,
                    frame_key
                )

                frame_output_json = coords_dir / f"coords_{frame_key}.json"
                with open(frame_output_json, "w", encoding="utf-8") as f:
                    json.dump(frame_coords, f, indent=2)

                logger.info("Saved frame control points JSON to %s", frame_output_json)

        final_output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(final_output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved final control points JSON to %s", final_output_json)

        bad_seg_json.parent.mkdir(parents=True, exist_ok=True)
        with open(bad_seg_json, "w", encoding="utf-8") as f:
            json.dump(bad_segmented, f, indent=2)
        logger.info("Saved bad segmented frames in JSON to %s", bad_seg_json)


if __name__ == "__main__":
    main()
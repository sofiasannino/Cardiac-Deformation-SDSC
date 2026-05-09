import SimpleITK as sitk
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict
import hydra
from omegaconf import DictConfig
import warnings
from tqdm import tqdm
import os
import logging
from copy import deepcopy

from control_points_utils import test_control_points_2d_3d
from control_points_utils import _serialize_points,  _frame_name

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

    def GetLabels(): 
        return self.labels

    def TransformPointsBack(self, transform: sitk.Transform) -> None:
        """
        Transform all stored points from the anchor reference system
        back to the original image system, modifying self.points in place.

        Assumes:
        - self.points[label] = [(point, direction), ...]
        - `transform` maps ORIGINAL -> ANCHOR
        """

        logger.info("Transforming points back to original reference system")
        inverse_transform = transform.GetInverse()

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

    def GetSphericalDirections(self) -> List[Direction]:
        """Sample directions on the sphere"""
        logger.info("Sampling spherical directions")
        theta = np.linspace(0, np.pi, num=self.theta_num)
        phi = np.linspace(0, 2 * np.pi, num=self.phi_num, endpoint=False)

        directions = []
        for t in theta:
            for p in phi:
                x = np.sin(t) * np.cos(p)
                y = np.sin(t) * np.sin(p)
                z = np.cos(t)
                directions.append((float(x), float(y), float(z)))
        logger.info("Sampled %s spherical directions", len(directions))
        return directions

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

            # contour of the FULL label mask, for fallback
            contour = sitk.BinaryContour(class_mask)
            contour_arr = sitk.GetArrayFromImage(contour)   # [z, y, x]
            contour_idx_zyx = np.argwhere(contour_arr > 0)
            logger.debug("Label %s: contour has %s voxels", label, len(contour_idx_zyx))

            extended_search_count = 0
            fallback_count = 0
            failed_fallback_count = 0
            for d in tqdm(self.directions, desc = f"Extracting points from label {label} in all directions", leave=False):
                direction = np.array(d, dtype=float)

                # starting rule as your previous working version
                point = centroid + direction * 2 * shape_stats.GetEquivalentSphericalRadius(1)

                found = False
                for _ in range(self.max_iter):
                    point_idx = tuple(int(x) for x in class_mask.TransformPhysicalPointToIndex(tuple(point)))

                    inside_image = all(0 <= point_idx[i] < size[i] for i in range(len(size)))

                # check against the FULL label mask
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

    def ExtractPointsFromContour(self, mask: sitk.Image):
        """
        For each label:
        - compute centroid from the largest connected component
        - compute the contour of the FULL label mask
        - for each spherical direction, choose the contour point most aligned with that direction
        - store as (point, direction) inside points[label]

        Output format:
            self.points[label] = [(point, direction), ...]
        """

        logger.info("Extracting control points from contour method")
        # reset dictionary
        self.points = {label: [] for label in self.labels}

        for label in tqdm(self.labels, desc="Processing labels in ExtractPointsFromContour"):
            logger.info("Processing label %s in contour extraction", label)
            # full binary mask for the current label
            class_mask = sitk.Cast(mask == label, sitk.sitkUInt8)

            # connected components for stable centroid computation
            cc = sitk.ConnectedComponent(class_mask)
            relabeled = sitk.RelabelComponent(cc)
            largest_cc = sitk.Cast(relabeled == 1, sitk.sitkUInt8)

            shape_stats = sitk.LabelShapeStatisticsImageFilter()
            shape_stats.Execute(largest_cc)

            # centroid of the largest connected component
            centroid = np.array(shape_stats.GetCentroid(1), dtype=float)

            # contour of the FULL label mask
            contour = sitk.BinaryContour(class_mask)
            contour_arr = sitk.GetArrayFromImage(contour)   # [z, y, x]
            contour_idx_zyx = np.argwhere(contour_arr > 0)

            if len(contour_idx_zyx) == 0:
                warnings.warn(f"No contour found for label {label}.", RuntimeWarning)
                logger.warning("No contour found for label %s", label)
                continue

            logger.debug("Label %s contour extraction has %s voxels", label, len(contour_idx_zyx))

            # precompute contour points in physical coordinates
            contour_points_phys = []
            for idx_zyx in tqdm(contour_idx_zyx, desc=f"Precomputing contour points for label {label}", leave=False):
                z, y, x = idx_zyx
                idx_xyz = (int(x), int(y), int(z))
                p_phys = class_mask.TransformIndexToPhysicalPoint(idx_xyz)
                contour_points_phys.append(p_phys)

            contour_points_phys = np.array(contour_points_phys, dtype=float)   # [N, 3]

            # vectors from centroid to contour points
            rel_vectors = contour_points_phys - centroid[None, :]   # [N, 3]

            for d in tqdm(self.directions, desc=f"Selecting contour points for label {label}", leave=False):
                direction = np.array(d, dtype=float)

                # projection score of each contour point onto the direction
                scores = rel_vectors @ direction   # [N]

                best_idx = int(np.argmax(scores))
                best_point = contour_points_phys[best_idx]

                self.points[label].append((tuple(float(x) for x in best_point), tuple(float(x) for x in direction),))
    
            logger.info(
                "Finished contour extraction for label %s: stored_points=%s unique_points=%s",
                label,
                len(self.points[label]),
                len(set(p for p, _ in self.points[label])) if len(self.points[label]) > 0 else 0,
            )

    def InitializeFromMask(self, mask : sitk.Image, num_points_per_label: int):
        """Initialize control points from a segmentation mask. For each label, sample num_points_per_label points."""
        logger.info("Initializing control points from mask with num_points_per_label=%s", num_points_per_label)
        for label in tqdm(self.labels, desc="Initializing points from mask"):
            logger.info("Initializing label %s from mask", label)

            # exctract binary mask for the current label
            class_mask = sitk.Cast(mask == label, sitk.sitkUInt8)
            # compute all discontinuous regions
            cc = sitk.ConnectedComponent(class_mask)
            shape_stats = sitk.LabelShapeStatisticsImageFilter()
            shape_stats.Execute(cc)
            
            # find biggest connected component 
            max_num_pixels = 0
            largest_cc = None
            for lab in shape_stats.GetLabels():
                num_pixels = shape_stats.GetNumberOfPixels(lab)
                print(f"sphere diameter for label {lab}: {shape_stats.GetEquivalentSphericalDiameter(lab)}") # DEBUG
                print(f"number of indexes for label {lab}:{len(shape_stats.GetIndexes(lab)) // 3}") # DEBUG
                logger.debug(
                    "Label %s component %s: sphere diameter=%s num_voxels=%s",
                    label, lab, shape_stats.GetEquivalentSphericalDiameter(lab), len(shape_stats.GetIndexes(lab)) // 3
                )
                if num_pixels > max_num_pixels:
                    max_num_pixels = num_pixels
                    largest_cc = sitk.Cast(cc == lab, sitk.sitkUInt8)

            # sample points from the largest connected component round the centroid of the component
            if largest_cc is not None:
                shape_stats.Execute(largest_cc)

                # compute the centroid of the largest connected component
                centroid = np.array(shape_stats.GetCentroid(1))
                flat_idx = shape_stats.GetIndexes(1)
                physical_points = []

                # get physical coordinates of the points
                for k in tqdm(range(0, len(flat_idx), 3), desc=f"Collecting physical points for label {label}", leave=False):
                    idx = [int(flat_idx[k]), int(flat_idx[k+1]), int(flat_idx[k+2])]
                    p = largest_cc.TransformIndexToPhysicalPoint(idx)
                    physical_points.append(p)

                # sort points by distance to the centroid
                physical_points.sort(key=lambda p: np.linalg.norm(np.array(p) - np.array(centroid)))

                # keep farthest points from the centroid to capture the shape of the structure
                if len(physical_points) < num_points_per_label:
                    print(f"Warning: label {label} has only {len(physical_points)} points, less than the requested {num_points_per_label}. Keeping all points.")
                    logger.warning(
                        "Label %s has only %s points, less than requested %s. Keeping all points.",
                        label, len(physical_points), num_points_per_label
                    )
                    selected_points = physical_points
                else:
                    selected_points = physical_points[-num_points_per_label:]

            self.points[label] = selected_points
            logger.info("Initialized label %s with %s selected points", label, len(selected_points))

    def DefineAnchor(self, patient_path: Path):
        
        """Define the anchor patient for registration as the first frame mask."""
        masks_dir = patient_path / "labels"
        
        # choose the first frame as the anchor
        anchor_frame = sorted(masks_dir.glob("*.nii.gz"))[0]
        logger.info("Selected anchor frame: %s", anchor_frame)
        return anchor_frame
        

    def TransformToAnchor(self, fixed_mask_path : Path, moving_mask_path : Path):
        """Compute the transform from a mask to the reference mask using SimpleITK registration."""
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
    test = False 
    if test : 
        logger.info("Running test mode")
        toy_mask = sitk.ReadImage("/home/renku/work/s3-bucket/ACDC/training/patient001/patient001_frame01_gt.nii.gz", sitk.sitkUInt8)
        control_points.ExtractPoints(toy_mask)
        test_control_points_2d_3d(control_points, toy_mask)
        points_1 = control_points.GetPoints(1)
        pts = [p for p, d in points_1]
        print("total 1:", len(pts))
        print("unique 1:", len(set(pts)))

        points_2 = control_points.GetPoints(2)
        pts = [p for p, d in points_2]
        print("total 2:", len(pts))
        print("unique 2:", len(set(pts)))

        points_3 = control_points.GetPoints(3)
        pts = [p for p, d in points_3]
        print("total 3:", len(pts))
        print("unique 3:", len(set(pts)))


    
    # define anchor frame (reference)
    data_path = Path(config.data_path)
    anchor_frame = control_points.DefineAnchor(data_path / "patient001")

    # loop over patients and frames to find control points
    patients_dir = sorted([f for f in (data_path).iterdir() if f.is_dir()])
    logger.info("Found %s patient directories", len(patients_dir))

    results = {}
    bad_segmented = {}
    final_output_json = Path(Path(config.out_json_path) / "total_coords.json")
    bad_seg_json = Path(Path(config.out_json_path) / "bad_seg.json")
    final_frame = None

    for pat in tqdm(patients_dir, desc="Processing patients"):
        logger.info("Processing patient %s", pat.name)
        labels_dir = pat / "labels"
        if not labels_dir.exists():
            logger.warning("Skipping patient %s because labels directory does not exist", pat.name)
            continue

        coords_dir = pat / "coords"
        coords_dir.mkdir(parents=True, exist_ok=True)

        # assuming frames are NIfTI files
        frames = sorted(list(labels_dir.glob("*.nii.gz")) + list(labels_dir.glob("*.nii")))
        logger.info("Patient %s has %s frames", pat.name, len(frames))
        results[pat.name] = {}

        for fr in tqdm(frames, desc=f"Processing frames for {pat.name}", leave=False):
            fr_img = sitk.ReadImage(str(fr))
            logger.info("Processing frame %s for patient %s", fr.name, pat.name)
            # learn transform btw anchor and frame mask 
            # suppressing known normal NifTi warnings 
            old = sitk.ProcessObject.GetGlobalWarningDisplay()
            sitk.ProcessObject_SetGlobalWarningDisplay(False)
            try:
                transform = control_points.TransformToAnchor(anchor_frame, fr)
                # align frame to anchor 
                aligned_fr = control_points.AlignToAnchor(fr, anchor_frame, transform, IsMask=True)
            finally:
                sitk.ProcessObject_SetGlobalWarningDisplay(old)

            # extract control points
            check = control_points.ExtractPoints(aligned_fr)
            if not check:
                logger.warning( "Bad segmentation for patient=%s frame=%s. Skipping coords.", pat.name,
                frame_key,)
                bad_segmented.setdefault(pat.name, []).append(frame_key)
            continue
            
            # map them back and into index coordinates  
            control_points.TransformPointsBack(transform)
            
            pp = control_points.GetPoints()

            # convert physical coordinates to continuous index coordinates
            for label in control_points.GetLabels():
                points_per_label = pp[label]

                for pt in points_per_label:
                    point_phys = tuple(float(x) for x in pt["point"])

                    point_idx_xyz = fr_img.TransformPhysicalPointToContinuousIndex(point_phys)

                    # keep SimpleITK index order [x, y, z]
                    pt["point"] = [float(x) for x in point_idx_xyz]
            # save 
            frame_key = _frame_name(fr)
            frame_coords = _serialize_points(
            control_points.points,
            control_points.labels,
        )

            results[pat.name][frame_key] = frame_coords
            logger.debug(
            "Saved serialized points for patient=%s frame=%s",
            pat.name, frame_key
        )

            frame_output_json = coords_dir / f"coords_{frame_key}.json"
            with open(frame_output_json, "w", encoding="utf-8") as f:
                json.dump(frame_coords, f, indent=2)
            logger.info("Saved frame control points JSON to %s", frame_output_json)

            final_frame = fr

    final_output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(final_output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved final control points JSON to %s", final_output_json)

    bad_seg_json.parent.mkdir(parents=True, exist_ok=True)
    with open(bad_seg_json, "w", encoding = "utf-8" ) as f: 
        json.dump(bad_segmented, f, indent=2)
    logger.info("Saved bad segmented frames in JSON to %s", bad_seg_json)

    # plot 
    if final_frame is not None:
        test_control_points_2d_3d(control_points, sitk.ReadImage(str(final_frame)))


if __name__ == "__main__":
    main()
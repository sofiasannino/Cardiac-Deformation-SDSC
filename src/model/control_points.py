import SimpleITK as sitk
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict
import hydra
from omegaconf import DictConfig
import warnings

Point3D = Tuple[float, float, float]
Direction = Tuple[float, float, float]


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

    def GetPoints(self, label: int) -> List[Tuple[Point3D, Direction]]:
        """Return list of (point, direction) for a given label"""
        return self.points[label]

    def TransformPointsBack(
        self,
        points_with_dirs: List[Tuple[Point3D, Direction]],
        transform: sitk.Transform
    ) -> List[Tuple[Point3D, Direction]]:
        """Apply a SimpleITK transform to the points""" # CHANGE THIS 
        transformed = []
        for p, d in points_with_dirs:
            transformed_p = transform.TransformPoint(p)
            transformed.append((transformed_p, d))
        return transformed

    def GetSphericalDirections(self) -> List[Direction]:
        """Sample directions on the sphere"""
        theta = np.linspace(0, np.pi, num=self.theta_num)
        phi = np.linspace(0, 2 * np.pi, num=self.phi_num, endpoint=False)

        directions = []
        for t in theta:
            for p in phi:
                x = np.sin(t) * np.cos(p)
                y = np.sin(t) * np.sin(p)
                z = np.cos(t)
                directions.append((float(x), float(y), float(z)))
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

        # reset dictionary with the new format
        self.points = {label: [] for label in self.labels}

        for label in self.labels:
            # full binary mask for the current label
            class_mask = sitk.Cast(mask == label, sitk.sitkUInt8)

            # connected components
            cc = sitk.ConnectedComponent(class_mask)
            relabeled = sitk.RelabelComponent(cc)
            largest_cc = sitk.Cast(relabeled == 1, sitk.sitkUInt8)

            shape_stats = sitk.LabelShapeStatisticsImageFilter()
            shape_stats.Execute(largest_cc)


            # centroid of the largest connected component
            centroid = np.array(shape_stats.GetCentroid(1), dtype=float)

            size = class_mask.GetSize()

            for d in self.directions:
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

            if not found:
                warnings.warn(f"Reached maximum number of iterations for label {label} in direction {d}",RuntimeWarning,)
                # contour of the FULL label mask, for fallback
            contour = sitk.BinaryContour(class_mask)
            contour_arr = sitk.GetArrayFromImage(contour)   # [z, y, x]
            contour_idx_zyx = np.argwhere(contour_arr > 0)
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
                warnings.warn(f"Fallback used for label {label} in direction {d}",RuntimeWarning)
            else:
                warnings.warn(f"Failed to find point for label {label} in direction {d}",RuntimeWarning)
    
    def InitializeFromMask(self, mask : sitk.Image, num_points_per_label: int):
        """Initialize control points from a segmentation mask. For each label, sample num_points_per_label points."""
        for label in self.labels:

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
                for k in range(0, len(flat_idx), 3):
                    idx = [int(flat_idx[k]), int(flat_idx[k+1]), int(flat_idx[k+2])]
                    p = largest_cc.TransformIndexToPhysicalPoint(idx)
                    physical_points.append(p)

                # sort points by distance to the centroid
                physical_points.sort(key=lambda p: np.linalg.norm(np.array(p) - np.array(centroid)))

                # keep farthest points from the centroid to capture the shape of the structure
                if len(physical_points) < num_points_per_label:
                    print(f"Warning: label {label} has only {len(physical_points)} points, less than the requested {num_points_per_label}. Keeping all points.")
                    selected_points = physical_points
                else:
                    selected_points = physical_points[-num_points_per_label:]

            self.points[label] = selected_points

    def DefineAnchor(self, patient_path: Path):
        
        """Define the anchor patient for registration as the first frame mask."""
        masks_dir = patient_path / "labels"
        
        # choose the first frame as the anchor
        anchor_frame = sorted(masks_dir.glob("*.nii.gz"))[0]
        return anchor_frame
        

    def TransformToAnchor(self, fixed_mask_path : Path, moving_mask_path : Path):
        """Compute the transform from a mask to the reference mask using SimpleITK registration."""
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
            minStep=1e-3,
            numberOfIterations=200)
        registration_method.SetInitialTransform(initial_transform, inPlace=False)

        # Execute Registration
        final_transform = registration_method.Execute(fixed_reg, moving_reg)

        print(f"Final Metric Value: {registration_method.GetMetricValue()}")
        print(f"Optimizer stop condition: {registration_method.GetOptimizerStopConditionDescription()}")

        return final_transform


    def AlignToAnchor(self, moving_mri_path, fixed_mri_reference, transform, IsMask = True):
        """Apply the computed transform to the a MRI image to align it to the reference MRI image."""
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
    
        return resampler.Execute(moving_mri)
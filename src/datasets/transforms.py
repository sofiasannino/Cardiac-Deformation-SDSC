import torch
import SimpleITK as sitk
import numpy as np
import numbers
import random 


import torchvision




def resample(image4d, target_spacing):
        '''resample input image to target spacing'''
        arr4d = sitk.GetArrayFromImage(image4d)  # [T, Z, Y, X]
        spacing4d = image4d.GetSpacing()

        # assume spacing is (x, y, z, ..)
        spatial_spacing = spacing4d[:3]

        def safe_direction_3d(image):
            direction_4d = np.array(image.GetDirection()).reshape(4, 4)
            direction_3d = direction_4d[:3, :3].reshape(-1).tolist()
            return direction_3d

        def resample_3d(image3d, target_spacing) :
            original_spacing = image3d.GetSpacing()
            original_size = image3d.GetSize()
            target_size_res = [
            int(round(original_size[0] * original_spacing[0] / target_spacing[0])),
            int(round(original_size[1] * original_spacing[1] / target_spacing[1])),
            int(round(original_size[2] * original_spacing[2] / target_spacing[2]))
        ]
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(target_spacing)
            resampler.SetSize(target_size_res)
            resampler.SetOutputDirection(image3d.GetDirection())
            resampler.SetOutputOrigin(image3d.GetOrigin())
            resampler.SetInterpolator(sitk.sitkLinear)
            resampled_image = resampler.Execute(image3d)
            return resampled_image

        frames = []
        for t in range(arr4d.shape[0]):
            frame_arr = arr4d[t]  # [Z, Y, X]

            frame_img = sitk.GetImageFromArray(frame_arr)
            frame_img.SetSpacing(spatial_spacing)
            frame_img.SetOrigin(image4d.GetOrigin()[:3])
            frame_img.SetDirection(safe_direction_3d(image4d))

            frame_res = resample_3d(frame_img, target_spacing)
            frame_np = sitk.GetArrayFromImage(frame_res).astype(np.float32)  # [Z, Y, X]
            frames.append(frame_np)

        return np.stack(frames, axis=0)  # [T, D, H, W]

    


def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tensor. Got %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())

    return True


def to_tensor(clip):
    """
    Convert tensor data type from uint8 to float
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (T, D, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, D, H, W)
    """
    clip = torch.from_numpy(clip).contiguous()
    _is_tensor_video_clip(clip)
    return clip.float()

def center_crop(clip, crop_size):
    assert _is_tensor_video_clip(clip), "clip should be a 4D torch.tensor"
    h, w = clip.size(-2), clip.size(-1)
    th, tw = crop_size
    assert h >= th and w >= tw, "height and width must be no smaller than crop_size"

    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return crop(clip, i, j, th, tw)


def crop(clip, i, j, h, w):
    """
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (T, D, H, W)
    """
    assert len(clip.size()) == 4, "clip should be a 4D tensor"
    return clip[..., i:i + h, j:j + w]

def resize(clip, target_size, ES_index):
    """
    Args:
        clip (torch.Tensor): video of shape (T, D, H, W)
    Return:
        new_clip (torch.Tensor): resized video of shape (T', D, H, W)
    """
    T, D, H, W = clip.shape

    if T == 1:
        return clip.repeat(target_size, 1, 1, 1)

    new_ES_index = int(round((target_size - 1) * ES_index / (T - 1)))
    new_clip = torch.zeros((target_size, D, H, W), dtype=clip.dtype, device=clip.device)

    # left part: before ES
    if new_ES_index > 0:
        new_grid_1 = np.linspace(0, ES_index - 1, new_ES_index)
        for i in range(new_ES_index):
            m = new_grid_1[i]
            if float(m).is_integer():
                new_clip[i, ...] = clip[int(m), ...]
            else:
                i_prev = int(np.floor(m))
                i_prox = int(np.ceil(m))
                alpha = m - i_prev
                new_clip[i, ...] = clip[i_prev, ...] + alpha * (clip[i_prox, ...] - clip[i_prev, ...])

    # ES exactly preserved
    new_clip[new_ES_index, ...] = clip[ES_index, ...]

    # right part: from ES onward
    right_len = target_size - new_ES_index
    if right_len > 1:
        new_grid_2 = np.linspace(ES_index, T - 1, right_len)
        for j in range(1, right_len):   # start from 1 because j=0 is ES itself
            m = new_grid_2[j]
            i = new_ES_index + j
            if float(m).is_integer():
                new_clip[i, ...] = clip[int(m), ...]
            else:
                i_prev = int(np.floor(m))
                i_prox = int(np.ceil(m))
                alpha = m - i_prev
                new_clip[i, ...] = clip[i_prev, ...] + alpha * (clip[i_prox, ...] - clip[i_prev, ...])

    return new_clip
    
class ToTensorVideo(object):
    """
    Convert tensor data type from uint8 to float, and
    permute the dimensions of clip tensor
    """

    def __init__(self):
        pass

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.uint8): Size is (T, D, H, W)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (T, D, H, W)
        """
        return to_tensor(clip)
    
class CenterCropVideo(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clip):
        return center_crop(clip, self.size)
    

class ResizedVideo(object):
    def __init__(self, size, ES_index):
       
        self.size = size
        self.ES_index = ES_index

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be resized in time. Size is (C, T, H, W)
        Returns:
            torch.tensor: time-resized video clip.
                size is ( T',D, H, W) according to size
        """

        return resize(clip, self.size, self.ES_index)
    

class NormalizeVideo(object):
    """
    Normalize the video clip to [0,1]
   
    """

    def __init__(self):
        pass

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): video clip to be normalized. Size is (T, D, H, W)
        """
        clip = clip - clip.min()
        clip = clip/ clip.std()
        clip = clip / clip.max()
        return clip 




class Rotate(object):
    """
    Randomly rotate the whole video by the same angle on the spatial plane (H, W).

    Args:
        clip (torch.Tensor): Video tensor with shape (T, D, H, W).
        factor (tuple[float, float]): Min and max rotation angle in degrees.
        p (float): Probability of applying the rotation.
    """
    def __init__(self, factor=[-10, 10], p=0.5):
        self.factor = factor
        self.p = p

    def __call__(self, clip):
        assert _is_tensor_video_clip(clip), "clip should be a 4D torch.tensor"

        if random.random() < self.p:
            angle = random.uniform(self.factor[0], self.factor[1])
            clip = torchvision.transforms.functional.rotate(
                clip,
                angle,
                interpolation=torchvision.transforms.functional.InterpolationMode.BILINEAR
            )
        return clip

          




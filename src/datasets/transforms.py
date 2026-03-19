import torch
import SimpleITK as sitk
import numpy as np
import numbers




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

def resize(clip, target_size, interpolation_mode): 
    """
    Args: 
        clip (torch.tensor): Video to be cropped in time through interpolation, restricted between ED and ES. Size is (T, D, H, W)
    Return:
        clipe (torch.tensor): Video cropped in time through interpolation. Size is (T', D, H, W) 
    """
    shape = tuple((target_size, clip.shape[1], clip.shape[2], clip.shape[3]))
    new_clip = torch.zeros(shape, dtype= clip.dtype, device = clip.device) 
     
    new_grid = np.linspace(0, clip.shape[0]-1, target_size) # choosing target_zize number of points

    # Interpolating with Taylor using the new grid and keeping ED/ES at beginning and end  
    for i in np.arange(target_size) : 
        m = new_grid[i]
        i_prev = i_prox= 0 
           
        if float(m).is_integer() :
                new_clip[i, ...] = clip[int(m), ...]
        else :  
            i_prox = int(np.ceil(m))
            i_prev = int(np.floor(m))
                
            # approximation using taylor, approximating the derivative with CD
            new_clip[i, ...] = clip[i_prev, ...] + (new_grid[i] - i_prev) * ((clip[i_prox, ...] - clip[i_prev, ...])/(i_prox - i_prev))

        
    return new_clip   #[Tmax, D, H, W]

    
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
    def __init__(self, size, ES_index, interpolation_mode="bilinear"):
       
        self.size = size
        self.ES_index = ES_index
       
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be resized in time. Size is (C, T, H, W)
        Returns:
            torch.tensor: time-resized video clip.
                size is ( T',D', H', W') according to size and containing just the cycle ED-ES
        """
        clip = clip[0:self.ES_index, ...]
        return resize(clip, self.size, self.interpolation_mode)
    

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

          




import torch, cv2
import numpy.random as random
import numpy as np
import dataloaders.helpers as helpers

class ScaleNRotate(object):
    """Scale (zoom-in, zoom-out) and Rotate the image and the ground truth.
    Args:
        two possibilities:
        1.  rots (tuple): (minimum, maximum) rotation angle
            scales (tuple): (minimum, maximum) scale
        2.  rots [list]: list of fixed possible rotation angles
            scales [list]: list of fixed possible scales
    """
    def __init__(self, rots=(-30, 30), scales=(.75, 1.25), semseg=False):
        assert (isinstance(rots, type(scales)))
        self.rots = rots
        self.scales = scales
        self.semseg = semseg

    def __call__(self, sample):

        if type(self.rots) == tuple:
            # Continuous range of scales and rotations
            rot = (self.rots[1] - self.rots[0]) * random.random() - \
                  (self.rots[1] - self.rots[0])/2

            sc = (self.scales[1] - self.scales[0]) * random.random() - \
                 (self.scales[1] - self.scales[0]) / 2 + 1
        elif type(self.rots) == list:
            # Fixed range of scales and rotations
            rot = self.rots[random.randint(0, len(self.rots))]
            sc = self.scales[random.randint(0, len(self.scales))]

        for elem in sample.keys():
            if 'meta' in elem:
                continue
            tmp = sample[elem]
            h, w = tmp.shape[:2]
            center = (w / 2, h / 2)
            assert(center != 0)  # Strange behaviour warpAffine
            M = cv2.getRotationMatrix2D(center, rot, sc)
            if ((tmp == 0) | (tmp == 1)).all():
                flagval = cv2.INTER_NEAREST
            elif 'gt' in elem and self.semseg:
                flagval = cv2.INTER_NEAREST
            else:
                flagval = cv2.INTER_CUBIC
            tmp = cv2.warpAffine(tmp, M, (w, h), flags=flagval)
            sample[elem] = tmp

        return sample

    def __str__(self):
        return 'ScaleNRotate:(rot='+str(self.rots)+',scale='+str(self.scales)+')'


class FixedResize(object):
    """Resize the image and the ground truth to specified resolution.
    Args:
        resolutions (dict): the list of resolutions
    """
    def __init__(self, resolutions=None, flagvals=None):
        self.resolutions = resolutions
        self.flagvals = flagvals
        if self.flagvals is not None:
            assert(len(self.resolutions) == len(self.flagvals))

    def __call__(self, sample):
        # Fixed range of scales
        if self.resolutions is None:
            return sample
        elems = list(sample.keys())
        for elem in elems:
            if  elem == 'meta':
                continue
            if elem in self.resolutions:
                if self.resolutions[elem] is None:
                    continue
                if self.flagvals is None:
                    sample[elem] = helpers.fixed_resize(sample[elem], self.resolutions[elem])
                else:
                    sample[elem] = helpers.fixed_resize(sample[elem], self.resolutions[elem], flagval=self.flagvals[elem])
            else:
                del sample[elem]
        return sample

    def __str__(self):
        return 'FixedResize:'+str(self.resolutions)


class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):

        if random.random() < 0.5:
            for elem in sample.keys():
                if 'meta' in elem:
                    continue
                tmp = sample[elem]
                tmp = cv2.flip(tmp, flipCode=1)
                sample[elem] = tmp
        return sample

    def __str__(self):
        return 'RandomHorizontalFlip'


class DistanceMap(object):
    """
    Returns the distance_map
    sigma: sigma of Gaussian to create a heatmap from a point
    pert: number of pixels fo the maximum perturbation
    elem: which element of the sample to choose as the binary mask
    """
    def __init__(self, sigma=10, elem='gt',pad_pixel =10):
        self.sigma = sigma
        self.elem = elem
        self.pad_pixel =pad_pixel

    def __call__(self, sample):

        if sample[self.elem].ndim == 3:
            raise ValueError('distance_map not implemented for multiple object per image.')
        _target = sample[self.elem]

        targetshape=_target.shape
        if np.max(_target) == 0:
            sample['distance_map'] = np.zeros([targetshape[0],targetshape[1],2], dtype=_target.dtype) 
        else:
            _points = helpers.GetDistanceMap(_target,self.pad_pixel)
            sample['distance_map'] = helpers.make_gt(_target, _points, sigma=self.sigma, one_mask_per_point=False)
        return sample

    def __str__(self):
        return 'DistanceMap:(sigma='+str(self.sigma)+', pad_pixel='+str(self.pad_pixel)+', elem='+str(self.elem)+')'

class ConcatInputs(object):

    def __init__(self, elems=('image', 'point')):
        self.elems = elems

    def __call__(self, sample):

        res = sample[self.elems[0]]
        for elem in self.elems[1:]:
            assert(sample[self.elems[0]].shape[:2] == sample[elem].shape[:2])

            # Check if third dimension is missing
            tmp = sample[elem]
            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]
            res = np.concatenate((res, tmp), axis=2)
        sample['concat'] = res
        return sample

    def __str__(self):
        return 'ExtremePoints:'+str(self.elems)

class CropFromMask(object):
    """
    Returns image cropped in bounding box from a given mask
    """
    def __init__(self, crop_elems=('image', 'gt','void_pixels'),
                 mask_elem='gt',
                 relax=0,
                 zero_pad=False):

        self.crop_elems = crop_elems
        self.mask_elem = mask_elem
        self.relax = relax
        self.zero_pad = zero_pad

    def __call__(self, sample):
        _target = sample[self.mask_elem]
        for elem in self.crop_elems:
            _img = sample[elem]
            if self.mask_elem == elem:
                _tmp_img = _img
                _tmp_target = _target
                if np.max(_target) == 0:
                   _crop = np.zeros(_tmp_img.shape, dtype=_img.dtype)
                else:
                    _crop = helpers.crop_from_mask(_tmp_img, _tmp_target, relax=self.relax, zero_pad=self.zero_pad)
            else:
                if np.max(_target) == 0:
                    _crop = np.zeros(_img.shape, dtype=_img.dtype)
                else:
                    _tmp_target = _target
                    _crop = helpers.crop_from_mask(_img, _tmp_target, relax=self.relax, zero_pad=self.zero_pad)
            sample['crop_' + elem] = _crop
        return sample

    def __str__(self):
        return 'CropFromMask:(crop_elems='+str(self.crop_elems)+', mask_elem='+str(self.mask_elem)+\
               ', relax='+str(self.relax)+',zero_pad='+str(self.zero_pad)+')'     
                                        
class ToImage(object):
    """
    Return the given elements between 0 and 255
    """
    def __init__(self, norm_elem='image', custom_max=255.):
        self.norm_elem = norm_elem
        self.custom_max = custom_max

    def __call__(self, sample):
        if isinstance(self.norm_elem, tuple):
            for elem in self.norm_elem:
                tmp = sample[elem]
                sample[elem] = self.custom_max * (tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-10)
        else:
            tmp = sample[self.norm_elem]
            sample[self.norm_elem] = self.custom_max * (tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-10)
        return sample

    def __str__(self):
        return 'NormalizeImage'


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        for elem in sample.keys():
            if 'meta' in elem:
                continue
            elif 'bbox' in elem:
                tmp = sample[elem]
                sample[elem] = torch.from_numpy(tmp)
                continue
            tmp = sample[elem]
            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]
            tmp = tmp.transpose((2, 0, 1))
            sample[elem] = torch.from_numpy(tmp)
        return sample

    def __str__(self):
        return 'ToTensor'

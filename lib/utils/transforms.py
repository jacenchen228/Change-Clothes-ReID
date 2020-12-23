from __future__ import division, print_function, absolute_import
import math
import random
import torch
from PIL import Image

from .transforms_func import (
    Resize, Compose, ToTensor, Normalize, ColorJitter, RandomHorizontalFlip, RandomCrop, Pad,
    Random2DTranslation, RandomErasing
)


class ColorAugmentation(object):
    """Randomly alters the intensities of RGB channels.
    Reference:
        Krizhevsky et al. ImageNet Classification with Deep ConvolutionalNeural
        Networks. NIPS 2012.
    Args:
        p (float, optional): probability that this operation takes place.
            Default is 0.5.
    """

    def __init__(self, p=0.5):
        self.p = p
        self.eig_vec = torch.Tensor(
            [
                [0.4009, 0.7192, -0.5675],
                [-0.8140, -0.0045, -0.5808],
                [0.4203, -0.6948, -0.5836],
            ]
        )
        self.eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])

    def _check_input(self, tensor):
        assert tensor.dim() == 3 and tensor.size(0) == 3

    def __call__(self, tensor):
        if random.uniform(0, 1) > self.p:
            return tensor
        alpha = torch.normal(mean=torch.zeros_like(self.eig_val)) * 0.1
        quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor


class FlipLR(object):
    """
    Flip Horizontally
    """
    def __init__(self):
        pass

    def __call__(self, img):
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()  # N x C x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip


def build_transforms(
        height,
        width,
        transforms='random_flip',
        norm_mean1=[0.485, 0.456, 0.406],
        norm_std1=[0.229, 0.224, 0.225],
        norm_mean2=[0.0],
        norm_std2=[1.0],
        **kwargs):
    """Builds train and test transform functions.s.
    """
    if transforms is None:
        transforms = []

    if isinstance(transforms, str):
        transforms = [transforms]

    if not isinstance(transforms, list):
        raise ValueError(
            'transforms must be a list of strings, but found to be {}'.format(
                type(transforms)
            )
        )

    if len(transforms) > 0:
        transforms = [t.lower() for t in transforms]

    if norm_mean1 is None or norm_std1 is None:
        norm_mean1 = [0.485, 0.456, 0.406]  # imagenet mean
        norm_std1 = [0.229, 0.224, 0.225]  # imagenet std
    normalize = Normalize(mean1=norm_mean1, std1=norm_std1, mean2=norm_mean2, std2=norm_std2)

    print('Building train transforms ...')
    transform_tr = []

    if 'random_flip' in transforms:
        print('+ random flip')
        transform_tr += [RandomHorizontalFlip()]

    print('+ resize to {}x{}'.format(height, width))
    transform_tr += [Resize((height, width))]

    if 'pad' in transforms:
        transform_tr += [Pad(10)]

    if 'random_crop' in transforms:
        transform_tr += [RandomCrop((height, width))]

    # if 'random_crop' in transforms:
    #     print(
    #         '+ random crop (enlarge to {}x{} and '
    #         'crop {}x{})'.format(
    #             int(round(height * 1.125)), int(round(width * 1.125)), height,
    #             width
    #         )
    #     )
    #     transform_tr += [Random2DTranslation(height, width)]

    if 'color_jitter' in transforms:
        print('+ color jitter')
        transform_tr += [
            ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0)
        ]

    print('+ to torch tensor of range [0, 1]')
    transform_tr += [ToTensor()]

    print('+ normalization (mean1={}, std1={}, mean2={}, std2={})'.format(norm_mean1, norm_std1, norm_mean2, norm_std2))
    transform_tr += [normalize]

    if 'random_erase' in transforms:
        print('+ random erase')
        transform_tr += [RandomErasing(mean1=norm_mean1)]

    transform_tr = Compose(transform_tr)

    print('Building test transforms ...')
    print('+ resize to {}x{}'.format(height, width))
    print('+ to torch tensor of range [0, 1]')
    print('+ normalization (mean1={}, std1={}, mean2={}, std2={})'.format(norm_mean1, norm_std1, norm_mean2, norm_std2))

    transform_te = Compose([
        Resize((height, width)),
        ToTensor(),
        normalize,
    ])

    return transform_tr, transform_te


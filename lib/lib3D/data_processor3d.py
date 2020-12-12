import numpy as np
from PIL import Image

from .constants import *
from .transforms import *

__all__ = ['data_processor3d', 'data_processor3d_gray']


def aug_params(if_train):
    """Get augmentation parameters."""
    flip = 0  # flipping
    pn = np.ones(3)  # per channel pixel-noise
    rot = 0  # rotation
    sc = 1  # scaling
    if if_train:
        # We flip with probability 1/2
        if np.random.uniform() <= 0.5:
            flip = 1

        # Each channel is multiplied with a number
        # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
        pn = np.random.uniform(1 - NOISE_FACTOR, 1 + NOISE_FACTOR, 3)

        # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
        rot = min(2 * ROT_FACTOR, max(-2 * ROT_FACTOR, np.random.randn() * ROT_FACTOR))

        # The scale is multiplied with a number
        # in the area [1-scaleFactor,1+scaleFactor]
        sc = min(1 + SCALE_FACTOR,
                 max(1 - SCALE_FACTOR, np.random.randn() * SCALE_FACTOR + 1))
        # but it is zero with probability 3/5
        if np.random.uniform() <= 0.6:
            rot = 0

    return flip, pn, rot, sc


def rgb_processing(rgb_img, center, scale, rot, flip, pn):
    """Process rgb image and do augmentation."""
    rgb_img = crop(rgb_img, center, scale,
                  [IMG_RES, IMG_RES], rot=rot)
    # flip the image
    if flip:
        rgb_img = flip_img(rgb_img)
    # in the rgb image we add pixel noise in a channel-wise manner
    rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
    rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
    rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
    # (3,224,224),float,[0,1]
    # rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0

    rgb_img = Image.fromarray(rgb_img)

    return rgb_img


def gray_processing(gray_img, center, scale, rot, flip, pn):
    """Process rgb image and do augmentation."""
    gray_img = crop(gray_img, center, scale,
                  [IMG_RES, IMG_RES], rot=rot)
    # flip the image
    if flip:
        gray_img = flip_img(gray_img)

    gray_img[:,:] = np.minimum(255.0, np.maximum(0.0, gray_img[:,:]*pn[0]))

    gray_img = Image.fromarray(gray_img)

    return gray_img

def silhouette_processing(silhouette_img, center, scale, rot, flip):
    """Process silhouette image and do augmentation."""
    silhouette_img = crop(silhouette_img, center, scale,
                  [IMG_RES, IMG_RES], rot=rot, interp='nearest')
    # flip the image
    if flip:
        silhouette_img = flip_img(silhouette_img)

    # (3,224,224),float,[0,1]
    # silhouette_img = np.transpose(silhouette_img.astype('float32'),(2,0,1))/255.0


    silhouette_img = Image.fromarray(silhouette_img)

    return silhouette_img


def j2d_processing(kp, center, scale, r, f):
    """Process gt 2D keypoints and apply all augmentation transforms."""
    nparts = kp.shape[0]
    for i in range(nparts):
        kp[i,0:2] = transform(kp[i,0:2]+1, center, scale,
                              [IMG_RES, IMG_RES], rot=r)

    # convert to normalized coordinates to [-normalize_factor, normalize_factor]
    normalize_factor = 3.0
    kp[:,:-1] = (2.*kp[:,:-1]/IMG_RES - 1.) * normalize_factor

    # flip the x coordinates
    if f:
         kp = flip_kp(kp)
    kp = kp.astype('float32')

    return kp


def data_processor3d(img, joints=None, mask=None, if_train=False):
    img = np.array(img)

    # Assume that the person is centerered in the image
    height = img.shape[0]
    width = img.shape[1]
    center = np.array([width // 2, height // 2])
    scale = max(height, width) * 1.0 / 200

    # Get augmentation parameters
    flip, pn, rot, sc = aug_params(if_train)

    # Process image
    img = rgb_processing(img, center, sc * scale, rot, flip, pn)

    # Process joints
    joints = j2d_processing(joints, center, sc*scale, rot, flip)

    if mask is not None:
        # Assume that the person is centerered in the mask
        mask = np.array(mask)

        mask_height = mask.shape[0]
        mask_width = mask.shape[1]
        mask_center = np.array([mask_width // 2, mask_height // 2])
        mask_scale = max(mask_height, mask_width) * 1.0 / 200

        mask = silhouette_processing(mask, mask_center, sc * mask_scale, rot, flip)

        return img, joints, mask

    return img, joints


def data_processor3d_gray(gray_img, joints=None, mask=None, if_train=False):
    gray_img = np.array(gray_img)

    # Assume that the person is centerered in the image
    height = gray_img.shape[0]
    width = gray_img.shape[1]
    center = np.array([width // 2, height // 2])
    scale = max(height, width) * 1.0 / 200

    # Get augmentation parameters
    flip, pn, rot, sc = aug_params(if_train)

    # Process image
    gray_img = gray_processing(gray_img, center, sc * scale, rot, flip, pn)

    # Process joints
    joints = j2d_processing(joints, center, sc*scale, rot, flip)

    if mask is not None:
        # Assume that the person is centerered in the mask
        mask = np.array(mask)

        mask_height = mask.shape[0]
        mask_width = mask.shape[1]
        mask_center = np.array([mask_width // 2, mask_height // 2])
        mask_scale = max(mask_height, mask_width) * 1.0 / 200

        mask = silhouette_processing(mask, mask_center, sc * mask_scale, rot, flip)

        return gray_img, joints, mask

    return gray_img, joints

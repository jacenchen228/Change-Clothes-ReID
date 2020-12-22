import os
import os.path as osp
import errno
import warnings
import PIL
from PIL import Image
import random
import numpy as np
import pickle
import sys

import torch


def read_image(path, flag=False):
    """Reads image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.
        flag (bool): flag determining whether loading RGB images.
    Returns:
        PIL image
    """
    got_img = False
    if not osp.exists(path):
        raise IOError('"{}" does not exist'.format(path))
    while not got_img:
        try:
            if flag == True:
                img = Image.open(path).convert('RGB')
            else:
                img = Image.open(path)
            got_img = True
        except IOError:
            print('IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'.format(path))
            pass
    return img


def check_isfile(fpath):
    """Checks if the given path is a file.

    Args:
        fpath (str): file path.

    Returns:
       bool
    """
    isfile = osp.isfile(fpath)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(fpath))
    return isfile


def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def collect_env_info():
    """Returns env info as a string.

    Code source: github.com/facebookresearch/maskrcnn-benchmark
    """
    from torch.utils.collect_env import get_pretty_env_info
    env_str = get_pretty_env_info()
    env_str += '\n        Pillow ({})'.format(PIL.__version__)
    return env_str


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_smpl_param(smpl_path, if_pose=False):
    with open(smpl_path, 'rb') as f:
        res = pickle.load(f,  encoding='iso-8859-1')

    shape_param = res['betas']
    shape_param = torch.Tensor(shape_param)

    if if_pose:
        pose_param = res['pose']
        pose_param = torch.Tensor(pose_param)

        param = torch.cat([pose_param, shape_param], dim=0)

        return param

    return shape_param

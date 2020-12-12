from .dataset import *
from .dataset import *
from .prcc3d import *
from .prcc_3d_relabel import *
from .market import *
from .vc_clothes import *

__image_datasets = {
    'prcc3d': PRCC3D,
    'prcc_3d_relabel': PRCC_3D_RELABEL,
    'market': Market,
    'vc-clothes': VC_Clothes
}


def init_image_dataset(name, **kwargs):
    """Initializes an image dataset."""
    avai_datasets = list(__image_datasets.keys())
    if name not in avai_datasets:
        raise ValueError('Invalid dataset name. Received "{}", '
                         'but expected to be one of {}'.format(name, avai_datasets))
    return __image_datasets[name](**kwargs)


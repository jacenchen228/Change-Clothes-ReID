from .dataset import *
from .dataset import *
from .prcc import *
from .market import *
from .dukemtmcreid import *
from .vc_clothes import *
from .biwi import *
from .biwi_walking import *

__image_datasets = {
    'prcc': PRCC,
    'market': Market,
    'dukemtmc': DukeMTMCreID,
    'vc-clothes': VC_Clothes,
    'biwi': BIWI,
    'biwi_walking': BIWI_Walking
}


def init_image_dataset(name, **kwargs):
    """Initializes an image dataset."""
    avai_datasets = list(__image_datasets.keys())
    if name not in avai_datasets:
        raise ValueError('Invalid dataset name. Received "{}", '
                         'but expected to be one of {}'.format(name, avai_datasets))
    return __image_datasets[name](**kwargs)


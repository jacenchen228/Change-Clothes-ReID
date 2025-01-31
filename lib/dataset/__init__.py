from .dataset import *
from .dataset import *
from .prcc import *
from .prcc_unchange import *
from .market import *
from .dukemtmcreid import *
from .vc_clothes import *
from .biwi import *
from .biwi_walking import *
from .msmt17 import *
from .ltcc import *
from .ltcc_change import *

__image_datasets = {
    'prcc': PRCC,
    'prcc_unchange': PRCC_Unchange,
    'market': Market,
    'dukemtmc': DukeMTMCreID,
    'msmt': MSMT17,
    'vc-clothes': VC_Clothes,
    'biwi': BIWI,
    'biwi_walking': BIWI_Walking,
    'ltcc': LTCC,
    'ltcc_change': LTCC_Change
}


def init_image_dataset(name, **kwargs):
    """Initializes an image dataset."""
    avai_datasets = list(__image_datasets.keys())
    if name not in avai_datasets:
        raise ValueError('Invalid dataset name. Received "{}", '
                         'but expected to be one of {}'.format(name, avai_datasets))
    return __image_datasets[name](**kwargs)


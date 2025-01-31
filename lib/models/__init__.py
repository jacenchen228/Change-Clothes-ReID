from __future__ import absolute_import


from .resnet import *
from .resnet50_part import *
from .resnet50_part_mgn import *
from .baseline import *
from .dim_gcn_model import *
from .dim_gcn_new import *
from .dim_gcn_multi import *
from .dim_global_gcn import *
from .dim_part_gcn import *
from .dim_gcnplain_multi import *
from .dim_gcn_single import *
from .dim_multigcn_multi import *
from .gcn_model import *
from .dim_model import *
from .dim_multi import *
from .pcb import *

__model_factory = {
    'resnet50': resnet50,
    'resnet50_part': resnet50_part,
    'resnet50_part_mgn': resnet50_part_mgn,
    'resnet34_part': resnet34_part,
    'baseline': my_baseline,
    'gcn_model': gcn_model,
    'gcn_model34': gcn_model_contour34,
    'dim_model50': dim_model50,
    'dim_gcn_model': dim_gcn_model,
    'dim_gcn_new50': dim_gcn_new50,
    'dim_gcn_new34': dim_gcn_new34,
    'dim_gcn_multi50': dim_gcn_multi50,
    'dim_gcn_multi34': dim_gcn_multi34,
    'dim_multigcn_multi50': dim_multigcn_multi50,
    'dim_global_gcn50': dim_global_gcn50,
    'dim_part_gcn50': dim_part_gcn50,
    'dim_gcnplain_multi50': dim_gcnplain_multi50,
    'dim_gcn_single50': dim_gcn_single50,
    'dim_multi50': dim_multi50,
    'pcb_p4': pcb_p4,
    'pcb_p6': pcb_p6
}


def show_avai_models():
    """Displays available models.

    Examples::
        >>> from torchreid import models
        >>> models.show_avai_models()
    """
    print(list(__model_factory.keys()))


def build_model(name, num_classes, loss='softmax', pretrained=True, use_gpu=True, **kwargs):
    """A function wrapper for building a model.

    Args:
        name (str): model name.
        num_classes (int): number of training identities.
        loss (str, optional): loss function to optimize the model. Currently
            supports "softmax" and "triplet". Default is "softmax".
        pretrained (bool, optional): whether to load ImageNet-pretrained weights.
            Default is True.
        use_gpu (bool, optional): whether to use gpu. Default is True.

    Returns:
        nn.Module

    Examples::
        >>> from torchreid import models
        >>> model = models.build_model('resnet50', 751, loss='softmax')
    """
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError('Unknown model: {}. Must be one of {}'.format(name, avai_models))
    return __model_factory[name](
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu,
        **kwargs
    )
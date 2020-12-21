from __future__ import absolute_import

import torch

from .resnet import *
from .resnetmid import *
from .senet import *
from .densenet import *
from .inceptionresnetv2 import *
from .inceptionv4 import *
from .xception import *

from .nasnet import *
from .mobilenetv2 import *
from .shufflenet import *
from .squeezenet import *
from .shufflenetv2 import *

from .mudeep import *
from .hacnn import *
from .pcb import *
from .mlfn import *
from .osnet import *

from .dim import *
from .dim_wo_bnneck import *
from .pcb_dim import *
# from .dim_part import *
from .dim_gnn import *
from .dim_gnn_hie import *
from .dim_part_gnn_hie import *
from .dim_part_gnn_hie1 import *
from .dim_part_gnn_hie2 import *
from .baseline_dim import *
# from .dim_part_split import *
from .my_baseline import *
from .baseline_gnn import *
from .baseline_gnn_hie import *

from .mhn_ide import *
from .hpm import *
from .mgn import *
from .dml import *
from .depth_two_path import depth_two_path

__model_factory = {
    # image classification models
    'resnet50': resnet50,
    'se_resnet50': se_resnet50,
    'se_resnet50_fc512': se_resnet50_fc512,
    'se_resnet101': se_resnet101,
    'se_resnext50_32x4d': se_resnext50_32x4d,
    'se_resnext101_32x4d': se_resnext101_32x4d,
    'densenet121': densenet121,
    'densenet169': densenet169,
    'densenet201': densenet201,
    'densenet161': densenet161,
    'densenet121_fc512': densenet121_fc512,
    'inceptionresnetv2': inceptionresnetv2,
    'inceptionv4': inceptionv4,
    'xception': xception,
    # lightweight models
    'nasnsetmobile': nasnetamobile,
    'mobilenetv2_x1_0': mobilenetv2_x1_0,
    'mobilenetv2_x1_4': mobilenetv2_x1_4,
    'shufflenet': shufflenet,
    'squeezenet1_0': squeezenet1_0,
    'squeezenet1_0_fc512': squeezenet1_0_fc512,
    'squeezenet1_1': squeezenet1_1,
    'shufflenet_v2_x0_5': shufflenet_v2_x0_5,
    'shufflenet_v2_x1_0': shufflenet_v2_x1_0,
    'shufflenet_v2_x1_5': shufflenet_v2_x1_5,
    'shufflenet_v2_x2_0': shufflenet_v2_x2_0,
    # reid-specific models
    'mudeep': MuDeep,
    'resnet50mid': resnet50mid,
    'hacnn': HACNN,
    'pcb_p6': pcb_p6,
    'pcb_p4': pcb_p4,
    'mlfn': mlfn,
    'osnet_x1_0': osnet_x1_0,
    'osnet_x0_75': osnet_x0_75,
    'osnet_x0_5': osnet_x0_5,
    'osnet_x0_25': osnet_x0_25,
    'osnet_ibn_x1_0': osnet_ibn_x1_0,
    # dim
    'dim_graph_resnet50_C': dim_graph_resnet50_C,
    'dim_graph_hie_resnet50_C': dim_graph_hie_resnet50_C,
    'dim_part_graph_hie': dim_part_graph_hie_resnet50_C,
    'dim_part_graph_hie1': dim_part_graph_hie_resnet50_C1,
    'dim_part_graph_hie_abla': dim_part_graph_hie_abla,
    'dim_graph_resnet50_C_wobnneck': dim_graph_resnet50_C_wobnneck,
    'my_baseline_dim': my_baseline_dim,
    'my_baseline': my_baseline,
    'my_baseline_gnn': my_baseline_gnn,
    'my_baseline_gnn_hie': my_baseline_gnn_hie,
    # pcb-based dim
    'pcb_p6_dim_contour': pcb_p6_dim_contour,
    'pcb_p6_dim_img': pcb_p6_dim_img,
    'pcb_p4_dim_contour': pcb_p4_dim_contour,
    'pcb_p4_dim_img': pcb_p4_dim_img,
    # sota
    'mhn6': mhn6,
    'hpm': hpm,
    'mgn': mgn,
    'dml': dml,
    # depth variant
    'depth_two_path': depth_two_path
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
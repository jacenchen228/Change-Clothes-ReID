from __future__ import absolute_import

from .resnet import resnet50
from .resnet_2stream import resnet50_2stream
# from .model_3D import resnet50_3D
# from .model_3D_2stream import model_3D_2stream
from .my_3d_branch import my_3d_branch
from .model_3D_2stream_v1 import model_3D_2stream_v1
from .model_3D_2stream_v2 import model_3D_2stream_v2
from .model_3D_2stream_v3 import model_3D_2stream_v3
from .model_3D_2stream_v4 import model_3D_2stream_v4
from .model_2stream_v3 import model_2stream_v3
from .model_2stream_part import model_2stream_part
from .model_3D_2stream_tmp import model_3D_2stream_tmp
from .mgn import mgn
from .mgn_strong import MGN_Strong
from .mgn3d import MGN3D
from .mhn_ide import mhn6
from .pcb import pcb_p4, pcb_p6
from .pcb_2stream import pcbp4_2stream, pcbp6_2stream
from .reid_3d_branch import reid_3d_branch
from .resnet_strong import resnet50_strong

__model_factory = {
    # image classification models
    'resnet50': resnet50,
    'resnet50_2stream': resnet50_2stream,
    'resnet50_strong': resnet50_strong,
    # 'resnet50_3D': resnet50_3D,
    # 'model_3D_2stream': model_3D_2stream,
    'model_3D_2stream_v1': model_3D_2stream_v1,
    'model_3D_2stream_v2': model_3D_2stream_v2,
    'model_3D_2stream_v3': model_3D_2stream_v3,
    'model_3D_2stream_v4': model_3D_2stream_v4,
    'model_2stream_v3': model_2stream_v3,
    'model_2stream_part': model_2stream_part,
    'model_3D_2stream_tmp': model_3D_2stream_tmp,
    'my_3D_branch': my_3d_branch,
    'reid_3d_branch': reid_3d_branch,
    'mgn': mgn,
    'mhn': mhn6,
    'pcb_p4': pcb_p4,
    'pcb_p6': pcb_p6,
    'pcbp4_2stream': pcbp4_2stream,
    'pcbp6_2stream': pcbp6_2stream
}


def show_avai_models():
    """Displays available models.

    Examples::
        >>> from lib import models
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
        >>> from lib import models
        >>> model = models.build_model('resnet50', 751, loss='softmax')
    """
    avai_models = list(__model_factory.keys())
    if name == 'resnet50_strong':
        return resnet50_strong(pretrained=True, num_classes=num_classes)
    if name == 'mgn3d':
        return MGN3D(num_classes=num_classes, num_layers=50, **kwargs)
    if name == 'mgn_strong':
        return MGN_Strong(num_classes=num_classes, num_layers=50)
    if name not in avai_models:
        raise KeyError('Unknown model: {}. Must be one of {}'.format(name, avai_models))
    return __model_factory[name](
        num_classes=num_classes,
        pretrained=pretrained,
        use_gpu=use_gpu,
        **kwargs
    )

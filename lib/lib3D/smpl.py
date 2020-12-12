import torch
import numpy as np
import smplx
from smplx import SMPL as _SMPL
from smplx.body_models import ModelOutput
from smplx.lbs import vertices2joints

JOINT_REGRESSOR_TRAIN_EXTRA = '/home/jiaxing/SPIN-master/data/J_regressor_extra.npy'

# # LSP order
# JOINT_NAMES = [
# 'Right Ankle',
# 'Right Knee',
# 'Right Hip',
# 'Left Hip',
# 'Left Knee',
# 'Left Ankle',
# 'Right Wrist',
# 'Right Elbow',
# 'Right Shoulder',
# 'Left Shoulder',
# 'Left Elbow',
# 'Left Wrist',
# 'Neck (LSP)',
# 'Top of Head (LSP)'
# ]

# COCO order
JOINT_NAMES = [
'Nose',
'Left Eye',
'Right Eye',
'Left Ear',
'Right Ear',
'Left Shoulder',
'Right Shoulder',
'Left Elbow',
'Right Elbow',
'Left Wrist',
'Right Wrist',
'Left Hip',
'Right Hip',
'Left Knee',
'Right Knee',
'Left Ankle',
'Right Ankle'
]

# Map joints to SMPL joints
JOINT_MAP = {
'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
'Spine (H36M)': 51, 'Jaw (H36M)': 52,
'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}


class SMPL(_SMPL):
    """ Extension of the official SMPL  implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        joints = [JOINT_MAP[i] for i in JOINT_NAMES]
        J_regressor_extra = np.load(JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)

        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        output = ModelOutput(vertices=smpl_output.vertices,
                             global_orient=smpl_output.global_orient,
                             body_pose=smpl_output.body_pose,
                             joints=joints,
                             betas=smpl_output.betas,
                             full_pose=smpl_output.full_pose)
        return output

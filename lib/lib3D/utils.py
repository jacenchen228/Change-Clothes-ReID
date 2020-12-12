import glob
import os.path as osp
import numpy as np

import torch

from .constants import  *


def batch_rodrigues(theta):
    # theta N x 3
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)

    return quat2mat(quat)


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.
    K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]


def proj_2djoint(pred_cam, pred_joints):
    """
    Project estimated 2d joints according to specific perspectives
    """
    batch_size = pred_cam.shape[0]

    # Convert Weak Perspective Camera [s, tx, ty] to
    # camera translation [tx, ty, tz] in 3D given the bounding box size
    # This camera translation can be used in a full perspective projection
    pred_cam_t = torch.stack([pred_cam[:, 1],
                              pred_cam[:, 2],
                              2 * FOCAL_LENGTH / (IMG_RES * pred_cam[:, 0] + 1e-9)], dim=-1)

    camera_center = torch.zeros(batch_size, 2).cuda()
    pred_keypoints_2d = perspective_projection(pred_joints,
                                               rotation=torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).cuda(),
                                               translation=pred_cam_t,
                                               focal_length=FOCAL_LENGTH,
                                               camera_center=camera_center)
    # Normalize keypoints to [-normalize_factor, normalize_factor]
    normalize_factor = 3.0
    pred_keypoints_2d = pred_keypoints_2d / (IMG_RES / 2.) * normalize_factor

    return pred_keypoints_2d


class RealMotionWarpper(object):
    def __init__(self, dir_name):
        file_list = glob.glob(osp.join(dir_name, '*', '*.npz'))
        all_data = list()

        # extract SMPL joints from SMPL-H model
        joints_to_use = np.array([
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 37
        ])
        self.joints_to_use = np.arange(0, 156).reshape((-1, 3))[joints_to_use].reshape(-1)

        for file_name in file_list:
            if file_name.endswith('shape.npz'):
                continue

            data = np.load(file_name)
            pose = data['poses'][:, self.joints_to_use]
            shape = np.repeat(data['betas'][:10][np.newaxis], pose.shape[0], axis=0)

            theta = np.concatenate([pose, shape], axis=1)
            all_data.append(theta)

        self.data = np.concatenate(all_data, axis=0).astype(np.float32)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]

# def read_real_motion(dir_name):
#     file_list = glob.glob(osp.join(dir_name, '*', '*.npz'))
#     all_data = list()
#
#     # extract SMPL joints from SMPL-H model
#     joints_to_use = np.array([
#         0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
#         11, 12, 13, 14, 15, 16, 17, 18, 19,
#         20, 21, 22, 37
#     ])
#
#     for file_name in file_list:
#         if file_name.endswith('shape.npz'):
#             continue
#
#         data = np.load(file_name)
#         pose = data['poses'][:, joints_to_use]
#         shape = np.repeat(data['betas'][:10][np.newaxis], pose.shape[0], axis=0)
#
#         theta = np.concatenate([pose, shape], axis=1)
#         all_data.append(theta)
#
#     all_data = np.concatenate(all_data, axis=0)
#
#     return all_data

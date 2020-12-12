import torch

from .util import batch_rodrigues


def shape_l2_loss(real_shape, predict_shape, use_gpu=True):
    """
    :param real_shape: N x 10
    :param predict_shape: N x 10
    :param use_gpu:
    :return:
    """
    w_shape = torch.ones((real_shape.shape[0])).float()
    if use_gpu:
        w_shape = w_shape.cuda()
    k = torch.sum(w_shape) * 10.0 * 2.0 + 1e-8
    shape_dif = (real_shape - predict_shape) ** 2
    return torch.matmul(shape_dif.sum(1), w_shape) * 1.0 / k


def pose_l2_loss(real_pose, predict_pose, use_gpu=True):
    """
    :param real_pose: N x 72
    :param predict_pose: N x 72
    :param use_gpu:
    :return:
    """
    w_pose = torch.ones((real_pose.shape[0])).float()
    if use_gpu:
        w_pose = w_pose.cuda()
    k = torch.sum(w_pose) * 207.0 * 2.0 + 1e-8
    real_rs, fake_rs = batch_rodrigues(real_pose.contiguous().view(-1, 3)).view(-1, 24, 9)[:, 1:, :], batch_rodrigues(
        predict_pose.contiguous().view(-1, 3)).view(-1, 24, 9)[:, 1:, :]
    dif_rs = ((real_rs - fake_rs) ** 2).view(-1, 207)
    return torch.matmul(dif_rs.sum(1), w_pose) * 1.0 / k


def kp_2d_l1_loss(real_2d_kp, predict_2d_kp):
    """
    :param self:
    :param real_2d_kp: N x K x 3
    :param predict_2d_kp: N x K x 2
    :return:
    """
    kp_gt = real_2d_kp.view(-1, 3)
    kp_pred = predict_2d_kp.contiguous().view(-1, 2)
    vis = kp_gt[:, 2]
    k = torch.sum(vis) * 2.0 + 1e-8
    dif_abs = torch.abs(kp_gt[:, :2] - kp_pred).sum(1)
    return torch.matmul(dif_abs, vis) * 1.0 / k

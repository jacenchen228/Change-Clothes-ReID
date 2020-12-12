import torch.nn as nn
from lib.lib3D import hmr_new, SMPL, Discriminator, batch_rodrigues

SMPL_MODEL_DIR = '/home/jiaxing/SPIN-master/data/smpl'


class My3DBranch(nn.Module):
    def __init__(self, batch_size=16, **kwargs):
        super(My3DBranch, self).__init__()

        self.batch_size = batch_size

        # Load SMPL model
        self.smpl = SMPL(SMPL_MODEL_DIR,
                    batch_size=self.batch_size,
                    create_transl=False)

        # Create HMR model and ReID backbone
        self.estimator3D = hmr_new(smpl_mean_params='/home/jiaxing/SPIN-master/data/smpl_mean_params.npz', pretrained=False)
        # self.estimator_3D = hmr_new(smpl_mean_params='/home/jiaxing/SPIN-master/data/smpl_mean_params.npz', pretrained=False)
        self.discriminator = Discriminator()

    def forward(self, inputs, real_params=None, return_featuremaps=False, return_params_3D=False):
        v_3d, pred_rotmats, pred_betas, pred_cam, pred_displace = self.estimator3D(inputs)
        pred_params = {'rotmat':pred_rotmats,
                       'beta':pred_betas,
                       'cam':pred_cam}

        pred_displace = pred_displace.view(pred_displace.shape[0], 6890, 3)

        pred_outputs1 = self.smpl(betas=pred_betas, body_pose=pred_rotmats[:,1:],
                                 global_orient=pred_rotmats[:,0].unsqueeze(1), pose2rot=False, v_personal=pred_displace)
        pred_outputs2 = self.smpl(betas=pred_betas, body_pose=pred_rotmats[:,1:],
                                 global_orient=pred_rotmats[:,0].unsqueeze(1), pose2rot=False)

        if return_params_3D:
            return pred_rotmats, pred_betas, pred_cam, pred_outputs1

        if not self.training:
            return [v_3d]

        # discriminator output
        encoder_disc_value = self.discriminator(pred_betas, pred_rotmats.view(-1, 24, 9))
        gen_disc_value = self.discriminator(pred_betas.detach(), pred_rotmats.detach().view(-1, 24, 9))

        real_poses, real_shapes = real_params[:, :72], real_params[:, 72:]
        real_rotmats = batch_rodrigues(real_poses.contiguous().view(-1, 3)).view(-1, 24, 9)
        real_disc_value = self.discriminator(real_shapes, real_rotmats)

        return v_3d, pred_params, pred_outputs1, pred_outputs2, encoder_disc_value, gen_disc_value, real_disc_value
        # return v_3d, pred_params, pred_outputs1, encoder_disc_value, gen_disc_value, real_disc_value


def my_3d_branch(**kwargs):
    model = My3DBranch(**kwargs)

    return model

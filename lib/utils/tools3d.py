import os
import os.path as osp
import numpy as np
import cv2
import sys
from PIL import Image

import torch

# from lib.lib3D import Renderer, FOCAL_LENGTH, IMG_RES
from lib.lib3D import FOCAL_LENGTH, IMG_RES
from lib.lib3D.render_pytorch import Renderer
# from lib.hmr import SMPLRenderer, SMPL, SMPL_MODEL

__all__ = ['save_3d_param', 'visualize_smpl', 'visualize_3d_silhouette', 'visualize_3d_rgb']

@torch.no_grad()
def save_3d_param(dataloader, model, print_freq, use_gpu, save_dir, **kwargs):
    """Save 3D parameters.

    """
    model.eval()

    for batch_idx, data in enumerate(dataloader):
        # imgs, paths = data[0], data[3]
        imgs, paths, segments = data[0], data[3], data[6]

        if use_gpu:
            imgs = imgs.cuda()
            segments = segments.cuda()

        # forward to get convolutional feature maps
        try:
            # outputs = model(segments, imgs, return_params_3D=True)
            outputs = model(imgs, imgs, return_params_3D=True)
        except TypeError:
            raise TypeError('forward() got unexpected keyword argument "return_params_3D". ' \
                            'Please add return_params_3D as an input argument to forward(). When ' \
                            'return_params_3D=True, return feature maps only.')

        if use_gpu:
            outputs = outputs.cpu()
        outputs = outputs.numpy()

        for j in range(outputs.shape[0]):
            # get save path
            path = paths[j]
            path = path.split('/rgb/')
            save_path = osp.join(save_dir, 'smpl', path[-1]).replace('.jpg', '.npy')
            save_sub_dir = osp.dirname(save_path)
            if not osp.exists(save_sub_dir):
                os.makedirs(save_sub_dir)

            # save 3D param
            param_3d = outputs[j]
            np.save(save_path, param_3d)

        if (batch_idx + 1) % print_freq == 0:
            print('- done batch {}/{}'.format(batch_idx + 1, len(dataloader)))


# def visualize_smpl(gpu_id):
#     from lib.hmr import SMPL, SMPL_MODEL
#     from lib.hmr.render import SMPLRenderer
#
#     # smpl = SMPL(SMPL_MODEL, obj_saveable=True).to(gpu_id)
#     # smpl = SMPL(SMPL_MODEL, obj_saveable=True, joint_type='lsp').to(gpu_id)
#     smpl = SMPL(SMPL_MODEL).to(gpu_id)
#     renderer = SMPLRenderer(face_path='/home/jiaxing/MyReID/lib/hmr/hmr_pretrain/smpl_faces.npy')
#
#     # data_dir = '/home/jiaxing/MyReID/log-bagoftricks/prcc_3d-resnet_3d-two_path_segImg-rgb_3d_cent_tri/amsgrad-lr0.0008-bs32-height256-seed2-alpha0.5-pretrain3d_5epochs/smpl'
#     data_dir = '/home/jiaxing/MyReID/log/prcc_3d-resent_3d-two_path_twoImgs-rgb_3d_cent_tri/amsgrad-lr0.0008-bs32-height256-seed2-alpha0.5-fused_concate-imagenet_init/smpl'
#     results_list = glob.glob(osp.join(data_dir, '*', '*', '*.npy'))
#
#     flength = 500
#     principal_pt = np.array([112, 112])
#
#     for idx, result_path in enumerate(results_list):
#         # res = np.load(result_path)
#
#         res = np.array([ 1.07094193e+00, 5.29803801e-03, 2.47314319e-01,3.25350642e+00,
#   -9.23386365e-02, 4.43265364e-02, 3.13512348e-02, 5.48436716e-02,
#    5.71859814e-03, -3.93712372e-02, -8.32256451e-02, -8.38281736e-02,
#    3.35151106e-02, -1.22006563e-03, 2.43517440e-02, 4.35255229e-01,
#   -7.59129375e-02, -7.00019300e-02,  5.95253050e-01,  3.40036228e-02,
#    5.85786067e-03, -5.10209575e-02, -1.56931998e-03, -4.46985923e-02,
#   -5.12763038e-02,  1.49789810e-01,  1.43658184e-02, -1.40282601e-01,
#   -3.09749216e-01,  1.81755766e-01,  1.00048818e-01, -4.19249795e-02,
#   -3.34277712e-02, -3.08114082e-01,  2.11470857e-01,  1.65950298e-01,
#   -1.87955707e-01,  2.46411234e-01, -3.93227816e-01, -1.77408397e-01,
#    5.69788255e-02,  3.04902866e-02,  7.76196718e-02,  1.54165313e-01,
#   -3.40378374e-01, -2.57011876e-03, -2.37160921e-03,  2.70848066e-01,
#    2.63628870e-01,  2.80512013e-02, -9.90556926e-02,  7.68299848e-02,
#   -2.22632065e-01, -1.07450223e+00,  2.86371201e-01,  2.26451606e-01,
#    1.10627103e+00,  4.14414287e-01, -8.30454528e-01,  1.64493516e-01,
#    2.13461325e-01,  8.77676845e-01, -1.29248768e-01, -8.93108770e-02,
#   -2.18692012e-02, -9.71929878e-02,  3.17001343e-03,  3.53618264e-02,
#    7.35158548e-02, -1.21466644e-01,-4.33226973e-02, -1.00613669e-01,
#   -6.65447935e-02,  9.80167836e-02,  1.37097672e-01,  1.42710876e+00,
#   -3.17499578e-01,  5.68052411e-01,  3.71869874e+00,  2.07222128e+00,
#    2.81292170e-01, -2.33484656e-01,  4.85356182e-01,  4.75008696e-01,
#   -8.57853949e-01])
#
#         print('param mean is {}'.format(np.mean(res)))
#
#         cam = res[0:3]
#         # cam = np.array([5.52277687e-03, 1.68048273e-01, 1.48606132e+01])
#         pose = torch.Tensor(res[3:75]).cuda(gpu_id)
#         shape = torch.Tensor(res[75:]).cuda(gpu_id)
#
#         pose = pose.view(1, -1)
#         shape = shape.view(1, -1)
#
#         verts, _, _ = smpl(beta=shape, theta=pose, get_skin=True)
#
#         # calculate shifted verts
#         verts = verts.cpu().numpy()
#         print('mean of verts is ', np.mean(verts))
#         cam_pos = cam[1:]
#         # trans = np.hstack([cam_pos, 4.1685])
#         trans = np.array([0.00529804, 0.24731432, 4.16856004])
#         verts_shifted = verts + trans
#
#         # generate cam params for render
#         # cam_for_render = np.hstack([flength, principal_pt])
#         cam_for_render = np.array([618.30357143, 45.75446429, 137.26339286])
#
#         rend_img = renderer(verts_shifted, cam=cam_for_render, img_size=(277, 93))
#
#         if idx < 5:
#             plt.imshow(rend_img)
#             plt.axis('off')
#             plt.show()
#             plt.pause(1)
#             input('Press any key to continue...')
#
#         cv2.imwrite(result_path.replace('.npy', '.png'), rend_img)

@torch.no_grad()
def visualize_smpl(model, dataloader, save_dir):
    model.eval()

    # Setup renderer for visualization
    renderer = Renderer(focal_length=FOCAL_LENGTH, img_res=IMG_RES, faces=model.module.smpl.faces)

    # Data struct for rendering side views
    aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0]

    # Load data to estimate 3d shape and visualization
    loader_len = len(dataloader)
    for batch_idx, batch_item in enumerate(dataloader):
        img_paths, imgs_3d, imgs_3d_normed = batch_item[3], batch_item[6], batch_item[7]
        pred_rotmats, pred_betas, pred_cameras, pred_outputs = model(imgs_3d_normed.cuda(), return_params_3D=True)
        pred_vertices_all = pred_outputs.vertices

        camera_translations = torch.stack([pred_cameras[:, 1], pred_cameras[:, 2],
                                          2 * FOCAL_LENGTH / (IMG_RES * pred_cameras[:, 0] + 1e-9)],
                                         dim=-1)

        for sample_idx in range(imgs_3d.shape[0]):
            # Calculate camera parameters for rendering
            camera_translation = camera_translations[sample_idx].cpu().numpy()
            pred_vertices = pred_vertices_all[sample_idx].cpu().numpy()
            img_3d = imgs_3d[sample_idx].permute(1, 2, 0).numpy()
            img_path = img_paths[sample_idx]

            # Render parametric shape
            img_shape = renderer(pred_vertices, camera_translation, img_3d)

            # Render side views
            center = pred_vertices.mean(axis=0)
            rot_vertices = np.dot((pred_vertices - center), aroundy) + center

            # Render non-parametric shape
            img_shape_side = renderer(rot_vertices, camera_translation, np.ones_like(img_3d))

            rel_path = img_path.split('/rgb/')[-1]
            save_sub_dir = osp.join(save_dir+'smpl', osp.dirname(rel_path))
            if not osp.exists(save_sub_dir):
                os.makedirs(save_sub_dir)
            outfile = osp.join(save_sub_dir, osp.splitext(rel_path.split('/')[-1])[0])

            # print(outfile)

            # Save reconstructions
            cv2.imwrite(outfile + '_shape.png', 255 * img_shape[:, :, ::-1])
            cv2.imwrite(outfile + '_shape_side.png', 255 * img_shape_side[:, :, ::-1])

        print('{}/{} batches have been visualized!'.format(batch_idx+1, loader_len))@torch.no_grad()


@torch.no_grad()
def visualize_3d_silhouette(model, dataloader, save_dir, batch_size, use_gpu, **kwargs):
    model.eval()

    # Setup renderer for visualization
    renderer = Renderer(img_size=IMG_RES, faces=model.module.smpl.faces, use_gpu=use_gpu,
                        focal_length=FOCAL_LENGTH, batch_size=1)

    # Load data to estimate 3d shape and visualization
    loader_len = len(dataloader)
    for batch_idx, batch_item in enumerate(dataloader):
        if batch_idx > 5:
            break

        img_paths, imgs_3d, imgs_3d_normed = batch_item[3], batch_item[5], batch_item[6]
        pred_rotmats, pred_betas, pred_cameras, pred_outputs = model(imgs_3d_normed.cuda(), return_params_3D=True)
        pred_vertices_all = pred_outputs.vertices

        camera_translations = torch.stack([pred_cameras[:, 1], pred_cameras[:, 2],
                                          2 * FOCAL_LENGTH / (IMG_RES * pred_cameras[:, 0] + 1e-9)],
                                         dim=-1)

        for sample_idx in range(imgs_3d.shape[0]):
            # Calculate camera parameters for rendering
            camera_translation = camera_translations[sample_idx].unsqueeze(0)
            pred_vertices = pred_vertices_all[sample_idx].unsqueeze(0)
            img_path = img_paths[sample_idx]
            img = Image.open()

            # print(torch.mean(pred_vertices))

            # Apply camera translation on predicted vertices
            pred_vertices += camera_translation

            # # Since perspective projection would flip along X-axis
            # # here we manually flip the vertices ahead
            # rot_mat = torch.Tensor(
            #     [[-1.0, 0, 0],
            #      [0, -1.0, 0],
            #      [0, 0, 1]]
            # )
            # if use_gpu:
            #     rot_mat = rot_mat.cuda()
            # pred_vertices = torch.matmul(pred_vertices, rot_mat.T)

            # Render parametric shape
            img_shape = renderer.render_silhouette(pred_vertices)
            # print(torch.unique(img_shape))
            img_shape = img_shape.permute(1, 2, 0).cpu().numpy()
            img_shape[np.where((img_shape>0) & (img_shape<1))] = 1

            rel_path = img_path.split('/rgb/')[-1]
            save_sub_dir = osp.join(save_dir+'3d_silhouette', osp.dirname(rel_path))
            if not osp.exists(save_sub_dir):
                os.makedirs(save_sub_dir)
            outfile = osp.join(save_sub_dir, osp.splitext(rel_path.split('/')[-1])[0])


            cv2.imwrite(outfile + '_shape.png', 255 * img_shape)

        print('{}/{} batches have been visualized!'.format(batch_idx+1, loader_len))


@torch.no_grad()
def visualize_3d_rgb(model, dataloader, save_dir, batch_size, use_gpu, **kwargs):
    model.eval()

    # Setup renderer for visualization
    renderer = Renderer(img_size=IMG_RES, faces=model.module.smpl.faces, use_gpu=use_gpu,
                        focal_length=FOCAL_LENGTH, batch_size=1)

    # Load data to estimate 3d shape and visualization
    loader_len = len(dataloader)
    for batch_idx, batch_item in enumerate(dataloader):
        if batch_idx > 5:
            break

        imgs, img_paths, imgs_3d, imgs_3d_normed = batch_item[0], batch_item[3], batch_item[5], batch_item[6]
        pred_rotmats, pred_betas, pred_cameras, pred_outputs = model(imgs, imgs_3d_normed.cuda(), return_params3D=True)
        pred_vertices_all = pred_outputs.vertices

        camera_translations = torch.stack([pred_cameras[:, 1], pred_cameras[:, 2],
                                          2 * FOCAL_LENGTH / (IMG_RES * pred_cameras[:, 0] + 1e-9)],
                                         dim=-1)

        for sample_idx in range(imgs_3d.shape[0]):
            # Calculate camera parameters for rendering
            camera_translation = camera_translations[sample_idx].unsqueeze(0)
            pred_vertices = pred_vertices_all[sample_idx].unsqueeze(0)
            img_path = img_paths[sample_idx]
            img = Image.open(img_path)

            # Apply camera translation on predicted vertices
            pred_vertices += camera_translation

            # # To store the vertice 3D position
            # # print(pred_vertices.shape)
            # pred_vertices_arr = pred_vertices.squeeze(0).cpu().numpy()
            # vertices_path = img_path.replace('/rgb/', '/vertices_arr/').replace('.jpg', '.npy')
            # dir_name = osp.dirname(vertices_path)
            # if not osp.exists(dir_name):
            #     os.makedirs(dir_name)
            # np.save(vertices_path, pred_vertices_arr)
            #
            # continue

            # # Since perspective projection would flip along X-axis
            # # here we manually flip the vertices ahead
            # rot_mat = torch.Tensor(
            #     [[-1.0, 0, 0],
            #      [0, -1.0, 0],
            #      [0, 0, 1]]
            # )
            # if use_gpu:
            #     rot_mat = rot_mat.cuda()
            # pred_vertices = torch.matmul(pred_vertices, rot_mat.T)

            # Render parametric shape
            img_shape = renderer.render_rgb(pred_vertices)

            img_shape = img_shape.squeeze(0).permute(1, 2, 0).cpu().numpy()

            # rel_path = img_path.split('/rgb/')[-1]
            rel_path = img_path.split('/')[-1]
            save_sub_dir = osp.join(save_dir+'3d_rgb', osp.dirname(rel_path))
            if not osp.exists(save_sub_dir):
                os.makedirs(save_sub_dir)
            outfile = osp.join(save_sub_dir, osp.splitext(rel_path.split('/')[-1])[0])

            cv2.imwrite(outfile + '_shape.png', 255 * img_shape)
            img.save(outfile+'.jpg')

        print('{}/{} batches have been visualized!'.format(batch_idx+1, loader_len))
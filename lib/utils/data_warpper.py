import numpy as np
import cv2
import os
import os.path as osp
from PIL import Image

import torch
from torchvision.transforms import ToTensor

from lib.lib3D import data_processor3d, data_processor3d_gray
from lib.lib3D.constants import IMG_RES
from .tools import read_image, load_smpl_param


class DataWarpper(object):
    def __init__(self, data, transforms1, transforms2=None, if_train=True):
        self.data = data
        self.transforms1 = transforms1
        self.transforms2 = None
        if transforms2 is not None:
            self.transforms2 = transforms2
            # self.transforms_seg = ToTensor()
        self.if_train = if_train

        self.to_tensor = ToTensor()

        self.cnt = 0


    def __getitem__(self, idx):
        # img_path, pid, camid, mask_path, joints, segment_path, img = self.data[idx]
        img_path, pid, camid, mask_path, joints, img = self.data[idx]
        # img = read_image(img_path, True)

        # read silhouette
        if mask_path is None:
            mask = Image.fromarray(np.zeros((img.height, img.width)))
        else:
            mask = read_image(mask_path)

        # # read input data for 3d reconstruction
        img_3d, joints, mask = data_processor3d(img.copy(), joints, mask=mask, if_train=False)

        # if self.cnt < 100:
        #     mask_save_path = mask_path.replace('/data/jiaxing', '/home/jiaxing')
        #     mask_save_dir = osp.dirname(mask_save_path)
        #     if not osp.exists(mask_save_dir):
        #         os.makedirs(mask_save_dir)
        #     mask.save(mask_save_path)
        #     img_3d.save(mask_save_path.replace('.png', '.jpg'))
        #     self.cnt += 1

        img_3d_norm = self.transforms2(img_3d.copy())
        img_3d = self.to_tensor(img_3d)

        mask = self.to_tensor(mask)

        joints = torch.Tensor(joints)

        img = self.transforms1(img)
        # new_height, new_width = img.shape[1], img.shape[2]

        # ratio_width = float(new_width) / ori_width
        # ratio_height = float(new_height) / ori_height

        # segment = read_image(segment_path)
        # if self.transforms2 is not None:
        #     segment = self.transforms2(segment)

        # segment = np.load(segment_path)
        # segment = cv2.resize(segment, (new_width, new_height))
        # if self.transforms_seg is not None:
        #     segment = self.transforms_seg(segment)

        # ratio_width = 1.0 / ori_width
        # ratio_height = 1.0 / ori_height

        # param_3d = load_smpl_param(smpl_path, if_pose=True)
        # param_3d = load_smpl_param(smpl_path)

        # joints[:, 0] = joints[:, 0] * ratio_width
        # joints[:, 1] = joints[:, 1] * ratio_height

        # # load vertice-wise supervision
        # vertices = np.load(vertice_path)
        # vertices = torch.Tensor(vertices)

        return img, pid, camid, img_path, joints, img_3d, img_3d_norm, mask

    def __len__(self):
        return len(self.data)


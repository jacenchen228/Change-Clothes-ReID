from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
from .tools import read_image

class DataWarpper(object):
    def __init__(self, data, transforms):
        self.data = data
        self.transforms = transforms

    def __getitem__(self, idx):
        data_item = self.data[idx]

        if len(data_item) == 5:
            img_path, pid, camid, img, contour_img = data_item
            img, contour_img = self.transforms(img, contour_img)
            return img, contour_img, pid, camid, img_path

        img_path, pid, camid, clothid, img, contour_img = data_item
        img, contour_img = self.transforms(img, contour_img)

        return img, contour_img, pid, camid, img_path, clothid

    def __len__(self):
        return len(self.data)

class DataWarpper_Outmemory(object):
    def __init__(self, data, transforms):
        self.data = data
        self.transforms = transforms

        # self.mean1 = torch.tensor([0.485*255.0, 0.456*255.0, 0.406*255.0]).view(-1, 1, 1)
        # self.std1 = torch.tensor([0.229*255.0, 0.224*255.0, 0.225*255.0]).view(-1, 1, 1)
        # self.mean2 = torch.tensor([0.0]).view(-1, 1, 1)
        # self.std2 = torch.tensor([255.0]).view(-1, 1, 1)

    def __getitem__(self, idx):
        data_item = self.data[idx]

        if len(data_item) == 4:
            img_path, contour_path, pid, camid = data_item

            img = read_image(img_path, True)
            contour_img = read_image(contour_path)

            img, contour_img = self.transforms(img, contour_img)

            # img.sub_(self.mean1).div_(self.std1)
            # contour_img.sub_(self.mean2).div_(self.std2)

            return img, contour_img, pid, camid, img_path

        img_path, contour_path, pid, camid, clothid = data_item

        img = read_image(img_path, True)
        contour_img = read_image(contour_path)

        img, contour_img = self.transforms(img, contour_img)

        # img.sub_(self.mean1).div_(self.std1)
        # contour_img.sub_(self.mean2).div_(self.std2)

        return img, contour_img, pid, camid, img_path, clothid

    def __len__(self):
        return len(self.data)

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import random
import numpy as np

from .dataset import *
from lib.utils import read_image

class BIWI(ImageDataset):

    dataset_dir = 'BIWI'

    def __init__(self, root='', **kwargs):

        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.trainlist_path = osp.join(self.dataset_dir, 'list/train.txt')
        self.querylist_path = osp.join(self.dataset_dir, 'list/query1.txt') # query1.txt -> Still Set
        # self.querylist_path = osp.join(self.dataset_dir, 'list/query2.txt')  # query2.txt -> Walking Set
        self.gallerylist_path = osp.join(self.dataset_dir, 'list/gallery.txt')

        required_files = [
            self.dataset_dir,
            self.trainlist_path,
            self.gallerylist_path,
            self.querylist_path,
        ]
        self.check_before_run(required_files)

        # we assume every samples in BIWI is captured under different camera views
        self.cam_var = 0

        train = self.process_dir(self.dataset_dir, self.trainlist_path)
        query = self.process_dir(self.dataset_dir, self.querylist_path, if_test=True)
        gallery = self.process_dir(self.dataset_dir, self.gallerylist_path, if_test=True)

        super(BIWI, self).__init__(train, query, gallery)

    def process_dir(self, dir_path, file_path, if_test=False):
        datalist = [line for line in open(file_path, 'r').read().splitlines()]

        pid_container = set()
        for idx, item in enumerate(datalist):
            img_rel_path, pid = item.split()

            pid = int(pid)
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for idx, item in enumerate(datalist):
            img_rel_path, pid = item.split()

            img_path = osp.join(dir_path, img_rel_path)
            img = read_image(img_path, True)
            contour_path = img_path.replace('/rgb/', '/contour/')
            contour_img = read_image(contour_path)

            pid = int(pid)
            camid = self.cam_var
            self.cam_var += 1
            if not if_test: pid = pid2label[pid]

            # load data into memory
            data.append((img_path, pid, camid, img, contour_img))

        return data

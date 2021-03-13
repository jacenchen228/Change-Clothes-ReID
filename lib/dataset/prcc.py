from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import random
import numpy as np

from .dataset import *
from lib.utils import read_image

ROOM2CAMID = {
    'A': 0,
    'A_large_view': 0,
    'B': 1,
    'C': 2
}


class PRCC(ImageDataset):
    """PRCC.

    Dataset statistics:
        - identities: 221(train + query).
        - images: (train) + 2228 (query) + 17661 (gallery).
        - cameras: 3.
    """
    dataset_dir = 'prcc'

    def __init__(self, root='', aug=False, **kwargs):

        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.trainlist_path = osp.join(self.dataset_dir, 'list/train.txt')
        self.querylist_path = osp.join(self.dataset_dir, 'list/query1.txt')
        self.gallerylist_path = osp.join(self.dataset_dir, 'list/gallery.txt')

        required_files = [
            self.dataset_dir,
            self.trainlist_path,
            self.gallerylist_path,
            self.querylist_path,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.dataset_dir, self.trainlist_path)
        query = self.process_dir(self.dataset_dir, self.querylist_path, if_test=True)
        gallery = self.process_dir(self.dataset_dir, self.gallerylist_path, if_test=True)

        self.train_aug = self.process_dir(self.dataset_dir, self.trainlist_path, if_aug=True)

        super(PRCC, self).__init__(train, query, gallery)

    def process_dir(self, dir_path, file_path, if_test=False, if_aug=False):
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
            # img = read_image(img_path, True)
            contour_path = img_path.replace('/rgb/', '/contour/')
            # contour_img = read_image(contour_path)

            pid = int(pid)
            if if_test:
                camid = ROOM2CAMID[img_path.split('/')[-3]]
            else:
                cam = img_path.split('/')[-1].split('_')[0]
                camid = ROOM2CAMID[cam]
            assert 0 <= camid <= 2
            if not if_test: pid = pid2label[pid]

            # load data into memory
            # data.append((img_path, pid, camid, img, contour_img))

            data.append((img_path, contour_path, pid, camid))

            # # process augmented data item
            # if not if_test and if_aug:
            #     aug_img_dir = osp.dirname(img_rel_path)
            #     aug_base_name = osp.basename(img_rel_path)
            #     aug_img_name, postfix = osp.splitext(aug_base_name)
            #     for i in range(10):
            #         aug_name = aug_img_name + '_' + str(i) + postfix
            #         # aug_dir = osp.join(dir_path, 'aug3_hue0.1', img_dir)
            #         # aug_dir = osp.join(dir_path, 'aug2', img_dir)
            #         aug_dir = osp.join(dir_path, 'aug4_hue0.1sat0.1', aug_img_dir)
            #         aug_path = osp.join(aug_dir, aug_name)
            #         aug_img = read_image(aug_path, True)
            #         data.append((aug_path, pid, camid, aug_img, contour_img))

        # if not if_test:
        #     dataset_len = len(data)
        #     sample_factor = 0.2
        #     sample_num = int(sample_factor * dataset_len)
        #
        #     random.shuffle(data)
        #     data = data[:sample_num]

        return data

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import random
import numpy as np

from .dataset import *
from lib.utils import read_image

ROOM2CAMID = {
    'A': 0,
    'B': 1,
    'C': 2
}


class VC_Clothes(ImageDataset):
    """PRCC.

    Dataset statistics:
        - identities: 221(train + query).
        - images: (train) + 2228 (query) + 17661 (gallery).
        - cameras: 3.
    """
    dataset_dir = 'VC-Clothes'

    def __init__(self, root='', **kwargs):

        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.trainlist_path = osp.join(self.dataset_dir, 'list/train.txt')
        self.querylist_path = osp.join(self.dataset_dir, 'list/query.txt')
        self.gallerylist_path = osp.join(self.dataset_dir, 'list/gallery.txt')

        # load pre estimated 2D joints
        self.train_joint_path = osp.join(self.dataset_dir, 'pose', 'train.npz')
        self.query_joint_path = osp.join(self.dataset_dir, 'pose', 'query.npz')
        self.gallery_joint_path = osp.join(self.dataset_dir, 'pose', 'gallery.npz')

        required_files = [
            self.dataset_dir,
            self.trainlist_path,
            self.gallerylist_path,
            self.querylist_path,
            self.train_joint_path,
            self.query_joint_path,
            self.gallery_joint_path
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.dataset_dir, self.trainlist_path, self.train_joint_path)
        query = self.process_dir(self.dataset_dir, self.querylist_path, self.query_joint_path, if_test=True, opt='query')
        gallery = self.process_dir(self.dataset_dir, self.gallerylist_path, self.gallery_joint_path, if_test=True, opt='gallery')

        super(VC_Clothes, self).__init__(train, query, gallery)

    def process_dir(self, dir_path, file_path, joint_path, if_test=False, opt='train'):
        datalist = [line for line in open(file_path, 'r').read().splitlines()]

        # load joint array
        data_len = len(datalist)
        if not if_test:
            joints_arr = np.load(joint_path)['est_joints']
            joints_arr = joints_arr.transpose(2, 1, 0)
        else:
            joints_arr = np.zeros((data_len, 17, 3), np.float32)

        pid_container = set()
        for idx, item in enumerate(datalist):
            _, pid, _ = item.split()

            pid = int(pid)
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for idx, item in enumerate(datalist):
            img_rel_path, pid, camid = item.split()

            joint = joints_arr[idx]

            # read rgb contour segment smpl_param
            img_path = osp.join(dir_path, img_rel_path)
            img = read_image(img_path, True)
            if not if_test:
                mask_path = img_path.replace('/train/', '/train_mask/').replace('.jpg', '.png')
            else:
                mask_path = None

            pid, camid = int(pid), int(camid)
            if not if_test: pid = pid2label[pid]

            if opt == 'gallery':
                if camid == 2:
                    data.append((img_path, pid, camid, mask_path, joint, img))
            elif opt == 'query':
                if camid == 3:
                    data.append((img_path, pid, camid, mask_path, joint, img))
            elif opt == 'train':
                data.append((img_path, pid, camid, mask_path, joint, img))
            # data.append((img_path,  pid, camid, mask_path, smpl_path, joint, segment_path, vertice_path))

        return data

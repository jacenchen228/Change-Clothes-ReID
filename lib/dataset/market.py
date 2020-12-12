from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

from .dataset import *
from lib.utils import read_image

class Market(ImageDataset):
    """Market.

    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    dataset_dir = 'market1501/Market-1501-v15.09.15'

    def __init__(self, root='', **kwargs):

        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.trainlist_path = osp.join(self.dataset_dir, 'list_new/train_has_pose.txt')
        self.querylist_path = osp.join(self.dataset_dir, 'list/query.txt')
        self.gallerylist_path = osp.join(self.dataset_dir, 'list/gallery.txt')

        # load pre estimated 2D joints
        # self.joints_dir = '/home/jiaxing/AlphaPose'
        # self.train_joint_path = osp.join(self.joints_dir, 'market_train', 'est_joints.npz')
        self.train_joint_path = osp.join(self.dataset_dir, 'joints', 'train.npz')
        # self.query_joint_path = osp.join(self.joints_dir, 'market_query', 'est_joints.npz')
        # self.gallery_joint_path = osp.join(self.joints_dir, 'market_gallery', 'est_joints.npz')

        required_files = [
            self.dataset_dir,
            self.trainlist_path,
            self.gallerylist_path,
            self.querylist_path,
            self.train_joint_path
        ]
        self.check_before_run(required_files)
        self.at_least_num = 4

        train = self.process_dir(self.dataset_dir, self.trainlist_path, self.train_joint_path)
        query = self.process_dir(self.dataset_dir, self.querylist_path, None, if_test=True)
        gallery = self.process_dir(self.dataset_dir, self.gallerylist_path, None, if_test=True)

        super(Market, self).__init__(train, query, gallery)

    def process_dir(self, dir_path, file_path, joint_path=None, if_test=False):
        datalist = [line for line in open(file_path, 'r').read().splitlines()]

        # load joint array
        data_len = len(datalist)
        if joint_path is not None:
            joints_arr = np.load(joint_path)['est_joints']
            joints_arr = joints_arr.transpose(2, 1, 0)
        else:
            joints_arr = np.zeros((data_len, 17, 3), np.float32)

        pid_sample_cnts = dict()
        for idx, item in enumerate(datalist):
            img_rel_path, pid, _ = item.split()
            pid = int(pid)

            if pid not in pid_sample_cnts:
                pid_sample_cnts[pid] = 1
            else:
                pid_sample_cnts[pid] += 1

        pid_container = set()
        for pid, cnt in pid_sample_cnts.items():
            if cnt >= self.at_least_num:
                pid_container.add(pid)

        pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for idx, item in enumerate(datalist):
            img_rel_path, pid, camid = item.split()
            pid, camid = int(pid), int(camid)

            joint = joints_arr[idx]

            # read rgb contour segment smpl_param
            img_path = osp.join(dir_path, img_rel_path)
            img = read_image(img_path, True)

            # if 'bounding_box_train' in img_path:
            #     gray_path = img_path.replace('/bounding_box_train/', '/gray_train/')
            # if 'bounding_box_test' in img_path:
            #     gray_path = img_path.replace('/bounding_box_test/', '/gray_gallery/')
            # if 'query' in img_path:
            #     gray_path = img_path.replace('/query/', '/gray_query/')

            if not if_test:
                mask_path = img_path.replace('/bounding_box_train/', '/mask_trainHasPose/').replace('.jpg', '.png')
                segment_path = img_path.replace('/rgb/', '/segment/').replace('.jpg', '.png')
            else:
                mask_path, segment_path = None, None

            if not if_test:
                if pid in pid2label:
                    pid = pid2label[pid]
                    data.append((img_path, pid, camid, mask_path, joint, img))
            else:
                data.append((img_path, pid, camid, mask_path, joint, img))

        return data

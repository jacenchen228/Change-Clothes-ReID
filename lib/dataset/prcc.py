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


class PRCC3D(ImageDataset):
    """PRCC.

    Dataset statistics:
        - identities: 221(train + query).
        - images: (train) + 2228 (query) + 17661 (gallery).
        - cameras: 3.
    """
    dataset_dir = 'prcc'

    def __init__(self, root='', **kwargs):

        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.trainlist_path = osp.join(self.dataset_dir, 'list_new/train.txt')
        self.querylist_path = osp.join(self.dataset_dir, 'list_new/query1.txt')
        self.gallerylist_path = osp.join(self.dataset_dir, 'list_new/gallery.txt')
        self.smpl_dir = '/home/jiaxing/mpips-smplify_public_v2/smplify_public'

        # load pre estimated 2D joints
        self.train_joint_path = osp.join(self.dataset_dir, 'pose', 'train.npz')
        self.query_joint_path = osp.join(self.dataset_dir, 'pose', 'query1.npz')
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
        query = self.process_dir(self.dataset_dir, self.querylist_path, self.query_joint_path, if_test=True)
        gallery = self.process_dir(self.dataset_dir, self.gallerylist_path, self.gallery_joint_path, if_test=True)

        super(PRCC3D, self).__init__(train, query, gallery)

    def process_dir(self, dir_path, file_path, joint_path, if_test=False):
        datalist = [line for line in open(file_path, 'r').read().splitlines()]

        # load joint array
        joints_arr = np.load(joint_path)['est_joints']
        joints_arr = joints_arr.transpose(2, 1, 0)

        pid_container = set()
        for idx, item in enumerate(datalist):
            img_rel_path, pid = item.split()

            img_path = osp.join(dir_path, img_rel_path)
            if '/train/' in img_path:
                smpl_sub_dir = 'prcc_train'
            elif '/test/A/' in img_path:
                smpl_sub_dir = 'prcc_gallery'
            elif '/test/C/' in img_path:
                smpl_sub_dir = 'prcc_query1'
            smpl_path = osp.join(self.smpl_dir, smpl_sub_dir, '%d.pkl'%(idx))

            # if not osp.exists(smpl_path):
            #     smpl_path = osp.join(self.smpl_dir, smpl_sub_dir, '%04d.pkl' % (idx))
            #     if not osp.exists(smpl_path):
            #         continue

            pid = int(pid)
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for idx, item in enumerate(datalist):
            img_rel_path, pid = item.split()

            joint = joints_arr[idx]

            # read rgb contour segment smpl_param
            img_path = osp.join(dir_path, img_rel_path)
            img = read_image(img_path, True)
            mask_path = img_path.replace('/rgb/', '/mask1/').replace('.jpg', '.png')
            segment_path = img_path.replace('/rgb/', '/segment/').replace('.jpg', '.png')

            # segment_path = img_path.replace('/rgb/', '/segment/').replace('.jpg', '.npy')
            if '/train/' in img_path:
                smpl_sub_dir = 'prcc_train'
            elif '/test/A/' in img_path:
                smpl_sub_dir = 'prcc_gallery'
            elif '/test/C/' in img_path:
                smpl_sub_dir = 'prcc_query1'
            smpl_path = osp.join(self.smpl_dir, smpl_sub_dir, '%d.pkl'%(idx))

            # if not osp.exists(smpl_path):
            #     smpl_path = osp.join(self.smpl_dir, smpl_sub_dir, '%04d.pkl' % (idx))
            #     if not osp.exists(smpl_path):
            #         continue

            # # depth + depth_maskWW
            # img_path = osp.join(dir_path, img_rel_path).replace('/rgb/', '/depth/').replace('.jpg', '.png')
            # contour_path = img_path.replace('/depth/', '/depth_mask/')

            pid = int(pid)
            if if_test:
                camid = ROOM2CAMID[img_path.split('/')[-3]]
            else:
                cam = img_path.split('/')[-1].split('_')[0]
                camid = ROOM2CAMID[cam]
            assert 0 <= camid <= 2
            if not if_test: pid = pid2label[pid]

            # data.append((img_path,  pid, camid, mask_path, smpl_path, joint, segment_path, vertice_path))
            data.append((img_path, pid, camid, mask_path, joint, img))

        return data

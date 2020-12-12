from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

from lib.dataset import *

ROOM2CAMID = {
    'A': 0,
    'B': 1,
    'C': 2
}


class PRCC_3D_RELABEL(ImageDataset):
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
        self.trainlist_path = osp.join(self.dataset_dir, 'list/train_has_pose.txt')
        self.querylist_path = osp.join(self.dataset_dir, 'list/query1.txt')
        self.gallerylist_path = osp.join(self.dataset_dir, 'list/gallery.txt')
        self.smpl_dir = '/home/jiaxing/mpips-smplify_public_v2/smplify_public'

        # load pre estimated 2D joints
        self.joints_dir = '/home/jiaxing/mpips-smplify_public_v2/smplify_public/results'
        self.train_joint_path = osp.join(self.joints_dir, 'prcc', 'est_joints.npz')
        self.query_joint_path = osp.join(self.joints_dir, 'prcc_query1', 'est_joints.npz')
        self.gallery_joint_path = osp.join(self.joints_dir, 'prcc_gallery', 'est_joints.npz')

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

        super(PRCC_3D_RELABEL, self).__init__(train, query, gallery)

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

            if not osp.exists(smpl_path):
                smpl_path = osp.join(self.smpl_dir, smpl_sub_dir, '%04d.pkl' % (idx))
                if not osp.exists(smpl_path):
                    continue

            pid = int(pid)
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        pid_total_num = len(pid2label)

        data = []
        for idx, item in enumerate(datalist):
            img_rel_path, pid = item.split()

            joint = joints_arr[idx]

            # read rgb contour segment smpl_param
            img_path = osp.join(dir_path, img_rel_path)
            contour_path = img_path.replace('/rgb/', '/sketch/')
            segment_path = img_path.replace('/rgb/', '/segment/').replace('.jpg', '.png')
            # segment_path = img_path.replace('/rgb/', '/segment/').replace('.jpg', '.npy')
            if '/train/' in img_path:
                smpl_sub_dir = 'prcc_train'
            elif '/test/A/' in img_path:
                smpl_sub_dir = 'prcc_gallery'
            elif '/test/C/' in img_path:
                smpl_sub_dir = 'prcc_query1'
            smpl_path = osp.join(self.smpl_dir, smpl_sub_dir, '%d.pkl'%(idx))

            if not osp.exists(smpl_path):
                smpl_path = osp.join(self.smpl_dir, smpl_sub_dir, '%04d.pkl' % (idx))
                if not osp.exists(smpl_path):
                    continue

            # # depth + depth_mask
            # img_path = osp.join(dir_path, img_rel_path).replace('/rgb/', '/depth/').replace('.jpg', '.png')
            # contour_path = img_path.replace('/depth/', '/depth_mask/')

            pid = int(pid)
            if if_test:
                camid = ROOM2CAMID[img_path.split('/')[-3]]
            else:
                cam = img_path.split('/')[-1].split('_')[0]
                camid = ROOM2CAMID[cam]
            assert 0 <= camid <= 2

            pid_relabel = 0
            if not if_test:
                pid = pid2label[pid]
                if 'C' in img_rel_path:
                    pid_relabel = pid + pid_total_num   # transform pid of samples with different clothes
                else:
                    pid_relabel = pid

            data.append((img_path, pid, camid, contour_path, smpl_path, joint, segment_path, pid_relabel))

        return data

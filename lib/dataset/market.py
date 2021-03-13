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
        self.trainlist_path = osp.join(self.dataset_dir, 'list/train_has_pose.txt')
        self.querylist_path = osp.join(self.dataset_dir, 'list/query.txt')
        self.gallerylist_path = osp.join(self.dataset_dir, 'list/gallery.txt')

        required_files = [
            self.dataset_dir,
            self.trainlist_path,
            self.gallerylist_path,
            self.querylist_path,
        ]
        self.check_before_run(required_files)
        # self.at_least_num = 4

        train = self.process_dir(self.dataset_dir, self.trainlist_path)
        query = self.process_dir(self.dataset_dir, self.querylist_path, if_test=True)
        gallery = self.process_dir(self.dataset_dir, self.gallerylist_path, if_test=True)

        super(Market, self).__init__(train, query, gallery)

    def process_dir(self, dir_path, file_path, if_test=False):
        datalist = [line for line in open(file_path, 'r').read().splitlines()]

        # Setting the minimum sample number for an identity
        # pid_sample_cnts = dict()
        # for idx, item in enumerate(datalist):
        #     img_rel_path, pid, _ = item.split()
        #     pid = int(pid)
        #
        #     if pid not in pid_sample_cnts:
        #         pid_sample_cnts[pid] = 1
        #     else:
        #         pid_sample_cnts[pid] += 1
        # pid_container = set()
        # for pid, cnt in pid_sample_cnts.items():
        #     # if cnt >= self.at_least_num:
        #     #     pid_container.add(pid)
        #     pid_container.add(pid)
        # pid2label = {pid: label for label, pid in enumerate(pid_container)}

        pid_container = set()
        for idx, item in enumerate(datalist):
            img_rel_path, pid, _ = item.split()
            pid = int(pid)

            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for idx, item in enumerate(datalist):
            img_rel_path, pid, camid = item.split()
            pid, camid = int(pid), int(camid)

            img_path = osp.join(dir_path, img_rel_path)
            # img = read_image(img_path, True)
            contour_path = img_path.replace('/rgb/', '/contour/')
            # contour_img = read_image(contour_path)

            if not if_test:
                if pid in pid2label:
                    pid = pid2label[pid]
                    # data.append((img_path, pid, camid, img, contour_img))
                    data.append((img_path, contour_path, pid, camid))
            else:
                data.append((img_path, contour_path, pid, camid))

        return data

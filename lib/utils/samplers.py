from __future__ import absolute_import
from __future__ import division

from collections import defaultdict
import numpy as np
import copy
import random

from torch.utils.data.sampler import Sampler, RandomSampler


class RandomIdentitySampler(Sampler):
    """Randomly samples N identities each with K instances.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid).
        batch_size (int): batch size.
        num_instances (int): number of instances per identity in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        if batch_size < num_instances:
            raise ValueError('batch_size={} must be no less '
                             'than num_instances={}'.format(batch_size, num_instances))

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, item in enumerate(self.data_source):
            pid = item[1]
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        # TODO: improve precision
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomIdentitySamplerRelabel(Sampler):
    """Randomly samples N identities each with K instances.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid).
        batch_size (int): batch size.
        num_instances (int): number of instances per identity in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances, num_train_pids):
        if batch_size < num_instances:
            raise ValueError('batch_size={} must be no less '
                             'than num_instances={}'.format(batch_size, num_instances))

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.pids = set()
        self.index_dict_relabel = defaultdict(list)
        for index, item in enumerate(self.data_source):
            self.index_dict_relabel[item[7]].append(index)    # item[7] = pid_relabel
        self.pids_relabel = list(self.index_dict_relabel.keys())
        self.pids_relabel.sort()
        self.pids_num_ori = num_train_pids

        # estimate number of examples in an epoch
        # TODO: improve precision
        self.length = 0
        for pid in self.pids_relabel:
            idxs = self.index_dict_relabel[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids_relabel:
            idxs = copy.deepcopy(self.index_dict_relabel[pid])

            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)

            # samples with the different clothes which have just one view
            # so here augment the data into the same number
            if pid >= self.pids_num_ori:
                residual = len(self.index_dict_relabel[pid - self.pids_num_ori]) - len(self.index_dict_relabel[pid])
            else:
                residual = len(self.index_dict_relabel[pid + self.pids_num_ori]) - len(self.index_dict_relabel[pid])

            while residual > 0:
                if residual > len(self.index_dict_relabel[pid]):
                    idxs.extend(self.index_dict_relabel[pid])
                    residual -= len(self.index_dict_relabel[pid])
                else:
                    tmp_sample = random.sample(self.index_dict_relabel[pid], residual)
                    idxs.extend(tmp_sample)
                    residual = 0

            random.shuffle(idxs)

            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids_relabel)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch//2)
            for pid in selected_pids:
                if pid in avai_pids:
                    batch_idxs = batch_idxs_dict[pid].pop(0)

                    if len(batch_idxs_dict[pid]) == 0:
                        avai_pids.remove(pid)

                    final_idxs.extend(batch_idxs)

                # samples of the same identity with different clothes
                if pid >= self.pids_num_ori:
                    pid_ide = pid - self.pids_num_ori
                else:
                    pid_ide = pid + self.pids_num_ori
                if pid_ide in avai_pids:
                    batch_idxs_ide = batch_idxs_dict[pid_ide].pop(0)

                    if len(batch_idxs_dict[pid_ide]) == 0:
                        avai_pids.remove(pid_ide)

                    final_idxs.extend(batch_idxs_ide)

        return iter(final_idxs)

    def __len__(self):
        return self.length


def build_train_sampler(data_source, train_sampler, batch_size=32, num_instances=4, **kwargs):
    """Builds a training sampler.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid).
        train_sampler (str): sampler name (default: ``RandomSampler``).
        batch_size (int, optional): batch size. Default is 32.
        num_instances (int, optional): number of instances per identity in a
            batch (for ``RandomIdentitySampler``). Default is 4.
    """
    if train_sampler == 'RandomIdentitySampler':
        sampler = RandomIdentitySampler(data_source, batch_size, num_instances)

    else:
        sampler = RandomSampler(data_source)

    return sampler

def build_train_sampler(data_source, train_sampler, num_train_pids, batch_size=32, num_instances=4, **kwargs):
    """Builds a training sampler.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid).
        train_sampler (str): sampler name (default: ``RandomSampler``).
        batch_size (int, optional): batch size. Default is 32.
        num_instances (int, optional): number of instances per identity in a
            batch (for ``RandomIdentitySampler``). Default is 4.
    """
    if train_sampler == 'RandomIdentitySampler':
        sampler = RandomIdentitySampler(data_source, batch_size, num_instances)
    elif train_sampler == 'RandomIdentitySamplerRelabel':
        sampler = RandomIdentitySamplerRelabel(data_source, batch_size, num_instances, num_train_pids)
    else:
        sampler = RandomSampler(data_source)

    return sampler

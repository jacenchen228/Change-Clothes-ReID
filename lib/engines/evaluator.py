import numpy as np
import time
from collections import defaultdict
import os.path as osp

import torch
from torch.autograd import Variable
from torch.nn import functional as F

from lib.utils import AverageMeter
from lib.utils import re_ranking, visualize_ranked_results
from lib.metrics import compute_distance_matrix, evaluate_rank
from lib.losses import normalize

ID2FEAT_NAME = {
    0: 'fuse'
}


class Evaluator(object):
    def __init__(self, queryloader, galleryloader,  model, query=None, gallery=None, use_gpu=True,
                 dist_metric='euclidean', normalize_feature=False, visrank=False,
                 visrank_topk=10, save_dir='', use_metric_cuhk03=False, ranks=[1, 5, 10, 20],
                 rerank=False, height=256, width=128, **kwargs):
        self.queryloader = queryloader
        self.galleryloader = galleryloader
        self.model = model

        self.use_gpu = use_gpu
        self.dist_metric = dist_metric
        self.normalize_feature = normalize_feature
        self.visrank = visrank
        self.visrank_topk = visrank_topk
        self.save_dir = save_dir
        self.use_metric_cuhk03 = use_metric_cuhk03
        self.ranks = ranks
        self.rerank = rerank
        self.test_time = 10 # duplicate evaluate times for protocal of prcc

        # for ranklist visualization
        self.query = query
        self.gallery = gallery
        self.height = height
        self.width = width

        # find sample ids belonging to specific class ids for prcc protocal
        gallery_ids = [item[1] for item in gallery] # item[1] = pid
        self.pid2id = dict()
        pid_unique = set(gallery_ids)
        gallery_ids = np.array(gallery_ids)
        for pid in pid_unique:
            self.pid2id[pid] = np.argwhere(gallery_ids == pid)

    @torch.no_grad()
    def evaluate(self):
        batch_time = AverageMeter()

        self.model.eval()

        print('Extracting features from query set ...')
        q_pids, q_camids = [], []  # query person IDs and query camera IDs
        qf_dict = defaultdict(list)
        for batch_idx, data in enumerate(self.queryloader):
            imgs, contours, pids, camids = self._parse_data(data)

            if self.use_gpu:
                imgs = imgs.cuda()
                contorus = contours.cuda()

            end = time.time()

            features_list = self._extract_features(imgs, contorus)
            # features_list_ = self._extract_features(imgs.flip(3), contours.flip(3))

            batch_time.update(time.time() - end)

            for i in range(len(features_list)):
                features = features_list[i]
                features = features.data.cpu()
                # features_ = features_list_[i]
                # features_ = features_.data.cpu()
                #
                # features_all = (features + features_) / 2
                # features_all = normalize(features_all)

                qf_dict[i].append(features)

            q_pids.extend(pids)
            q_camids.extend(camids)

        qf_list = []
        for i in range(len(qf_dict.keys())):
            qf = torch.cat(qf_dict[i], 0)
            qf_list.append(qf)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        # print('Done, obtained {}-by-{}-by-{} matrix'.format(qf.size(0), qf.size(1), qf.size(2)))
        # print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        print('Extracting features from gallery set ...')
        g_pids, g_camids = [], []  # gallery person IDs and gallery camera IDs
        gf_dict = defaultdict(list)
        end = time.time()
        for batch_idx, data in enumerate(self.galleryloader):
            imgs, contours, pids, camids = self._parse_data(data)
            if self.use_gpu:
                imgs = imgs.cuda()
                contours = contours.cuda()

            end = time.time()
            features_list = self._extract_features(imgs, contours)
            # features_list_ = self._extract_features(imgs.flip(3), contours.flip(3))

            batch_time.update(time.time() - end)

            for i in range(len(features_list)):
                features = features_list[i]
                features = features.data.cpu()
                # features_ = features_list_[i]
                # features_ = features_.data.cpu()
                #
                # features_all = (features + features_) / 2
                # features_all = normalize(features_all)

                gf_dict[i].append(features)

            g_pids.extend(pids)
            g_camids.extend(camids)

        gf_list = []
        for i in range(len(gf_dict.keys())):
            gf = torch.cat(gf_dict[i], 0)
            gf_list.append(gf)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        # print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

        print('Speed: {:.4f} sec/batch'.format(batch_time.avg))

        # calculate distmats with different features
        distmats_list = list()
        for i, (qf, gf) in enumerate(zip(qf_list, gf_list)):
            if self.normalize_feature:
                print('Normalzing features with L2 norm ...')
                qf = F.normalize(qf, p=2, dim=1)
                gf = F.normalize(gf, p=2, dim=1)
            print('Computing distance matrix with metric={} feat={} ...'.format(self.dist_metric, ID2FEAT_NAME[i]))
            distmat = compute_distance_matrix(qf, gf)
            distmat = distmat.numpy()

            if self.rerank:
                print('Applying person re-ranking ...')
                distmat_qq = compute_distance_matrix(qf, qf, self.dist_metric)
                distmat_gg = compute_distance_matrix(gf, gf, self.dist_metric)
                distmat = re_ranking(distmat, distmat_qq, distmat_gg)

            distmats_list.append(distmat)

        cmcs_dict, mAPs_dict = defaultdict(list), defaultdict(list)
        query, gallery = np.array(self.query), np.array(self.gallery)
        for test_i in range(self.test_time):
            selected_ids = []
            for k in self.pid2id:
                selected_ids.append(np.random.choice(self.pid2id[k].reshape(-1)))

            for i, distmat in enumerate(distmats_list):
                distmat_selected = distmat[:, selected_ids]
                g_pids_selected = g_pids[selected_ids]
                g_camids_selected = g_camids[selected_ids]

                if self.visrank and (ID2FEAT_NAME[i] == 'feat 0' or ID2FEAT_NAME[i] == 'feat 2' or ID2FEAT_NAME[i] == 'feat 4') and test_i == 0:
                    # contains tuples of (img_path(s), contour_path(s), pid, camid)
                    gallery_selected = gallery[selected_ids]
                    visualize_ranked_results(
                        distmat_selected,
                        (query, gallery_selected),
                        'image',
                        width=self.width,
                        height=self.height,
                        save_dir=osp.join(self.save_dir, 'visrank_' + ID2FEAT_NAME[i] + '_' + str(test_i)),
                        topk=self.visrank_topk
                    )

                cmc, mAP = evaluate_rank(
                    distmat_selected,
                    q_pids,
                    g_pids_selected,
                    q_camids,
                    g_camids_selected,
                    use_metric_cuhk03=self.use_metric_cuhk03
                )

                cmcs_dict[i].append(cmc)
                mAPs_dict[i].append(mAP)

        rank1s = list()
        for i in range(len(distmats_list)):
            cmcs = cmcs_dict[i]
            mAPs = mAPs_dict[i]

            cmc_mean = np.mean(cmcs, 0)
            mAP_mean = np.mean(mAPs)

            print('Computing CMC and mAP with feat={} ...'.format(ID2FEAT_NAME[i]))
            print('** Results **')
            print('mAP: {:.1%}'.format(mAP_mean))
            print('CMC curve')
            for r in self.ranks:
                print('Rank-{:<3}: {:.1%}'.format(r, cmc_mean[r - 1]))

            rank1s.append(cmc_mean[0])

        return rank1s

    def _parse_data(self, data):
        imgs = data[0]
        contours = data[1]
        pids = data[2]
        camids = data[3]

        return imgs, contours, pids, camids

    def _extract_features(self, input, input_ext):
        self.model.eval()
        return self.model(input, input_ext)

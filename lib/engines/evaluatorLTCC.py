import numpy as np
import time
from collections import defaultdict
import os.path as osp

import torch
from torch.nn import functional as F

from lib.utils import AverageMeter
from lib.utils import re_ranking, visualize_ranked_results
from lib.metrics import compute_distance_matrix, evaluate_rank_ltcc

ID2FEAT_NAME = {
    0: 'global feat',
    1: 'part feat',
    2: 'concate feat',
    3: 'concate feat without normalizing global and part',
    4: 'concate contour feat',
    5: 'feat 5',
    6: 'feat 6'
}

class EvaluatorLTCC(object):
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

    @torch.no_grad()
    def evaluate(self):
        batch_time = AverageMeter()

        self.model.eval()

        print('Extracting features from query set ...')
        q_pids, q_camids, q_clothids = [], [], []  # query person IDs, query camera IDs and cloth IDs
        qf_dict = defaultdict(list)
        for batch_idx, data in enumerate(self.queryloader):
            imgs, contours, pids, camids, clothids = self._parse_data(data)

            if self.use_gpu:
                imgs = imgs.cuda()
                contours = contours.cuda()

            end = time.time()

            features_list = self._extract_features(imgs, contours)
            features_list_flip = self._extract_features(imgs.flip(3), contours.flip(3))

            batch_time.update(time.time() - end)

            for i in range(len(features_list)):
                features_ori = features_list[i]
                features_ori = F.normalize(features_ori, p=2, dim=1)
                features_ori = features_ori.data.cpu()

                features_flip = features_list_flip[i]
                features_flip = F.normalize(features_flip, p=2, dim=1)
                features_flip = features_flip.data.cpu()

                features = (features_ori + features_flip) / 2
                features = F.normalize(features, p=2, dim=1)

                qf_dict[i].append(features)
                # qf_dict[i].append(features_ori)

            q_pids.extend(pids)
            q_camids.extend(camids)
            q_clothids.extend(clothids)

        qf_list = []
        for i in range(len(qf_dict.keys())):
            qf = torch.cat(qf_dict[i], 0)
            qf_list.append(qf)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        # print('Done, obtained {}-by-{}-by-{} matrix'.format(qf.size(0), qf.size(1), qf.size(2)))
        print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        print('Extracting features from gallery set ...')
        g_pids, g_camids, g_clothids = [], [], []  # gallery person IDs, gallery camera IDs and gallery cloth IDs
        gf_dict = defaultdict(list)
        end = time.time()
        for batch_idx, data in enumerate(self.galleryloader):
            imgs, contours, pids, camids, clothids = self._parse_data(data)
            if self.use_gpu:
                imgs = imgs.cuda()
                contours = contours.cuda()

            end = time.time()

            features_list = self._extract_features(imgs, contours)
            features_list_flip = self._extract_features(imgs.flip(3), contours.flip(3))

            batch_time.update(time.time() - end)

            for i in range(len(features_list)):
                features_ori = features_list[i]
                features_ori = F.normalize(features_ori, p=2, dim=1)
                features_ori = features_ori.data.cpu()

                features_flip = features_list_flip[i]
                features_flip = F.normalize(features_flip, p=2, dim=1)
                features_flip = features_flip.data.cpu()

                features = (features_ori + features_flip) / 2
                features = F.normalize(features, p=2, dim=1)

                gf_dict[i].append(features)
                # gf_dict[i].append(features_ori)

            g_pids.extend(pids)
            g_camids.extend(camids)
            g_clothids.extend(clothids)

        gf_list = []
        for i in range(len(gf_dict.keys())):
            gf = torch.cat(gf_dict[i], 0)
            gf_list.append(gf)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

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

        for i, distmat in enumerate(distmats_list):

            if self.visrank and ID2FEAT_NAME[i] == 'feat 5':
                # contains tuples of (img_path(s), contour_path(s), pid, camid)]
                visualize_ranked_results(
                    distmat,
                    (query, gallery),
                    'image',
                    width=self.width,
                    height=self.height,
                    save_dir=osp.join(self.save_dir, 'visrank_' + ID2FEAT_NAME[i]),
                    topk=self.visrank_topk
                )

            cmc, mAP = evaluate_rank_ltcc(
                distmat,
                q_pids,
                g_pids,
                q_camids,
                g_camids,
                q_clothids,
                g_clothids
            )

            cmcs_dict[i].append(cmc)
            mAPs_dict[i].append(mAP)

        rank1s = list()
        mAPs = list()
        for i in range(len(distmats_list)):
            cmcs_i = cmcs_dict[i]
            mAPs_i = mAPs_dict[i]

            cmc_mean = np.mean(cmcs_i, 0)
            mAP_mean = np.mean(mAPs_i)

            print('Computing CMC and mAP with feat={} ...'.format(ID2FEAT_NAME[i]))
            print('** Results **')
            print('mAP: {:.1%}'.format(mAP_mean))
            print('CMC curve')
            for r in self.ranks:
                print('Rank-{:<3}: {:.1%}'.format(r, cmc_mean[r - 1]))

            rank1s.append(cmc_mean[0])
            mAPs.append(mAP_mean)

        return rank1s, mAPs

    def _parse_data(self, data):
        imgs = data[0]
        contours = data[1]

        pids = data[2]
        camids = data[3]
        clothids = data[5]

        return imgs, contours, pids, camids, clothids

    def _extract_features(self, input1, input2):
        self.model.eval()
        # return self.model(segments, input)
        return self.model(input1, input2)

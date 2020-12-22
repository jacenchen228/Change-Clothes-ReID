from __future__ import absolute_import
from __future__ import print_function

__all__ = ['visualize_ranked_results', 'visactmap']

import numpy as np
import os
import os.path as osp
import shutil
import cv2
from PIL import Image

import torch
from torch.nn import functional as F

from .tools import mkdir_if_missing

GRID_SPACING = 10
QUERY_EXTRA_SPACING = 90
BW = 5 # border width
GREEN = (1, 215, 117)
RED = (111, 107, 241)
PAD_SPACING = 5


def visualize_ranked_results(distmat, dataset, data_type, width=128, height=256, save_dir='', topk=10):
    """Visualizes ranked results.

    Supports both image-reid and video-reid.

    For image-reid, ranks will be plotted in a single figure. For video-reid, ranks will be
    saved in folders each containing a tracklet.

    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        dataset (tuple): a 2-tuple containing (query, gallery), each of which contains
            tuples of (img_path(s), pid, camid).
        data_type (str): "image" or "video".
        width (int, optional): resized image width. Default is 128.
        height (int, optional): resized image height. Default is 256.
        save_dir (str): directory to save output images.
        topk (int, optional): denoting top-k images in the rank list to be visualized.
            Default is 10.
    """
    num_q, num_g = distmat.shape
    mkdir_if_missing(save_dir)

    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Visualizing top-{} ranks ...'.format(topk))
    
    query, gallery = dataset
    assert num_q == len(query)
    assert num_g == len(gallery)
    
    indices = np.argsort(distmat, axis=1)

    def _cp_img_to(src, dst, rank, prefix, matched=False):
        """
        Args:
            src: image path or tuple (for vidreid)
            dst: target directory
            rank: int, denoting ranked position, starting from 1
            prefix: string
            matched: bool
        """
        if isinstance(src, tuple) or isinstance(src, list):
            if prefix == 'gallery':
                suffix = 'TRUE' if matched else 'FALSE'
                dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3)) + '_' + suffix
            else:
                dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3))
            mkdir_if_missing(dst)
            for img_path in src:
                shutil.copy(img_path, dst)
        else:
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3) + '_name_' + osp.basename(src))
            shutil.copy(src, dst)

    for q_idx in range(num_q):
        item = query[q_idx]
        qimg_path, qpid, qcamid = item[:3]
        # qsegment_path = item[6]
        qpid, qcamid = int(qpid), int(qcamid)
        num_cols = topk + 1
        # grid_img = 255 * np.ones((2*height+GRID_SPACING, num_cols*width+(topk-1)*GRID_SPACING+QUERY_EXTRA_SPACING, 3), dtype=np.uint8)
        grid_img = 255 * np.ones((height, num_cols*width+(topk-1)*GRID_SPACING+QUERY_EXTRA_SPACING, 3), dtype=np.uint8)

        if data_type == 'image':
            qimg = cv2.imread(qimg_path)
            qimg = Image.fromarray(cv2.cvtColor(qimg, cv2.COLOR_BGR2RGB))
            qimg = cv2.cvtColor(np.asarray(qimg), cv2.COLOR_RGB2BGR)

            qimg = cv2.resize(qimg, (width, height))
            qimg = cv2.copyMakeBorder(qimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            qimg = cv2.resize(qimg, (
            width, height))  # resize twice to ensure that the border width is consistent across images

            # qsegment = cv2.imread(qsegment_path)
            # qsegment = Image.fromarray(cv2.cvtColor(qsegment, cv2.COLOR_BGR2RGB))
            # qsegment = cv2.cvtColor(np.asarray(qsegment), cv2.COLOR_RGB2BGR)
            #
            # qsegment = cv2.resize(qsegment, (width, height))
            # qsegment = cv2.copyMakeBorder(qsegment, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            # qsegment = cv2.resize(qsegment, (
            # width, height))  # resize twice to ensure that the border width is consistent across images

            grid_img[:height, :width, :] = qimg
            # grid_img[height+GRID_SPACING:, :width, :] = qsegment
        else:
            qdir = osp.join(save_dir, osp.basename(osp.splitext(qimg_path)[0]))
            mkdir_if_missing(qdir)
            _cp_img_to(qimg_path, qdir, rank=0, prefix='query')

        rank_idx = 1
        for g_idx in indices[q_idx, :]:
            item = gallery[g_idx]
            gimg_path, gpid, gcamid = item[:3]
            # gsegment_path = item[6]
            gpid, gcamid = int(gpid), int(gcamid)
            invalid = (qpid == gpid) & (qcamid == gcamid)

            if not invalid:
                matched = gpid == qpid
                if data_type == 'image':
                    border_color = GREEN if matched else RED
                    gimg = cv2.imread(gimg_path)
                    gimg = Image.fromarray(cv2.cvtColor(gimg, cv2.COLOR_BGR2RGB))
                    gimg = cv2.cvtColor(np.asarray(gimg), cv2.COLOR_RGB2BGR)

                    gimg = cv2.resize(gimg, (width, height))
                    gimg = cv2.copyMakeBorder(gimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=border_color)
                    gimg = cv2.resize(gimg, (width, height))

                    # gsegment = cv2.imread(gsegment_path)
                    # gsegment = Image.fromarray(cv2.cvtColor(gsegment, cv2.COLOR_BGR2RGB))
                    # gsegment = cv2.cvtColor(np.asarray(gsegment), cv2.COLOR_RGB2BGR)
                    #
                    # gsegment = cv2.resize(gsegment, (width, height))
                    # gsegment = cv2.copyMakeBorder(gsegment, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=border_color)
                    # gsegment = cv2.resize(gsegment, (width, height))

                    start = rank_idx * width + (rank_idx - 1) * GRID_SPACING + QUERY_EXTRA_SPACING
                    end = (rank_idx + 1) * width + (rank_idx - 1) * GRID_SPACING + QUERY_EXTRA_SPACING
                    grid_img[:height, start:end, :] = gimg
                    # grid_img[height+GRID_SPACING:, start:end, :] = gsegment
                else:
                    _cp_img_to(gimg_path, qdir, rank=rank_idx, prefix='gallery', matched=matched)

                rank_idx += 1
                if rank_idx > topk:
                    break
                # if rank_idx > topk-1:
                #     break

        relpath = qimg_path.split('/rgb/')[-1]
        imname = osp.basename(osp.splitext(relpath)[0])
        dirname = osp.dirname(relpath)

        dir_path = osp.join(save_dir, dirname)
        if not osp.exists(dir_path):
            os.makedirs(dir_path)
        cv2.imwrite(osp.join(dir_path, imname + '.jpg'), grid_img)

        # imname = osp.basename(osp.splitext(qimg_path)[0])
        # cv2.imwrite(osp.join(save_dir, imname+'.jpg'), grid_img)

        if (q_idx + 1) % 100 == 0:
            print('- done {}/{}'.format(q_idx + 1, num_q))

    print('Done. Images have been saved to "{}" ...'.format(save_dir))


@torch.no_grad()
def visactmap(testloader, model, save_dir, width, height, print_freq, use_gpu, **kwargs):
    """Visualizes CNN activation maps to see where the CNN focuses on to extract features.

    This function takes as input the query images of target datasets

    Reference:
  - Zagoruyko and Komodakis. Paying more attention to attention: Improving the
          performance of convolutional neural networks via attention transfer. ICLR, 2017
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
    """
    model.eval()

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # original images and activation maps are saved individually
    actmap_dir = osp.join(save_dir, 'actmap')
    mkdir_if_missing(actmap_dir)
    print('Visualizing activation maps ...')

    for batch_idx, data in enumerate(testloader):
        # imgs, paths = data[0], data[3]
        # imgs, paths = data[0], data[3]
        imgs, paths, segments = data[0], data[3], data[6]

        if use_gpu:
            imgs = imgs.cuda()
            segments = segments.cuda()

        # forward to get convolutional feature maps
        try:
            # outputs = model(segments, imgs, return_featuremaps=True)
            outputs = model(segments, imgs, return_featuremaps=True)
        except TypeError:
            raise TypeError('forward() got unexpected keyword argument "return_featuremaps". ' \
                            'Please add return_featuremaps as an input argument to forward(). When ' \
                            'return_featuremaps=True, return feature maps only.')

        if outputs.dim() != 4:
            raise ValueError('The model output is supposed to have ' \
                             'shape of (b, c, h, w), i.e. 4 dimensions, but got {} dimensions. '
                             'Please make sure you set the model output at eval mode '
                             'to be the last convolutional feature maps'.format(outputs.dim()))

        # compute activation maps
        outputs = (outputs ** 2).sum(1)
        b, h, w = outputs.size()
        outputs = outputs.view(b, h * w)
        outputs = F.normalize(outputs, p=2, dim=1)
        outputs = outputs.view(b, h, w)

        if use_gpu:
            imgs, outputs = imgs.cpu(), outputs.cpu()

        for j in range(outputs.size(0)):
            # get image name
            path = paths[j]

            # imname = osp.basename(osp.splitext(path)[0])
            path = path.split('/')
            imname = path[-2] + '_' + path[-1]

            # RGB image
            img = imgs[j, ...]
            for t, m, s in zip(img, imagenet_mean, imagenet_std):
                t.mul_(s).add_(m).clamp_(0, 1)
            img_np = np.uint8(np.floor(img.numpy() * 255))
            img_np = img_np.transpose((1, 2, 0))  # (c, h, w) -> (h, w, c)

            # activation map
            am = outputs[j, ...].numpy()
            am = cv2.resize(am, (width, height))
            am = 255 * (am - np.max(am)) / (np.max(am) - np.min(am) + 1e-12)
            am = np.uint8(np.floor(am))
            am = cv2.applyColorMap(am, cv2.COLORMAP_JET)

            # overlapped
            overlapped = img_np * 0.3 + am * 0.7
            overlapped[overlapped > 255] = 255
            overlapped = overlapped.astype(np.uint8)

            # save images in a single figure (add white spacing between images)
            # from left to right: original image, activation map, overlapped image
            grid_img = 255 * np.ones((height, 3 * width + 2 * GRID_SPACING, 3), dtype=np.uint8)
            grid_img[:, :width, :] = img_np[:, :, ::-1]
            grid_img[:, width + GRID_SPACING: 2 * width + GRID_SPACING, :] = am
            grid_img[:, 2 * width + 2 * GRID_SPACING:, :] = overlapped

            # cv2.imwrite(osp.join(actmap_dir, imname + '.jpg'), grid_img)
            cv2.imwrite(osp.join(actmap_dir, imname), grid_img)

        if (batch_idx + 1) % print_freq == 0:
            print('- done batch {}/{}'.format(batch_idx + 1, len(testloader)))


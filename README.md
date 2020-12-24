# Change-Clothes-ReID

This is the implementation of a novel person re-id model combining RGB and contour images to solve clothing-change person re-id. 

## Prerequisites

- python 3.7
- pytorch 1.2
- torchvision 0.4.0
- CUDA 10.0

## Data Preparation

We validate the performance of our model on 5 datasets including 2 large-scale datasets (Market1501, DukeMTMC-reID) and 3 clothing-confused datasets (PRCC, BIWI, VC-Clothes). Among clothing-confused datasets, PRCC, VC-Clothes and BIWI target at long-term person re-id in which the same pedestrian might change clothes.

1. For all datasets, we recommend to generate data list files train.txt, query.txt and gallery.txt for train, query and gallery sets in the format as follows:

        image_path1 person_id1 camera_id1
        image_path2 person_id2 camera_id2
        image_path3 person_id3 camera_id3
        ......

 For each line, different data items would be split by one space. All list files would be saved in the directory `list/`.

2. Pretrained models are utilized to extract human contours to combine with RGB images and learn a powerful representation for clothing-change person re-id.

- The keypoints estimator [RCF](https://github.com/yun-liu/rcf) is used to generate human contours. Specifically, we use the outputs where contours are marked as white (represented by value 1). The predicted results would be put in the directory `contour/` and the directory would be arranged the same as the original dataset.

## Train and Test

### Train

For training, different datasets and training hyper-parameters could be choosen in the command line. For example, the command line for training  the PRCC dataset could be set as the following example:

    python main.py -s prcc -t prcc --height 256 --width 128 --max-epoch 80 --batch-size 64 -a baseline --save-dir $SAVE_DIR --root $DATA_ROOT --gpu-devices $GPU_ID --transforms random_flip random_crop --dist-metric cosine 

### Test

For performance evaluation, the only hyper-parameter --evaluate should be added to the command line to change the mode. One example of corresponding command lines could be shown as follows:

    python main.py --evaluate -s prcc -t prcc --height 256 --width 128 --batch-size 64 -a baseline --save-dir $SAVE_DIR --root $DATA_ROOT --gpu-devices $GPU_ID --dist-metric cosine 

There are two different evaluation protocols used in our experiments. If you use the evaluation protocol as Market1501, the hyper-parameter --flag-general should be added to the evaluation command line. In default, we choose the evaluation protocol as the PRCC dataset. 

### Performance
#### PRCC
|Model| Rank-1 | Rank-5 |
|  :----:  |  :----:  | :----:  |
| Baseline  |  | |
| Our Model  |  | |
 

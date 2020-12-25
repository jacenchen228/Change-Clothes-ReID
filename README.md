# Change-Clothes-ReID

This is the implementation of a novel person re-id model combining RGB and contour images to solve clothing-change person re-id. 

## Prerequisites

- python 3.7
- pytorch 1.2
- torchvision 0.4.0
- CUDA 10.0
- apex

## Data Preparation

We validate the performance of our model on 3 clothing-confused datasets (`PRCC`, `BIWI`, `VC-Clothes`). Among clothing-confused datasets, `PRCC`, `VC-Clothes` and `BIWI` target at long-term person re-id in which the same pedestrian might change clothes.

1. For all datasets, we recommend to generate data list files train.txt, query.txt and gallery.txt for train, query and gallery sets in the format as follows:

        image_path1 person_id1
        image_path2 person_id2
        image_path3 person_id3
        ......

 For each line, different data items would be split by one space. All list files would be saved in the directory `$DATA_ROOT/list/`.

2. Pretrained models are utilized to extract human contours to combine with RGB images and learn a powerful representation for clothing-change person re-id.

- The contour extractor [RCF](https://github.com/yun-liu/rcf) is used to generate human contours. Specifically, we use the outputs where contours are marked as black (represented by value 0). The predicted results would be put in the directory `$DATA_ROOT/contour/` and the directory would be arranged the same as the original dataset.

3. The clothing-change datasets could be downloaded from [here](https://pan.baidu.com/s/1oUO0GM2nSPblJ69Xh2v-dg) (code:ix1g). The dataset directory should be decompressed to `$DATA_ROOT` and then you could specify it in the running commond as the following illustration. Taking the `PRCC` dataset as an example:

```Shell
        unzip prcc.zip
        mv prcc/ $DATA_ROOT/
```

## Train and Test

### Train

For training, different datasets and training hyper-parameters could be choosen in the command line. For example, the command line for training  the `PRCC` dataset could be set as the following example:

```Python
        python main.py -s prcc -t prcc -j 2 --height 256 --width 128 --max-epoch 80 --batch-size 64 -a baseline --save-dir $SAVE_DIR --root $DATA_ROOT --gpu-devices $GPU_ID --transforms random_flip random_crop --dist-metric cosine --lr $LR --optim $OPTIMIZER
```
        
### Test

For performance evaluation, the only hyper-parameter `--evaluate` should be added to the command line to change the mode. One example of corresponding command lines could be shown as follows:

```Python
        python main.py --evaluate -s $SOURCE_DATASET -t $TARGET_DATASET -j 2 --height 256 --width 128 --batch-size 64 -a $MODEL_NAME --save-dir $SAVE_DIR --root $DATA_ROOT --gpu-devices $GPU_ID --dist-metric cosine --load-weights $WEIGHT_PATH
```
        
The pretrained model weights could be downloaded from [here](https://pan.baidu.com/s/1WnrAxFFkX0ksquM7SJwSIA) (code:u9ir). You could specify the dataset name (check `lib/models/__init__.py`) and put the weight file in `$WEIGHT_PATH`. Then you could check the performances which are shown in the following.  

### Performance
#### PRCC
|Model| Rank-1 | Rank-5 |
|  :----:  |  :----:  | :----:  |
|[SPT+ASE](https://arxiv.org/abs/2002.02295)|34.4%|-|
| Baseline  | 35.8% | 58.9%|
| Our Model  | 46.1% | 65.9%|

#### BIWI Still
|Model| Rank-1 | Rank-5 |
|  :----:  |  :----:  | :----:  |
|[SPT+ASE](https://arxiv.org/abs/2002.02295)|21.3%|66.1%|
| Baseline  | 17.1% | 58.0%|
| Our Model  | 31.5% | 75.2%|

#### BIWI Walking
|Model| Rank-1 | Rank-5 |
|  :----:  |  :----:  | :----:  |
|[SPT+ASE](https://arxiv.org/abs/2002.02295)|18.7%|63.9%|
| Baseline  | 17.3% | 55.6%|
| Our Model  | 29.7% | 74.2%|

#### VC-Clothes
|Model| Rank-1 | mAP |
|  :----:  |  :----:  | :----:  |
|[Part-aligned](https://arxiv.org/abs/1804.07094)|69.4%|67.3%|
| Baseline  | 70.6% | 69.9%|
| Our Model  | 77.6% | 75.8%|
 

#!/bin/bash

val_arr=(2)

loop_time=${#val_arr[*]}
#loop_time=1

for i in $(seq 1 $loop_time)
do
    param_val=${val_arr[`expr $i-1`]}
    
    optim=amsgrad
    height=256
    width=128
    bs=32
    lr=0.0008
    weight_decay=0.0005
    seed=$param_val
    alpha=0.5
    spec_lr=0.003
    dim_margin=1.0
    part_num_contour=3
    part_num_rgb=3
    fix_epoch=1

    stepsize=10
    stepsize1=20    

    #stepsize=$param_val
    #stepsize1=`expr $stepsize + 20`

    python main.py \
    -s prcc_3d \
    -t prcc_3d \
    -j 4 \
    --height $height \
    --width $width \
    --optim $optim \
    --label-smooth \
    --lr $lr \
    --max-epoch 30 \
    --stepsize $stepsize $stepsize1 \
    --batch-size $bs \
    --seed $seed \
    -a model_3D_2stream_v1 \
    --root /data/jiaxing \
    --gpu-devices 1 \
    --save-dir log/prcc_3d-resent_3d-two_path_twoImgs_addSilhouette_withDisplace/amsgrad-lr0.0008-bs32-height256-seed2-alpha0.5 \
    --print-freq 10 \
    --eval-freq 0 \
    --weight-decay $weight_decay \
    --transforms random_flip1 random_crop1 \
    --flag-3d \
    --evaluate \
    --dim-margin $dim_margin \
    --dist-metric cosine 
   
done

#--save-dir log/prcc_3d-3d-two_path_imgs/$optim-$lr-bs$bs-height$height-seed$seed-alpha$alpha \
#--train-sampler RandomIdentitySampler \
#--save-dir log-bagoftricks/prcc_3d-no_3d-two_path_twoImgs-rgb_3d_cent_tri/$optim-lr$lr-bs$bs-height$height-seed$seed-alpha$alpha-dml \
#--load-weights log/prcc_3d-resent_3d-two_path_twoImgs-rgb_3d_cent_tri/amsgrad-lr0.0008-bs32-height256-seed2-alpha0.5-fused_concate-imagenet_init/model.pth.tar-25 \

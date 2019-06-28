#!/bin/bash

##################################################
# Training a watermarking network 
##################################################

# script configurations [model]
DATASET=cifar10
NETWORK=AlexNet
NETPATH=models/cifar10/train/AlexNet_cross-entropy_128_300_0.01_0.9_10_0.95.pth
VNTLOSS=l1
DNTLOSS=binary-cross-entropy
TNTLOSS=cross-entropy
# script configurations [wmark]
TRANSFM=watermark
WMARKFN=etc/watermarks/cifar10_wmark_32x32_three.png
NOISELV=0.1
BLENDFT=0.5
# script configurations [params]
BATCHSZ=64
EPOCHES=100
VLRATIO=1.0
DLRATIO=1.0
TLRATIO=1.0

# run experiments...
for GLR in 0.1 0.01 0.001
do
for VLR in 0.1 0.01 0.001
do
for DLR in 0.1 0.01 0.001
do
    # each experiment with one param. set
    echo "python3 train_wmark.py \
            --dataset $DATASET \
            --network $NETWORK --netpath $NETPATH \
            --vloss $VNTLOSS --dloss $DNTLOSS --tloss $TNTLOSS \
            --transform $TRANSFM --wmark-file $WMARKFN \
            --batch-size $BATCHSZ \
            --epoch $EPOCHES \
            --Glr $GLR --Dlr $DLR --Vlr $VLR \
            --vratio $VLRATIO --dratio $DLRATIO --tratio $TLRATIO"

    python3 train_wmark.py \
            --dataset $DATASET \
            --network $NETWORK --netpath $NETPATH \
            --vloss $VNTLOSS --dloss $DNTLOSS --tloss $TNTLOSS \
            --transform $TRANSFM --wmark-file $WMARKFN \
            --batch-size $BATCHSZ \
            --epoch $EPOCHES \
            --Glr $GLR --Dlr $DLR --Vlr $VLR \
            --vratio $VLRATIO --dratio $DLRATIO --tratio $TLRATIO
    # each...
done
done
done
# Fin.

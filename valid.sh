#!/bin/bash


##################################################
# [NOTE] when use this script, please make a copy
##################################################

# example configurations
# - wmark name: cifar10_wmark_32x32_three.png, cifar10_wmark_32x32_five.png
# - blend factors: 0.5 1.0
# - blend methods: , translate_, opavar_, spaper_
# - blend params : 0.0, 8 20 1.0

# script parameters
# - networks: AlexNet, VGG16, ResNet, ResNext, DenseNet, MobileNetV2
# - batches : 64, 128
DATASET=cifar10
NETWORK=VGG16
NETPATH=models/cifar10/train/VGG16_cross-entropy_128_300_0.1_0.9_100_0.1.pth
CLASSES=10
BATSIZE=32

# run validations
# [set one for AlexNet]
# - lr   : 0.01
# - steps: 10
# - gamma: 0.95 0.9 0.85 0.8 0.75 0.7 0.65 0.6 0.55 0.5
# [set one for others]
# - lr   : 0.1
# - steps: 100 50
# - gamma: 0.1 0.2 0.15 0.05

# : run command
echo "[valid.sh] python3 valid.py \
    --dataset $DATASET \
    --network $NETWORK \
    --trained $NETPATH \
    --classes $CLASSES \
    --batch-size $BATSIZE"
python3 valid.py \
    --dataset $DATASET \
    --network $NETWORK \
    --trained $NETPATH \
    --classes $CLASSES \
    --batch-size $BATSIZE
# Fin.

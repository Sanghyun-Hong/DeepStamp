#!/bin/bash

##################################################
# Base configurations (Dataset, Networks, etc...)
##################################################
DATASET=cifar10     # custom <- when we use synthesized watermarked dataset
DATALOC=            # custom dtaaset locations
NETWORK=AlexNet     # VGG16, ResNet


##################################################
# Network specific configuraitons (do not control..)
##################################################
# network specifics [AlexNet]
if [ $NETWORK = "AlexNet" ]; then
  NETLOSS=cross-entropy
  CLASSES=10
  # :: hyper-params
  BATCHSZ=128
  EPOCHES=100
  LR=(0.01)
  STEP=(10)
  GAMMA=(0.9 0.8 0.7 0.6 0.5)

# [VGG16]
elif [[ $NETWORK = "VGG16" ]]; then
  NETLOSS=cross-entropy
  CLASSES=10
  # :: hyper-params
  BATCHSZ=128
  EPOCHES=100
  LR=(0.1)
  STEP=(33)
  GAMMA=(0.1 0.2 0.15 0.05)

# [ResNet, ResNext]
elif [[ $NETWORK = "ResNet" ]]; then
  NETLOSS=cross-entropy
  CLASSES=10
  # :: hyper-params
  BATCHSZ=64
  EPOCHES=100
  LR=(0.1)
  STEP=(33)
  GAMMA=(0.1 0.2 0.15 0.05)
fi


##################################################
# Run experiments
##################################################
for lr in "${LR[@]}"
do
for step in "${STEP[@]}"
do
for gamma in "${GAMMA[@]}"
do
    # : run [cifar10]
    if [ $DATASET = "cifar10" ]; then
      # :: first, do training
      echo "python train.py \
        --num-workers 4 --pin-memory \
        --dataset $DATASET \
        --network $NETWORK --loss $NETLOSS --classes $CLASSES \
        --batch-size $BATCHSZ \
        --epoch $EPOCHES \
        --lr $lr --step-size $step --gamma $gamma"
      python train.py \
        --num-workers 4 --pin-memory \
        --dataset $DATASET \
        --network $NETWORK --loss $NETLOSS --classes $CLASSES \
        --batch-size $BATCHSZ \
        --epoch $EPOCHES \
        --lr $lr --step-size $step --gamma $gamma

      # :: second, run validation
      NETPATH="models/$DATASET/train/"$NETWORK"_"$NETLOSS"_"$BATCHSZ"_"$EPOCHES"_"$lr"_0.9_"$step"_"$gamma".pth"
      echo "python valid.py \
          --dataset cifar10 \
          --network $NETWORK \
          --trained $NETPATH \
          --classes 10 \
          --batch-size 32"
      python3 valid.py \
          --dataset cifar10 \
          --network $NETWORK \
          --trained $NETPATH \
          --classes 10 \
          --batch-size 32

    # : run [custom]
    else
      # :: first, do training
      echo "python train.py \
        --num-workers 4 --pin-memory \
        --dataset $DATASET \
        --datapath $DATALOC \
        --network $NETWORK --loss $NETLOSS --classes $CLASSES \
        --batch-size $BATCHSZ \
        --epoch $EPOCHES \
        --lr $lr --step-size $step --gamma $gamma"
      python train.py \
        --num-workers 4 --pin-memory \
        --dataset $DATASET \
        --datapath $DATALOC \
        --network $NETWORK --loss $NETLOSS --classes $CLASSES \
        --batch-size $BATCHSZ \
        --epoch $EPOCHES \
        --lr $lr --step-size $step --gamma $gamma

      # :: second, run validation
      DATTYPE="$(cut -d'/' -f3 <<< $DATALOC)"
      WMARKNM="$(cut -d'/' -f4 <<< $DATALOC)"
      WMMODEL="$(cut -d'/' -f5 <<< $DATALOC)"
      WMCONFI="$(cut -d'/' -f6 <<< $DATALOC)"
      NETPATH="models/$DATASET/train/cifar10_"$DATTYPE"_"$WMARKNM"_"$WMMODEL"_"$WMCONFI"_"$NETWORK"_"$NETLOSS"_"$BATCHSZ"_"$EPOCHES"_"$lr"_0.9_"$step"_"$gamma".pth"
      echo "python valid.py \
          --dataset cifar10 \
          --network $NETWORK \
          --trained $NETPATH \
          --classes 10 \
          --batch-size 32"
      python3 valid.py \
          --dataset cifar10 \
          --network $NETWORK \
          --trained $NETPATH \
          --classes 10 \
          --batch-size 32
    fi
    # : end if ...
done
done
done
# Fin

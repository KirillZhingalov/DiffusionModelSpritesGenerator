#!/bin/bash

python3 train.py \
    --diffusion-timestamps 500 \
    --diffusion-beta1 0.0001 \
    --diffusion-beta2 0.02 \
    --n-feat 256 \
    --n-cfeat 10 \
    --image-size 16 \
    --save-dir ./weights \
    --batch-size 128 \
    --n-epoch 32 \
    --lrate 0.001

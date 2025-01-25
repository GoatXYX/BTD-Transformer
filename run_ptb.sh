#!/bin/bash

if [ "$1" == "train" ]; then
    echo 'Run training...'
    pwd
    python train_upload.py \
        --cuda \
        --data /home/yuxuanxia/BTD-Transformer/data/ptb \
        --dataset ptb \
        --n_layer 2 \
        --d_model 128 \
        --n_head 1 \
        --d_head 32 \
        --d_inner 1024 \
        --dropout 0.3 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00005 \
        --warmup_step 0 \
        --max_step 20000 \
        --tgt_len 32 \
        --mem_len 0 \
        --eval_tgt_len 32 \
        --batch_size 20 \
        --gpu0_bsz 1
else
    echo "Usage: $0 train"
fi

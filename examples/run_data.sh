#!/bin/bash
source ./set_env.sh
python train.py --dataset /raid/zhangyourong/model/nsynth-train-mean/data_train_cosine00.pkl \
                --cluster hyperbolic \
                --epochs 50 \
                --batch_size 512 \
                --learning_rate 1e-4 \
                --temperature 5e-2 \
                --eval_every 1 \
                --patience 40 \
                --optimizer RAdam \
                --anneal_every 10 \
                --anneal_factor 0.5 \
                --init_size 0.05 \
                --num_samples 900000 \
                --seed 0

#!/usr/bin/env bash
python main.py --meta_model warp_leap --meta_batch_size 5 --meta_train_steps 250 --num_pretrain $1 --suffix $2 --overwrite
#!/usr/bin/env bash
python main.py --meta_model warp_leap --meta_batch_size 5 --task_train_steps 25 --task_val_steps 25 --num_pretrain $1 --suffix $2 --overwrite
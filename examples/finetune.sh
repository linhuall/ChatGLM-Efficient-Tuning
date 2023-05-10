#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python /kaggle/working/model/src/finetune.py \
    --do_train \
    --dataset datasetV3 \
    --dataset_dir /kaggle/working/model/data \
    --finetuning_type lora \
    --output_dir /kaggle/working/checkpoint \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 #\
    # --fp16

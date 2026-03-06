#!/bin/bash

# 禁用 wandb 记录（由于是离线环境/Kaggle环境）
export WANDB_MODE=disabled
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 启动 DeepSpeed 分布式训练
deepspeed --num_gpus=2 llava/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --bits 4 \
    --lora_r 64 \
    --lora_alpha 128 \
    --model_name_or_path /kaggle/input/datasets/monolith0456/vicuna-7b-v1-5 \
    --vision_tower /kaggle/input/datasets/jeinsong/openai-clip-vit-large-patch14-336 \
    --data_path /kaggle/temp/FakeClue_Official/data_json/train_mini.json \
    --image_folder /kaggle/temp/FakeClue_Official/images/train/ \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio square \
    --fp16 True \
    --output_dir /kaggle/working/checkpoints/llava-srm-v1 \
    --max_steps 200 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True

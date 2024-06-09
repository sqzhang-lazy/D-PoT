#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:

################## VICUNA ##################
# PROMPT_VERSION=v1
# MODEL_VERSION="vicuna-v1-3-7b"
################## VICUNA ##################

################## LLaMA-2 ##################
PROMPT_VERSION="llava_llama_2"
MODEL_VERSION="llama-2-7b-chat"
################## LLaMA-2 ##################

EXPERIMENT_NAME='llava-aitw'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

deepspeed llava/train/train_mem.py \
    --run_name $EXPERIMENT_NAME \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path llava-7b-hf \
    --version $PROMPT_VERSION \
    --data_path data/google_apps_plan.json \
    --vision_tower ./clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./llava-hf/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-$MODEL_VERSION-finetune-wplan-v4 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2560 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

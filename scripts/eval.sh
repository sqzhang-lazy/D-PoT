CUDA_VISIBLE_DEVICES=2 python llava/eval/model_aitw_llava_v4.py \
    --model_path checkpoints/llava-llama-2-7b-chat-finetune-wplan-v4 \
    --question_file data/single_llava.json \
    --answers_file llava_single_finetunewplan_plan_v4.json
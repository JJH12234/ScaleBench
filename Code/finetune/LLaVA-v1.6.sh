# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export CUDA_LAUNCH_BLOCKING=1
CUDA_VISIBLE_DEVICES=1,2 \
swift sft \
    --model llava-hf/llava-v1.6-mistral-7b-hf \
    --model_type llava1_6_mistral_hf \
    --template llava1_6_mistral_hf \
    --train_type lora \
    --dataset train.json \
    --torch_dtype float16 \
    --num_train_epochs 6 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-5 \
    --lora_rank 32 \
    --lora_alpha 16 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --save_total_limit 10 \
    --save_steps 100 \
    --logging_steps 20 \
    --max_length 8000 \
    --output_dir llava-v1.6-mistral-7b \
    --system 'You are currently a senior expert in scale recognition.' \
    --dataloader_num_workers 16 \
    --optim adamw_torch \
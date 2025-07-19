# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export CUDA_LAUNCH_BLOCKING=1
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model deepseek-ai/Janus-Pro-7B \
    --model_type deepseek_janus_pro \
    --template deepseek_janus_pro \
    --train_type lora \
    --dataset train.json \
    --torch_dtype float16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 2 \
    --learning_rate 2e-4 \
    --lora_rank 32 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --save_total_limit 10 \
    --save_steps 100 \
    --logging_steps 20 \
    --max_length 2048 \
    --output_dir Janus-Pro-7B \
    --system 'You are currently a senior expert in scale recognition.' \
    --dataloader_num_workers 16 \
    --split_dataset_ratio 0
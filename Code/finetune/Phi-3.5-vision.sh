# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export CUDA_LAUNCH_BLOCKING=1
CUDA_VISIBLE_DEVICES=0,3 \
swift sft \
    --model microsoft/Phi-3.5-vision-instruct \
    --model_type phi3_vision \
    --template phi3_vision \
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
    --max_length 1024 \
    --output_dir Phi-3.5-vision \
    --system 'You are currently a senior expert in scale and measurement recognition.' \
    --dataloader_num_workers 16 \
    --gradient_checkpointing false \
    --attn_impl eager \
    --optim adamw_torch \
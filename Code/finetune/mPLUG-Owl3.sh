# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export CUDA_LAUNCH_BLOCKING=1
CUDA_VISIBLE_DEVICES=3 \
swift sft \
    --model mPLUG/mPLUG-Owl3-7B-240728 \
    --model_type mplug_owl3 \
    --template mplug_owl3 \
    --train_type lora \
    --dataset train.json \
    --torch_dtype float16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 2 \
    --learning_rate 2e-4 \
    --lora_rank 32 \
    --lora_alpha 16 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --save_total_limit 10 \
    --save_steps 100 \
    --logging_steps 10 \
    --max_length 2048 \
    --output_dir mPLUG-Owl3-7B \
    --system 'You are currently a senior expert in scale and measurement recognition.' \
    --dataloader_num_workers 16 \
    --split_dataset_ratio 0
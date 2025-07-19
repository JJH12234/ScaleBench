export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path OpenGVLab/InternVL3-8B-hf \
    #--model_name_or_path OpenGVLab/InternVL3-14B-hf \
    #--model_name_or_path OpenGVLab/InternVL3-38B-hf \
    #--model_name_or_path OpenGVLab/InternVL3-78B-hf \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template intern_vl \
    --dataset_dir data \
    --dataset your_dataset \
    --cutoff_len 2048 \
    --learning_rate 2e-04 \
    --num_train_epochs 10 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir InternVL3 \
    --fp16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 32 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all \
    
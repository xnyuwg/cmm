GPUS=${1:-0}

CUDA_VISIBLE_DEVICES=${GPUS} python run_cmm.py \
--run_name=exp2 \
--gradient_accumulation_steps=32 \
--max_epochs=100 \
--use_partial_data=True

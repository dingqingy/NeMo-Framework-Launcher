#!/bin/bash

cd ../../megatron-lm
# MCORE_BRANCH=$(git rev-parse --abbrev-ref HEAD)
MCORE_COMMIT=$(git rev-parse --short HEAD)
cd -

OUT_DIR=8_19_debug
WB_PROJ=dingqingy_${OUR_DIR}

NUM_NODES=1
NUM_GPUS=$((NUM_NODES*8))
TP=2
PP=4
VP=11
DP=$((NUM_GPUS/TP/PP))

# N=$PP
N=$((PP+1))
M=${N} # num of microbatch per pipeline

MBS=1
GBS=$((MBS*M*DP))
SEQ_LEN=4096

echo "num GPU: $NUM_GPUS, dp size: $DP, N${N}M${M}, GBS: ${GBS}"
python main.py \
    training=gpt3/20b \
    base_results_dir=$(pwd)/${OUT_DIR} \
    training.run.prefix=${MCORE_COMMIT}_N${N}M${M} \
    training.trainer.num_nodes=${NUM_NODES} \
    training.model.micro_batch_size=${MBS} \
    training.model.global_batch_size=${GBS} \
    training.model.tensor_model_parallel_size=${TP} \
    training.model.pipeline_model_parallel_size=${PP} \
    training.model.encoder_seq_length=${SEQ_LEN} \
    training.trainer.max_steps=100 \
    training.run.time_limit=0:20:00 \
    training.exp_manager.create_wandb_logger=True \
    training.exp_manager.wandb_logger_kwargs.project=${WB_PROJ} \
    +training.model.optim.grad_sync_dtype=bf16 \
    ++training.model.cross_entropy_loss_fusion=true \
    ++training.model.defer_embedding_wgrad_compute=true \
    ++training.model.mcore_customization_config.contiguous_micro_batch=${N} 
    # training.exp_manager.wandb_logger_kwargs.project=dingqingy_gpt3_debug_overlap
    # ++training.model.deterministic_mode=false \


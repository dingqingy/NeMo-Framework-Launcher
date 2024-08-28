#!/bin/bash

cd ../../megatron-lm
MCORE_BRANCH=$(git rev-parse --abbrev-ref HEAD)
# MCORE_COMMIT=$(git rev-parse --short HEAD)
cd -

OUT_DIR=8_27_debug
WB_PROJ=dingqingy_${OUT_DIR}

NUM_NODES=1
NUM_GPUS=$((NUM_NODES*8))
TP=2
PP=4
VP=11
DP=$((NUM_GPUS/TP/PP))

# N=$PP
N=$((PP+1))
# M=${N} # num of microbatch per pipeline
M=$((2*N-1))

MBS=1
GBS=$((MBS*M*DP))
SEQ_LEN=4096

echo "num GPU: $NUM_GPUS, dp size: $DP, N${N}M${M}, GBS: ${GBS}"
python main.py \
    training=gpt3/20b \
    base_results_dir=$(pwd)/${OUT_DIR} \
    training.run.prefix=${MCORE_BRANCH}_N${N}M${M} \
    training.trainer.num_nodes=${NUM_NODES} \
    training.model.micro_batch_size=${MBS} \
    training.model.global_batch_size=${GBS} \
    training.model.tensor_model_parallel_size=${TP} \
    training.model.pipeline_model_parallel_size=${PP} \
    training.model.virtual_pipeline_model_parallel_size=${VP} \
    training.model.encoder_seq_length=${SEQ_LEN} \
    training.trainer.max_steps=50 \
    training.run.time_limit=0:20:00 \
    +training.model.optim.grad_sync_dtype=bf16 \
    ++training.model.cross_entropy_loss_fusion=true \
    ++training.model.defer_embedding_wgrad_compute=true \
    training.exp_manager.create_wandb_logger=True \
    training.exp_manager.wandb_logger_kwargs.project=${WB_PROJ} \
    ++training.model.deterministic_mode=false \
    ++training.model.mcore_customization_config.num_microbatches_per_virtual_pipe=${N} 


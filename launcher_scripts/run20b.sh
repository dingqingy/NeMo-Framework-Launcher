#!/bin/bash

cd ../../megatron-lm
MCORE_BRANCH=$(git rev-parse --abbrev-ref HEAD)
# MCORE_COMMIT=$(git rev-parse --short HEAD)
cd -

OUT_DIR=9_4_pp_only
WB_PROJ=dingqingy_${OUT_DIR}

NUM_NODES=1
NUM_GPUS=$((NUM_NODES*8))
TP=1
PP=8
VP=5
DP=$((NUM_GPUS/TP/PP))

# N=$PP
N=$((PP+1))
# M=${N} # num of microbatch per pipeline
M=$((2*N-1))

MBS=1
GBS=$((MBS*M*DP))
SEQ_LEN=4096
NUM_LAYERS=$((PP*VP))

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
    training.model.num_layers=${NUM_LAYERS} \
    training.trainer.max_steps=10 \
    training.run.time_limit=0:30:00 \
    +training.model.optim.grad_sync_dtype=bf16 \
    ++training.model.cross_entropy_loss_fusion=true \
    ++training.model.defer_embedding_wgrad_compute=true \
    ++training.model.ub_tp_comm_overlap=false \
    ++training.model.sequence_parallel=false \
    ++training.trainer.check_val_every_n_epoch=null \
    ++training.trainer.num_sanity_val_steps=1 \
    training.exp_manager.create_wandb_logger=True \
    training.exp_manager.wandb_logger_kwargs.project=${WB_PROJ} \
    +env_vars.NCCL_DEBUG=trace \
    +env_vars.NCCL_DEBUG_SUBSYS=call \
    ++training.model.mcore_customization_config.num_microbatches_per_virtual_pipe=${N} 
    # ++training.model.deterministic_mode=false \


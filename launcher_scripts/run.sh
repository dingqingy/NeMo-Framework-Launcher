#!/bin/bash

cd ../../megatron-lm
MCORE_BRANCH=$(git rev-parse --abbrev-ref HEAD)
# MCORE_COMMIT=$(git rev-parse --short HEAD)
cd -

OUT_DIR=8_26_perf
WB_PROJ=dingqingy_${OUT_DIR}

NUM_NODES=16
NUM_GPUS=$((NUM_NODES*8))
TP=4
PP=8
VP=12
DP=$((NUM_GPUS/TP/PP))

# N=$PP
N=9
# N=$((PP+1))
# M=${N}
# M=$((2*N-1))
M=17

MBS=1
GBS=$((MBS*M*DP))
SEQ_LEN=2048

MAX_STEPS=100
MINITES=31

echo "num GPUs: $NUM_GPUS, dp $DP, N${N}M${M} GBS ${GBS}, seq len ${SEQ_LEN}"

python3 main.py \
    training=gpt3/175b \
    base_results_dir=$(pwd)/${OUT_DIR} \
    training.run.prefix=${MCORE_BRANCH}_N${N}M${M} \
    training.trainer.num_nodes=${NUM_NODES} \
    training.model.global_batch_size=${GBS} \
    training.model.micro_batch_size=${MBS} \
    training.model.tensor_model_parallel_size=${TP} \
    training.model.pipeline_model_parallel_size=${PP} \
    training.model.virtual_pipeline_model_parallel_size=${VP} \
    training.model.encoder_seq_length=${SEQ_LEN} \
    training.trainer.max_steps=${MAX_STEPS} \
    training.run.time_limit=0:${MINITES}:00 \
    +training.model.optim.grad_sync_dtype=bf16 \
    ++training.model.cross_entropy_loss_fusion=true \
    ++training.model.defer_embedding_wgrad_compute=true \
    ++training.trainer.check_val_every_n_epoch=null \
    ++training.trainer.num_sanity_val_steps=0 \
    training.exp_manager.create_wandb_logger=True \
    training.exp_manager.wandb_logger_kwargs.project=${WB_PROJ} \
    ++training.model.deterministic_mode=false \
    ++training.model.ub_tp_comm_overlap=true \
    training.model.nsys_profile.enabled=true \
    training.model.nsys_profile.start_step=50 \
    training.model.nsys_profile.end_step=52 \
    training.model.nsys_profile.ranks=[0,16,32,48,64,80,96,112] \
    ++training.model.mcore_customization_config.num_microbatches_per_virtual_pipe=${N}

# checkpoint_callback, limit_val/test_batch
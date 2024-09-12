#!/bin/bash

cd ../../megatron-lm
MCORE_BRANCH=$(git rev-parse --abbrev-ref HEAD)
# MCORE_COMMIT=$(git rev-parse --short HEAD)
cd -

OUT_DIR=9_11_smallest_hang
WB_PROJ=dingqingy_${OUT_DIR}

NUM_NODES=1
NUM_GPUS=$((NUM_NODES*8))
TP=2
PP=4
VP=2
# VP=2
DP=$((NUM_GPUS/TP/PP))

# N=$PP
# N=9
N=$((PP+1))
M=${N}
# M=$((2*N-1))
# M=$((3*N))
# M=17

MBS=1
GBS=$((MBS*M*DP))
NUM_LAYERS=$((PP*VP))

# ok timing
# SEQ_LEN=4096
# HIDDEN_SIZE=6144
# NUM_HEADS=48
# broken timing
SEQ_LEN=1024
HIDDEN_SIZE=12288
NUM_HEADS=96

MAX_STEPS=50
MINITES=30

echo "num GPUs: $NUM_GPUS, dp $DP, N${N}M${M} GBS ${GBS}, seq len ${SEQ_LEN}"

python3 main.py \
    training=gpt3/175b \
    base_results_dir=$(pwd)/${OUT_DIR} \
    training.run.prefix=${MCORE_BRANCH}_N${N}M${M}_seq${SEQ_LEN}_hid${HIDDEN_SIZE}_heads${NUM_HEADS} \
    training.trainer.num_nodes=${NUM_NODES} \
    training.model.global_batch_size=${GBS} \
    training.model.micro_batch_size=${MBS} \
    training.model.tensor_model_parallel_size=${TP} \
    training.model.pipeline_model_parallel_size=${PP} \
    training.model.virtual_pipeline_model_parallel_size=${VP} \
    training.model.encoder_seq_length=${SEQ_LEN} \
    training.model.hidden_size=${HIDDEN_SIZE} \
    training.model.num_attention_heads=${NUM_HEADS} \
    training.model.num_layers=${NUM_LAYERS} \
    training.trainer.max_steps=${MAX_STEPS} \
    training.run.time_limit=0:${MINITES}:00 \
    +training.model.optim.grad_sync_dtype=bf16 \
    ++training.model.cross_entropy_loss_fusion=true \
    ++training.model.defer_embedding_wgrad_compute=true \
    ++training.trainer.check_val_every_n_epoch=null \
    ++training.trainer.val_check_interval=1 \
    ++training.trainer.num_sanity_val_steps=0 \
    training.exp_manager.create_wandb_logger=True \
    training.exp_manager.wandb_logger_kwargs.project=${WB_PROJ} \
    ++training.model.ub_tp_comm_overlap=false \
    ++training.model.sequence_parallel=false \
    training.model.nsys_profile.enabled=true \
    training.model.nsys_profile.start_step=19 \
    training.model.nsys_profile.end_step=21 \
    training.model.nsys_profile.ranks=[0,$((NUM_GPUS-1))] \
    ++training.model.mcore_customization_config.num_microbatches_per_virtual_pipe=${N}
    # training.model.nsys_profile.ranks=[0,2,4,6] \
    # +env_vars.NCCL_DEBUG=trace \
    # +env_vars.NCCL_DEBUG_SUBSYS=call \
    # training.model.nsys_profile.ranks=[0,16,32,48,64,80,96,112] \
    # ++training.model.deterministic_mode=false \

# checkpoint_callback, limit_val/test_batch

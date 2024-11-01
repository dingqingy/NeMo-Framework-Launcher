#!/bin/bash

cd ../../megatron-lm
MCORE_BRANCH=$(git rev-parse --abbrev-ref HEAD)
# MCORE_COMMIT=$(git rev-parse --short HEAD)
cd -

OUT_DIR=9_26_verification
WB_PROJ=dingqingy_${OUT_DIR}

NUM_NODES=16
NUM_GPUS=$((NUM_NODES*8))
TP=4
PP=8
# PROFILED_RANKS="[0,2,4,6]"
PROFILED_RANKS="[0,16,32,48,64,80,96,112]"
VP=12
# VP=3
DP=$((NUM_GPUS/TP/PP))

# N=${1}
# M=${2}
N=$PP
# N=9
# N=$((PP+1))
# N=$((PP+2))
# M=${N}
# M=$((2*N))
# M=$((N*PP))
M=$((9*N))
# M=17

MBS=1
GBS=$((MBS*M*DP))
NUM_LAYERS=$((PP*VP))

# 20B
# SEQ_LEN=4096
# HIDDEN_SIZE=6144
# NUM_HEADS=48
# 175B
SEQ_LEN=2048
HIDDEN_SIZE=12288
NUM_HEADS=96

MAX_STEPS=250
MINITES=30

# TP_OPT_ENABLED=false
if [[ $TP -gt 1 ]]
then
    TP_OPT_ENABLED=true
else    
    TP_OPT_ENABLED=false
fi

# WARMUP_OVERLAP=false
# FLUSH_OVERLAP=false
WARMUP_OVERLAP=${1}
FLUSH_OVERLAP=${2}

# RUN_PREFIX=${MCORE_BRANCH}_N${N}M${M}_warmup_${WARMUP_OVERLAP}_flush_${FLUSH_OVERLAP}
RUN_PREFIX=warmup_${WARMUP_OVERLAP}_flush_${FLUSH_OVERLAP}_N${N}M${M}

echo "num GPUs: $NUM_GPUS, dp $DP, N${N}M${M} GBS ${GBS}, seq len ${SEQ_LEN}"

python3 main.py \
    training=gpt3/175b \
    base_results_dir=$(pwd)/${OUT_DIR} \
    training.run.prefix=${RUN_PREFIX} \
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
    ++training.trainer.num_sanity_val_steps=0 \
    training.exp_manager.create_wandb_logger=True \
    training.exp_manager.wandb_logger_kwargs.project=${WB_PROJ} \
    ++training.model.ub_tp_comm_overlap=${TP_OPT_ENABLED} \
    ++training.model.sequence_parallel=${TP_OPT_ENABLED} \
    training.model.nsys_profile.enabled=true \
    training.model.nsys_profile.start_step=27 \
    ++training.trainer.val_check_interval=10 \
    ++training.trainer.limit_val_batches=5 \
    training.model.nsys_profile.end_step=31 \
    training.model.nsys_profile.ranks=$PROFILED_RANKS \
    training.trainer.log_every_n_steps=1 \
    training.exp_manager.step_timing_kwargs.buffer_size=1 \
    +env_vars.NEMO_MANUAL_GC_IN_VALIDATION=0 \
    +training.exp_manager.create_neptune_logger=false \
    +training.exp_manager.create_tensorboard_logger=false \
    ++training.model.mcore_customization_config.enable_warmup_overlap=${WARMUP_OVERLAP} \
    ++training.model.mcore_customization_config.enable_flush_overlap=${FLUSH_OVERLAP} \
    ++training.model.mcore_customization_config.num_microbatches_per_virtual_pipe=${N}

    # training.model.nsys_profile.ranks=[0,$(( $NUM_GPUS - $TP * $DP ))] \
    # training.model.nsys_profile.ranks=[0,2,4,6] \
    # +env_vars.NCCL_DEBUG_SUBSYS=call \
    # training.model.nsys_profile.ranks=[0,16,32,48,64,80,96,112] \
    # ++training.model.deterministic_mode=false \

# checkpoint_callback, limit_val/test_batch

#!/bin/bash

#!/bin/bash
MEGATRON_PATH="../../../megatron-lm" # path to megatron-lm
LAUNCHER_PATH="../" # path to launcher_scripts

cd $MEGATRON_PATH
# git switch main
git switch decouple_send_recv_issue
# git switch zb_h1
MCORE_BRANCH=$(git rev-parse --abbrev-ref HEAD)
# MCORE_COMMIT=$(git rev-parse --short HEAD)
cd -

OUT_DIR=10_18_llama3
WB_PROJ=dingqingy_${OUT_DIR}

NUM_NODES=16
# NUM_NODES=8
NUM_GPUS=$((NUM_NODES*8))
TP=4
CP=2
# PP=4
PP=8
# PROFILED_RANKS="[0,16,32,48]"
PROFILED_RANKS="[0,16,32,48,64,80,96,112]"
# PROFILED_RANKS="[0,4,8,12]"
VP=10
# NUM_LAYERS_PER_MODEL_CHUNK=2
DP=$((NUM_GPUS/TP/PP/CP))

N=$PP

MBS=1

MAX_STEPS=250
MINITES=30

if [[ $TP -gt 1 ]]
then
    TP_OPT_ENABLED=true
else    
    TP_OPT_ENABLED=false
fi


P2P_OVERLAP=true

# for N in $PP; do
# for N in $((PP+1)); do
for N in $PP $((PP+1)); do
    M=$N
    # NUM_LAYERS=$((PP*VP*NUM_LAYERS_PER_MODEL_CHUNK))
    GBS=$((MBS*M*DP))
    # for WARMUP_FLUSH_OVERLAP in false; do
    for WARMUP_FLUSH_OVERLAP in true false; do
        RUN_PREFIX=${MCORE_BRANCH}_N${N}M${M}_warmup_flush_${WARMUP_FLUSH_OVERLAP}

        echo "num GPUs: $NUM_GPUS, dp $DP, N${N}M${M} GBS ${GBS}"

        python3 $LAUNCHER_PATH/main.py \
            training=llama/llama3_70b \
            base_results_dir=$(pwd)/${OUT_DIR} \
            training.run.prefix=${RUN_PREFIX} \
            training.trainer.num_nodes=${NUM_NODES} \
            training.model.global_batch_size=${GBS} \
            training.model.micro_batch_size=${MBS} \
            training.model.tensor_model_parallel_size=${TP} \
            training.model.context_parallel_size=${CP} \
            training.model.pipeline_model_parallel_size=${PP} \
            training.model.virtual_pipeline_model_parallel_size=${VP} \
            training.trainer.max_steps=${MAX_STEPS} \
            training.run.time_limit=0:${MINITES}:00 \
            training.model.fp8=False \
            training.model.fp8_hybrid=False \
            +training.model.optim.grad_sync_dtype=bf16 \
            ++training.model.optim.overlap_grad_sync=false \
            ++training.model.cross_entropy_loss_fusion=true \
            ++training.model.defer_embedding_wgrad_compute=true \
            ++training.trainer.check_val_every_n_epoch=null \
            ++training.trainer.num_sanity_val_steps=0 \
            training.exp_manager.create_wandb_logger=True \
            training.exp_manager.create_checkpoint_callback=false \
            training.exp_manager.wandb_logger_kwargs.project=${WB_PROJ} \
            ++training.model.ub_tp_comm_overlap=${TP_OPT_ENABLED} \
            ++training.model.sequence_parallel=${TP_OPT_ENABLED} \
            ++training.model.overlap_p2p_comm=${P2P_OVERLAP} \
            training.model.nsys_profile.enabled=true \
            training.model.nsys_profile.start_step=27 \
            ++training.trainer.val_check_interval=30 \
            ++training.trainer.limit_val_batches=5 \
            training.model.nsys_profile.end_step=31 \
            training.model.nsys_profile.ranks=$PROFILED_RANKS \
            training.trainer.log_every_n_steps=1 \
            training.exp_manager.step_timing_kwargs.buffer_size=1 \
            +env_vars.NEMO_MANUAL_GC_IN_VALIDATION=0 \
            +training.exp_manager.create_neptune_logger=false \
            ++training.model.overlap_p2p_comm_warmup_flush=${WARMUP_FLUSH_OVERLAP} \
            ++training.model.microbatch_group_size_per_vp_stage=${N}

            # training.model.num_layers=${NUM_LAYERS} \
            # TE determinstics
            # +env_vars.NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
            # For perf benchmarking, can avoid GC impact using the following as well as disabling logger
            # Also validation frequency can be lower as gc frequency is at least as frequent as validation based on launcher (see stages.py)
            # +env_vars.NEMO_MANUAL_GC_IN_VALIDATION=0 \
            # +training.exp_manager.create_neptune_logger=false \
            # +training.exp_manager.create_tensorboard_logger=false \
            
            # training.model.nsys_profile.ranks=[0,$(( $NUM_GPUS - $TP * $DP ))] \
            # training.model.nsys_profile.ranks=[0,2,4,6] \
            # +env_vars.NCCL_DEBUG_SUBSYS=call \
            # training.model.nsys_profile.ranks=[0,16,32,48,64,80,96,112] \
            # ++training.model.deterministic_mode=false \
            # checkpoint_callback, limit_val/test_batch
done
done

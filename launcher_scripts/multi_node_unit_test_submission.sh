#!/bin/bash

# Parameters
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --dependency=singleton
#SBATCH --error=/lustre/fsw/coreai_dlalgo_llm/dingqingy/NeMo-Framework-Launcher/launcher_scripts/9_3_debug/multi_node_unit_test_%j.err
#SBATCH --exclusive
#SBATCH --job-name=coreai_dlalgo_llm.multi_node_unit_test
#SBATCH --mem=0
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --output=/lustre/fsw/coreai_dlalgo_llm/dingqingy/NeMo-Framework-Launcher/launcher_scripts/9_3_debug/multi_node_unit_test_%j.out
#SBATCH --partition=batch
#SBATCH --time=0:20:00

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

# command 1
srun --output /lustre/fsw/coreai_dlalgo_llm/dingqingy/NeMo-Framework-Launcher/launcher_scripts/9_3_debug/multi_node_unit_test_%j.out --error /lustre/fsw/coreai_dlalgo_llm/dingqingy/NeMo-Framework-Launcher/launcher_scripts/9_3_debug/multi_node_unit_test_%j.err --container-image gitlab-master.nvidia.com/dingqingy/containers:nemo_dev --container-mounts /lustre/fsw/coreai_dlalgo_llm/dingqingy/NeMo-Framework-Launcher/launcher_scripts:/lustre/fsw/coreai_dlalgo_llm/dingqingy/NeMo-Framework-Launcher/launcher_scripts,/lustre/fsw/coreai_dlalgo_llm/dataset/the_pile/train:/lustre/fsw/coreai_dlalgo_llm/dataset/the_pile/train,/lustre/fsw/coreai_dlalgo_llm/dingqingy/NeMo-Framework-Launcher/launcher_scripts/9_3_debug:/lustre/fsw/coreai_dlalgo_llm/dingqingy/NeMo-Framework-Launcher/launcher_scripts/9_3_debug,/lustre/fsw/coreai_dlalgo_llm/dingqingy/megatron-lm/megatron/core/pipeline_parallel:/opt/megatron-lm/megatron/core/pipeline_parallel,/lustre/fsw/coreai_dlalgo_llm/dingqingy/megatron-lm/megatron/core/transformer:/opt/megatron-lm/megatron/core/transformer,/lustre/fsw/coreai_dlalgo_llm/dingqingy/megatron-lm/megatron/core/models/gpt:/opt/megatron-lm/megatron/core/models/gpt --no-container-mount-home bash -c "
  torchrun \
  --nnodes 8 \
  --nproc_per_node 1 \
  --rdzv_id $RANDOM \
  --rdzv_backend c10d \
  --rdzv_endpoint $head_node_ip:29500 \
  /opt/megatron-lm/megatron/core/pipeline_parallel/unit_test/p2p_unit_test.py"

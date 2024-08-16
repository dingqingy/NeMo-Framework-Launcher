cd ../../megatron-lm
MCORE_BRANCH=$(git rev-parse --abbrev-ref HEAD)
cd -

python main.py \
    training.model.mcore_customization_config.contiguous_micro_batch=5 \
    training.model.global_batch_size=5 \
    training.run.prefix=${MCORE_BRANCH} \
    base_results_dir=$(pwd)/8_15_debugging \
    training.trainer.max_steps=100 \
    ++training.model.deterministic_mode=false
    # training.exp_manager.wandb_logger_kwargs.project=dingqingy_gpt3_overlap

python main.py \
    training.model.mcore_customization_config.contiguous_micro_batch=5 \
    training.model.global_batch_size=5 \
    training.run.prefix=tunable_overlap \
    ++training.model.deterministic_mode=false
    
    # training.exp_manager.wandb_logger_kwargs.project=dingqingy_gpt3_overlap
    # base_result_dir

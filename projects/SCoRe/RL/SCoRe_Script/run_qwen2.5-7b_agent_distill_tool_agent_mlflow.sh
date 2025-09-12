#!/bin/bash
# run on 8xH100
# make sure your current working directory is the root of the project

set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/SCoRe_Script/multiturn_config"
TOOL_CONFIG_PATH="$PROJECT_DIR/SCoRe_Script/multiturn_config/tool_config.yaml"

PROJECT_NAME="SCoRe-RL"
EXPERIMENT_NAME="qwen-2.5-7b"

# Hyperparameters
TRAIN_BATCH_SIZE=128
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=4096
PPO_MINI_BATCH_SIZE=16
LR=7e-7

# Model path and save path
BEFOREPOINTS="..."
AFTERPOINTS="checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME"

# Data
TRAIN_FILES="my_data/SCoRe_math/train.parquet"
VAL_FILES="my_data/SCoRe_math/SCoRe_val.parquet"

N_GPUS_PER_NODE=8


python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='all_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=${BEFOREPOINTS}  \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.format=agentdistill \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.trace.backend=mlflow \
    actor_rollout_ref.rollout.trace.token2text=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=${TOOL_CONFIG_PATH} \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "mlflow"]' \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${AFTERPOINTS} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=6 \
    trainer.test_freq=2 \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    trainer.total_epochs=3 \
    reward_model.reward_manager=batch \
    custom_reward_function.path=recipe/SCoRe/reward_function.py \
    custom_reward_function.name=compute_score_batch \

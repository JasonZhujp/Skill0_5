set -x
ENGINE=${1:-vllm}
shift  # Remove first argument so $@ only contains extra params
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# ==================== WandB (optional) ====================
# export WANDB_API_KEY="your_key_here"

export VERL_LOGGING_LEVEL=INFO
export HYDRA_FULL_ERROR=1

export RAY_BACKEND_LOG_LEVEL=warning
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DISABLE_GPU_MONITOR=1
export RAY_DEBUG_POST_MORTEM=0
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export RAY_ROTATION_MAX_BYTES=52428800
export RAY_ROTATION_BACKUP_COUNT=3
export TORCH_NCCL_AVOID_RECORD_STREAMS="1"

# Limit thread count to prevent pthread_create failures under Ray
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export RAY_IGNORE_UNHANDLED_ERRORS=1
export RAY_worker_register_timeout_seconds=600
export RAY_TASK_MAX_RETRIES=3
export RAY_memory=100000000000
export RAY_object_store_memory=40000000000

pip3 install alfworld
pip3 install sentence-transformers faiss-cpu

# ==================== Model & Data Config ====================
export MODEL_PATH="${MODEL_PATH:?Please set MODEL_PATH to your SFT checkpoint}"
export ALFWORLD_DATA="${ALFWORLD_DATA:?Please set ALFWORLD_DATA to the ALFWorld data directory}"

export WANDB_NAME="skill05_alfworld_ood"
project_name=skill_alfworld_ood
RUN_ID="$(date +%m%d-%H%M%S)"
experiment_name="${WANDB_NAME}${RUN_ID:+_$RUN_ID}"

export OUTPUT_DIR="./outputs/${project_name}/${experiment_name}"
mkdir -p "${OUTPUT_DIR}/logs"
export WANDB_DIR="${OUTPUT_DIR}/wandb"
mkdir -p "$WANDB_DIR"

# ==================== Training Hyperparameters ====================
num_cpus_per_env_worker=0.25
train_data_size=16
val_data_size=140
group_size=8

# Prepare placeholder data
python3 -m data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo_ood \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=6000 \
    data.max_response_length=768 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=30 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    +env.use_skills_only_memory=True \
    +env.skills_only_memory.skills_json_path=memory_data/alfworld_ood/claude_style_skills_id.json \
    +env.skills_only_memory.retrieval_mode=embedding \
    +env.skills_only_memory.embedding_model_path=Qwen/Qwen3-Embedding-0.6B \
    +env.skills_only_memory.top_k=3 \
    +env.skills_only_memory.enable_dynamic_update=False \
    +env.skills_only_memory.update_threshold=0.4 \
    +env.skills_only_memory.max_new_skills=3 \
    +env.alfworld_ood.id_task_types='[1,3,5]' \
    +env.alfworld_ood.ood_task_types='[2,4,6]' \
    +env.alfworld_ood.skills_json_path=memory_data/alfworld_ood/claude_style_skills_ood.json \
    +env.guide_internalize=True \
    +env.ours_mode=True \
    +env.ours.warmup_steps=0 \
    +env.ours.window_size=5 \
    +env.utilize.decomposed_contrastive=True \
    +env.utilize.omega=1.0 \
    +env.utilize.use_ema_delta=True \
    +env.utilize.delta_baseline_mode=window \
    +env.utilize.delta_window_size=5 \
    +env.utilize.ema_delta_alpha=0.1 \
    +env.utilize.adv2_clip=3.0 \
    +env.internalize.jsd_lambda=1.0 \
    +env.internalize.jsd_top_k=64 \
    +env.internalize.jsd_temperature=1.0 \
    actor_rollout_ref.actor.policy_loss.guide.enabled=False \
    actor_rollout_ref.actor.ppo_epochs=1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.rollout_data_dir=${OUTPUT_DIR}/rollout_data \
    +trainer.val_dump_path=${OUTPUT_DIR}/val_traj \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=999 \
    trainer.total_training_steps=120 \
    trainer.val_before_train=False $@ 2>&1 | tee "${OUTPUT_DIR}/logs/$(date +%Y%m%d_%H%M%S).log"

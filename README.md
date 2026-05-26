# Skill0.5

Code for *Skill0.5: Joint Skill Internalization and Utilization for Out-of-Distribution Generalization in Agentic Reinforcement Learning*.

## Overview

Existing skill-based RL methods force a rigid choice between full externalization (prohibitive context overhead) and full internalization (overfitting and knowledge conflicts). **Skill0.5** resolves this by jointly combining **general skill internalization** with **task-specific skill utilization**, treating the two inherently different skill types with distinct optimization strategies.

A dynamic *difficulty-aware router* streams tasks into mastery tiers:
- **Hard tasks**: internalize general skills via privileged distillation to build a cognitive foundation;
- **Medium tasks**: standard RL to maximize task success;
- **Easy tasks**: diagnostic probing that penalizes shortcuts and enforces faithful task-specific skill utilization.

Experiments on ALFWorld and WebShop show that Skill0.5 outperforms both memory-based and skill-based RL baselines across in-distribution and out-of-distribution scenarios.

---

## Installation

### Python Environment

```bash
conda create -n skill05 python=3.12 -y
conda activate skill05

pip install vllm==0.11.0
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .

pip install openai
```

Log in to Weights & Biases (optional, for training logging):

```bash
export WANDB_API_KEY=your_key_here
```

### Environment Setup

#### ALFWorld

```bash
pip install alfworld
pip install gymnasium==0.29.1
pip install stable-baselines3==2.6.0

# Download PDDL & Game files and pre-trained MaskRCNN detector
alfworld-download -f
```

#### WebShop

```bash
cd agent_system/environments/env_package/webshop
./setup.sh -d all
```

---

## Data Preparation

### WebShop OOD Splits

Before training on WebShop, generate the OOD category splits:

```bash
python -m data_preprocess.preprocess_webshop_ood \
    --human_goals_path agent_system/environments/env_package/webshop/webshop/data/items_ins_v2_human.json \
    --items_path agent_system/environments/env_package/webshop/webshop/data/items_shuffle_human.json \
    --output data_preprocess/webshop_ood_splits.json
```

This produces `webshop_ood_splits.json` containing the ID/OOD goal indices used during training and evaluation.

### Embedding Model

The skill retrieval system uses [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) for semantic similarity. It will be downloaded automatically from HuggingFace on first use, or you can pre-download it:

```bash
huggingface-cli download Qwen/Qwen3-Embedding-0.6B --local-dir /path/to/Qwen3-Embedding-0.6B
```

Then pass the local path via `+env.skills_only_memory.embedding_model_path=/path/to/Qwen3-Embedding-0.6B` in the training script.

---

## Training

All training scripts are under `scripts/` and assume the repo root as working directory.

### ALFWorld OOD

```bash
export MODEL_PATH=/path/to/your/sft_checkpoint
export ALFWORLD_DATA=/path/to/alfworld_data

bash scripts/train_alfworld_ood.sh
```

### WebShop OOD

```bash
export MODEL_PATH=/path/to/your/sft_checkpoint
export WEBSHOP_DATA=/path/to/webshop_data

bash scripts/train_webshop_ood.sh
```


### Checkpoint Merging

After training, merge FSDP sharded checkpoints into a single HuggingFace model:

```bash
python scripts/model_merger.py --ckpt_path /path/to/global_step_N
```

Or batch-merge all steps in a directory:

```bash
python scripts/model_merger_dir.py --ckpt_dir /path/to/checkpoint_dir --start_step 5 --end_step 200
```

---

## Acknowledgement

This project builds on [SkillRL](https://github.com/aiming-lab/SkillRL), [verl](https://github.com/volcengine/verl), [verl-agent](https://github.com/langfengQ/verl-agent), [ALFWorld](https://github.com/alfworld/alfworld), and [WebShop](https://github.com/princeton-nlp/WebShop). We thank the authors of those projects.

"""
Batch convert FSDP checkpoints to HuggingFace format and clean up original FSDP files.

Usage:
    python scripts/model_merger_dir.py \
        --ckpt_dir /path/to/checkpoint_root \
        --start_step 10 \
        --end_step 100

This will:
1. Find all global_step_X directories where start_step <= X <= end_step
2. Convert each step's FSDP actor checkpoint to HuggingFace format,
   saved at global_step_X/hf_ckpt_X/
3. Delete the original FSDP files (model_world_size_*, optim_world_size_*, extra_state_*)
   for all steps EXCEPT end_step (to preserve the ability to resume training)
"""

import argparse
import glob
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from model_merger import FSDPModelMerger, ModelMergerConfig


def get_all_steps(ckpt_dir: str) -> list[int]:
    """Get all global_step numbers from checkpoint directory."""
    steps = []
    for name in os.listdir(ckpt_dir):
        match = re.match(r"global_step_(\d+)$", name)
        if match:
            steps.append(int(match.group(1)))
    return sorted(steps)


def get_steps_in_range(ckpt_dir: str, start_step: int, end_step: int) -> list[int]:
    """Get steps within [start_step, end_step] range."""
    all_steps = get_all_steps(ckpt_dir)
    return [s for s in all_steps if start_step <= s <= end_step]


def convert_step(ckpt_dir: str, step: int):
    """Convert a single step's FSDP checkpoint to HuggingFace format."""
    local_dir = os.path.join(ckpt_dir, f"global_step_{step}", "actor")
    step_target_dir = os.path.join(ckpt_dir, f"global_step_{step}", f"hf_ckpt_{step}")

    if not os.path.exists(local_dir):
        print(f"[WARNING] Actor directory not found: {local_dir}, skipping step {step}")
        return False

    # Check if already converted
    if os.path.exists(step_target_dir) and os.path.exists(
        os.path.join(step_target_dir, "config.json")
    ):
        print(f"[SKIP] Step {step} already converted at {step_target_dir}")
        return True

    os.makedirs(step_target_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Converting step {step}: {local_dir} -> {step_target_dir}")
    print(f"{'='*60}")

    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=local_dir,
        hf_model_config_path=local_dir,
        target_dir=step_target_dir,
    )

    merger = FSDPModelMerger(config)
    merger.merge_and_save()

    print(f"[DONE] Step {step} converted successfully")
    return True


def delete_fsdp_files(ckpt_dir: str, step: int):
    """Delete FSDP-specific files (model shards, optim shards, extra_state) for a step."""
    actor_dir = os.path.join(ckpt_dir, f"global_step_{step}", "actor")

    patterns = [
        "model_world_size_*_rank_*.pt",
        "optim_world_size_*_rank_*.pt",
        "extra_state_world_size_*_rank_*.pt",
    ]

    total_deleted = 0
    total_size = 0

    for pattern in patterns:
        files = glob.glob(os.path.join(actor_dir, pattern))
        for f in files:
            size = os.path.getsize(f)
            total_size += size
            os.remove(f)
            total_deleted += 1

    freed_gb = total_size / (1024**3)
    print(f"[DELETE] Step {step}: removed {total_deleted} files, freed {freed_gb:.1f} GB")


def main():
    parser = argparse.ArgumentParser(description="Batch convert FSDP checkpoints to HuggingFace format")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Root checkpoint directory containing global_step_X folders")
    parser.add_argument("--start_step", type=int, required=True, help="Start step (inclusive)")
    parser.add_argument("--end_step", type=int, required=True, help="End step (inclusive, FSDP files will be preserved for this step)")
    args = parser.parse_args()

    steps = get_steps_in_range(args.ckpt_dir, args.start_step, args.end_step)

    if not steps:
        print(f"No steps found in range [{args.start_step}, {args.end_step}]")
        print(f"Available steps: {get_all_steps(args.ckpt_dir)}")
        return

    print(f"Checkpoint directory: {args.ckpt_dir}")
    print(f"Steps to convert: {steps}")
    print(f"End step (FSDP preserved): {args.end_step}")
    print()

    # Phase 1: Convert all steps
    converted_steps = []
    for step in steps:
        success = convert_step(args.ckpt_dir, step)
        if success:
            converted_steps.append(step)

    # Phase 2: Delete FSDP files for all steps except end_step
    print(f"\n{'='*60}")
    print("Cleaning up FSDP files (preserving end_step)...")
    print(f"{'='*60}")

    steps_to_clean = [s for s in converted_steps if s != args.end_step]
    for step in steps_to_clean:
        delete_fsdp_files(args.ckpt_dir, step)

    if args.end_step in converted_steps:
        print(f"[KEEP] Step {args.end_step}: FSDP files preserved for potential resume training")

    print(f"\nAll done! Converted {len(converted_steps)} steps, cleaned {len(steps_to_clean)} steps.")


if __name__ == "__main__":
    main()


# python scripts/model_merger_dir.py --ckpt_dir /path/to/your/checkpoint_dir --start_step 5 --end_step 200
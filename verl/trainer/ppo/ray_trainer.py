# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
import random
from collections import defaultdict, deque
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager
from gigpo import core_gigpo

from agent_system.multi_turn_rollout import TrajectoryCollector, adjust_batch

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    GRPO_PASSK = "grpo_passk"
    GiGPO = 'gigpo'


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics

def apply_invalid_action_penalty(data: DataProto, invalid_action_penalty_coef=float):
    reward_tensor = data.batch['token_level_scores']
    if 'step_rewards' in data.batch.keys():
        step_rewards = data.batch['step_rewards']
    for i in range(len(data)):
        data_item = data[i]  # DataProtoItem

        prompt_ids = data_item.batch['prompts']

        prompt_length = prompt_ids.shape[-1]

        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

        action_valids = data_item.non_tensor_batch['is_action_valid'].astype(np.float32)
        action_invalids = torch.tensor(1 - action_valids, dtype=torch.float32, device=prompt_ids.device).squeeze(0)
        # invalid action penalty
        # assert reward_tensor[i, valid_response_length - 1] != 0.0, f'i={i}'
        reward_tensor[i, valid_response_length - 1] -= invalid_action_penalty_coef * action_invalids

        if 'step_rewards' in data.batch.keys():
            step_rewards[i] -= invalid_action_penalty_coef * action_invalids
    
    valid_action_ratio = np.mean(data.non_tensor_batch['is_action_valid'].astype(np.float32)).item()
    metrics = {'episode/valid_action_ratio': valid_action_ratio}
    return data, metrics

def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, step_advantage_w=1.0, gigpo_mode="mean_std_norm", gigpo_enable_similarity=False, gigpo_similarity_thresh=0.95, **kwargs):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if kwargs.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                kwargs.get("pf_ppo_reweight_method", "pow"),
                kwargs.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # TODO: test on more adv estimator type
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO

        # Check if contrastive mode provides context types for probe-based advantage
        contrastive_context_types = kwargs.get('contrastive_context_types', None)
        if contrastive_context_types is not None:
            use_decomposed = kwargs.get('use_decomposed_contrastive', False)
            if use_decomposed:
                omega = kwargs.get('contrastive_omega', 1.0)
                ema_delta = kwargs.get('ema_delta', None)
                adv2_clip = kwargs.get('adv2_clip', 3.0)
                advantages, returns = core_algos.compute_grpo_decomposed_contrastive_advantage(
                    token_level_rewards=data.batch["token_level_rewards"],
                    response_mask=grpo_calculation_mask,
                    index=data.non_tensor_batch["uid"],
                    traj_index=data.non_tensor_batch['traj_uid'],
                    contrastive_context_types=contrastive_context_types,
                    omega=omega,
                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                    ema_delta=ema_delta,
                    adv2_clip=adv2_clip,
                )
            else:
                advantages, returns = core_algos.compute_grpo_contrastive_outcome_advantage(
                    token_level_rewards=data.batch["token_level_rewards"],
                    response_mask=grpo_calculation_mask,
                    index=data.non_tensor_batch["uid"],
                    traj_index=data.non_tensor_batch['traj_uid'],
                    contrastive_context_types=contrastive_context_types,
                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                )
        else:
            # Call compute_grpo_outcome_advantage with parameters matching its definition
            advantages, returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=grpo_calculation_mask,
                index=data.non_tensor_batch["uid"],
                traj_index=data.non_tensor_batch['traj_uid'],
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
        advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GiGPO:
        advantages, returns = core_gigpo.compute_gigpo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'], # for episode group reward computing
            step_rewards=data.batch['step_rewards'], # for step group reward computing
            response_mask=data.batch['response_mask'],
            anchor_obs=data.non_tensor_batch['anchor_obs'],
            index=data.non_tensor_batch['uid'],
            traj_index=data.non_tensor_batch['traj_uid'],
            step_advantage_w=step_advantage_w,
            mode=gigpo_mode,
            enable_similarity=gigpo_enable_similarity,
            similarity_thresh=gigpo_similarity_thresh,
            )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    """Context manager for timing code execution.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        val_dataset_ood: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
        traj_collector: TrajectoryCollector = None,
        envs=None,
        val_envs=None,
        val_envs_ood=None,
    ):
        """Initialize distributed PPO trainer with Ray backend."""

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.envs = envs
        self.val_envs = val_envs
        self.val_envs_ood = val_envs_ood
        self.traj_collector = traj_collector

        # Sliding window for adaptive routing (ours mode)
        ours_cfg = config.env.get('ours', {})
        self._routing_window_size = ours_cfg.get('window_size', 5)
        self._routing_window = deque(maxlen=self._routing_window_size)  # stores per-step non-zero mean
        self._routing_threshold = None  # Will be computed from window
        self._ema_delta = None  # EMA of easy-task delta (skill - noskill), initialized on first easy step
        # Sliding window option for delta baseline (alternative to EMA)
        utilize_cfg = config.env.get('utilize', {})
        self._delta_baseline_mode = utilize_cfg.get('delta_baseline_mode', 'ema')  # 'ema' or 'window'
        delta_window_size = utilize_cfg.get('delta_window_size', 5)
        self._delta_window = deque(maxlen=delta_window_size)

        # Dual distillation state: sliding windows for adaptive scheduling
        dual_cfg = config.env.get('dual_distill', {})
        dual_window_size = dual_cfg.get('window_size', 5)
        self._dual_intern_window = deque(maxlen=dual_window_size)
        self._dual_robust_window = deque(maxlen=dual_window_size)
        self._dual_intern_count = 0   # cumulative guided branch activations
        self._dual_robust_count = 0   # cumulative noskill branch activations

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get('lora_rank', 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.GiGPO
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler,
                                val_dataset_ood=val_dataset_ood)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        effective_rollout_n = config.actor_rollout_ref.rollout.n
        real_train_batch_size = config.data.train_batch_size * effective_rollout_n
        assert real_train_batch_size % n_gpus == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler,
                           val_dataset_ood=None):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        # Val ID dataloader
        # In OOD mode with env-generated tasks (e.g. ALFWorld), val_dataset is a
        # Subset placeholder so batch_size = len(dataset).
        # In OOD mode with data-driven tasks (e.g. Search), val_dataset has real
        # rows and should use config.data.val_batch_size to iterate in batches.
        if val_dataset_ood is not None:
            # Use config batch size if explicitly set; otherwise fall back to
            # len(dataset) for backward compat with ALFWorld-style placeholders.
            val_batch_size = self.config.data.get("val_batch_size", None)
            if val_batch_size is None:
                val_batch_size = len(self.val_dataset)
        else:
            val_batch_size = self.config.data.val_batch_size
            if val_batch_size is None:
                val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        # Val OOD dataloader (if provided)
        self.val_dataloader_ood = None
        if val_dataset_ood is not None:
            val_ood_batch_size = self.config.data.get("val_ood_batch_size", None)
            if val_ood_batch_size is None:
                # Fall back to val_batch_size, then to len(dataset)
                val_ood_batch_size = self.config.data.get("val_batch_size", None)
            if val_ood_batch_size is None:
                val_ood_batch_size = len(val_dataset_ood)
            self.val_dataloader_ood = StatefulDataLoader(
                dataset=val_dataset_ood,
                batch_size=val_ood_batch_size,
                num_workers=self.config.data.get("dataloader_num_workers", 8),
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn,
            )
            print(f"Val OOD dataloader: batch_size={val_ood_batch_size}, batches={len(self.val_dataloader_ood)}")

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path,
                          traj_uids=None, uids=None, extra_meta=None):
        """Dump rollout samples as JSONL, grouped by trajectory.

        When traj_uids is provided, steps belonging to the same trajectory are
        merged into a single JSON entry (matching val dump format).  Otherwise
        falls back to one-step-per-line (legacy behaviour).

        Args:
            extra_meta: Optional dict of extra fields to include in each trajectory entry.
        """
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)

        if traj_uids is not None and len(traj_uids) == n:
            # ── Grouped-by-trajectory format (aligned with val dump) ──
            from collections import OrderedDict
            traj_groups = OrderedDict()  # traj_uid -> {steps, scores, uid, extras}
            for i in range(n):
                uid_str = str(traj_uids[i])
                if uid_str not in traj_groups:
                    traj_groups[uid_str] = {
                        "steps": [],
                        "scores": [],
                        "uid": str(uids[i]) if uids is not None else None,
                    }
                traj_groups[uid_str]["steps"].append({"input": inputs[i], "output": outputs[i]})
                traj_groups[uid_str]["scores"].append(scores[i])

            with open(filename, "w") as f:
                for traj_uid_str, traj_data in traj_groups.items():
                    traj_score = sum(traj_data["scores"])  # total reward for this trajectory
                    entry = {
                        "traj_uid": traj_uid_str,
                        "uid": traj_data["uid"],
                        "score": traj_score,
                        "num_steps": len(traj_data["steps"]),
                        "step_scores": traj_data["scores"],
                        "steps": traj_data["steps"],
                        "global_step": self.global_steps,
                    }
                    if extra_meta:
                        entry.update(extra_meta)
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            print(f"Dumped {len(traj_groups)} trajectories to {filename}")
        else:
            # ── Legacy flat format (one step per line) ──
            base_data = {
                "input": inputs,
                "output": outputs,
                "score": scores,
                "step": [self.global_steps] * n,
            }

            for k, v in reward_extra_infos_dict.items():
                if len(v) == n:
                    base_data[k] = v

            with open(filename, "w") as f:
                for i in range(n):
                    entry = {k: v[i] for k, v in base_data.items()}
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            print(f"Dumped {n} steps (flat) to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []
        tool_calling_list = []
        traj_uid_list = []
        success_rate_dict = {}

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        # Per-step data for trajectory dump
        all_step_inputs = []
        all_step_outputs = []
        all_step_traj_uids = []

        # Reset eval cursor for sequential full-coverage evaluation
        if hasattr(self.val_envs, 'reset_eval_cursor'):
            self.val_envs.reset_eval_cursor()

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "env_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("env_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # # pad to be divisible by dp_size
            # test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            # test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

            # # unpad
            # test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            ################ agent-environment loop ###############
            test_output_gen_batch = self.traj_collector.multi_turn_loop(
                                                    gen_batch=test_gen_batch,
                                                    actor_rollout_wg=self.actor_rollout_wg,
                                                    envs=self.val_envs,
                                                    is_train=False,
                                                    )
            print('validation generation end')
            del test_batch
            test_batch = test_output_gen_batch
            # Store generated outputs (per-step, flattened across all steps)
            step_input_ids = test_output_gen_batch.batch["input_ids"]
            step_input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in step_input_ids]
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)
            step_traj_uids = test_output_gen_batch.non_tensor_batch['traj_uid']

            # Collect per-step data for trajectory dump
            all_step_inputs.extend(step_input_texts)
            all_step_outputs.extend(output_texts)
            all_step_traj_uids.extend(step_traj_uids)

            # test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
            tool_calling_list.append(test_output_gen_batch.non_tensor_batch['tool_callings'])
            traj_uid_list.append(test_output_gen_batch.non_tensor_batch['traj_uid'])
            # success rate
            for k in test_batch.non_tensor_batch.keys():
                if 'success_rate' in k:
                    if k not in success_rate_dict:
                        success_rate_dict[k] = []
                    success_rate_dict[k].append(test_batch.non_tensor_batch[k][0])
                    # all success_rate should be the same
                    for i in range(1, len(test_batch.non_tensor_batch[k])):
                        assert test_batch.non_tensor_batch[k][0] == test_batch.non_tensor_batch[k][i], f'not all success_rate are the same, 0: {test_batch.non_tensor_batch[k][0]}, {i}: {test_batch.non_tensor_batch[k][i]}'

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # Dump full multi-turn trajectories grouped by traj_uid
        val_dump_path = self.config.trainer.get("val_dump_path", None)
        if val_dump_path and all_step_traj_uids:
            os.makedirs(val_dump_path, exist_ok=True)
            filename = os.path.join(val_dump_path, f"{self.global_steps}.jsonl")
            # Group steps by traj_uid, preserving order
            from collections import OrderedDict
            traj_groups = OrderedDict()
            for inp, out, uid in zip(all_step_inputs, all_step_outputs, all_step_traj_uids):
                uid_str = str(uid)
                if uid_str not in traj_groups:
                    traj_groups[uid_str] = {"steps": []}
                traj_groups[uid_str]["steps"].append({"input": inp, "output": out})
            # Compute per-trajectory score from sample_scores (one score per step, same within a traj)
            traj_uids_flat = np.concatenate(traj_uid_list, axis=0)
            scores_flat = np.array(sample_scores)
            uid_to_score = {}
            for uid, score in zip(traj_uids_flat, scores_flat):
                uid_str = str(uid)
                if uid_str not in uid_to_score:
                    uid_to_score[uid_str] = score
            with open(filename, "w") as f:
                for uid_str, traj_data in traj_groups.items():
                    entry = {
                        "traj_uid": uid_str,
                        "score": uid_to_score.get(uid_str, 0.0),
                        "num_steps": len(traj_data["steps"]),
                        "steps": traj_data["steps"],
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"Dumped {len(traj_groups)} trajectories to {filename}")

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        tool_callings = np.concatenate(tool_calling_list, axis=0)
        traj_uids = np.concatenate(traj_uid_list, axis=0)
        success_rate = {k: np.mean(v) for k, v in success_rate_dict.items()}

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        # evaluate tool call based on data source
        # the values in tool_callings represent the tool call count for each trajectory; however, since the batch is expanded by step, we only need to take one value for each unique trajectories.
        data_source_tool_calling = {}
        unique_traj_uid, unique_idx = np.unique(traj_uids, return_index=True)
        unique_data_sources = data_sources[unique_idx]
        unique_tool_callings = tool_callings[unique_idx]

        for i in range(unique_tool_callings.shape[0]):
            data_source = unique_data_sources[i]
            if data_source not in data_source_tool_calling:
                data_source_tool_calling[data_source] = []
            data_source_tool_calling[data_source].append(unique_tool_callings[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/{data_source}/test_score'] = np.mean(rewards)

        for data_source, tool_calls in data_source_tool_calling.items():
            metric_dict[f'val/{data_source}/tool_call_count/mean'] = np.mean(tool_calls)
            # metric_dict[f'val/{data_source}/tool_call_count/max'] = np.max(tool_calls)
            # metric_dict[f'val/{data_source}/tool_call_count/min'] = np.min(tool_calls)

        for k, v in success_rate.items():
            metric_dict[f'val/{k}'] = v
        metric_dict['val/num_trajs'] = len(unique_traj_uid)

        # === Skill Bank 动态更新 ===
        if self.config.env.get('skills_only_memory', {}).get('enable_dynamic_update', False):
            self._update_skills_from_validation(
                sample_inputs=sample_inputs,
                sample_outputs=sample_outputs,
                sample_scores=sample_scores,
                success_rate=success_rate,
            )

        return metric_dict

    def _validate_ood(self):
        """Run validation on OOD environments with 'val_ood/' metric prefix.

        Mirrors _validate() logic but uses self.val_envs_ood and does NOT
        trigger skill dynamic update.
        """
        assert self.val_dataloader_ood is not None, \
            "_validate_ood requires val_dataloader_ood (pass val_dataset_ood to RayPPOTrainer)"
        reward_tensor_lst = []
        data_source_lst = []
        tool_calling_list = []
        traj_uid_list = []
        success_rate_dict = {}

        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        all_step_inputs = []
        all_step_outputs = []
        all_step_traj_uids = []

        # Reset eval cursor for sequential full-coverage evaluation
        if hasattr(self.val_envs_ood, 'reset_eval_cursor'):
            self.val_envs_ood.reset_eval_cursor()

        for test_data in self.val_dataloader_ood:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            input_ids = test_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "env_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("env_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }

            # Use OOD environments
            test_output_gen_batch = self.traj_collector.multi_turn_loop(
                gen_batch=test_gen_batch,
                actor_rollout_wg=self.actor_rollout_wg,
                envs=self.val_envs_ood,
                is_train=False,
            )
            print('OOD validation generation end')
            del test_batch
            test_batch = test_output_gen_batch

            step_input_ids = test_output_gen_batch.batch["input_ids"]
            step_input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in step_input_ids]
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)
            step_traj_uids = test_output_gen_batch.non_tensor_batch['traj_uid']

            all_step_inputs.extend(step_input_texts)
            all_step_outputs.extend(output_texts)
            all_step_traj_uids.extend(step_traj_uids)

            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", np.array(["ood"] * reward_tensor.shape[0])))
            tool_calling_list.append(test_batch.non_tensor_batch.get("tool_callings", np.zeros(reward_tensor.shape[0])))
            traj_uid_list.append(test_batch.non_tensor_batch.get("traj_uid", np.arange(reward_tensor.shape[0])))

            for k in test_batch.non_tensor_batch.keys():
                if 'success_rate' in k:
                    vals = test_batch.non_tensor_batch[k]
                    if k not in success_rate_dict:
                        success_rate_dict[k] = []
                    for i in range(len(vals)):
                        if i == 0 or test_batch.non_tensor_batch.get('traj_uid', [None]*len(vals))[i] != test_batch.non_tensor_batch.get('traj_uid', [None]*len(vals))[i-1]:
                            success_rate_dict[k].append(vals[i])

        # Dump OOD trajectories
        val_dump_path = self.config.trainer.get("val_dump_path", None)
        if val_dump_path and all_step_traj_uids:
            ood_dump_path = os.path.join(val_dump_path, "ood")
            os.makedirs(ood_dump_path, exist_ok=True)
            filename = os.path.join(ood_dump_path, f"{self.global_steps}.jsonl")
            from collections import OrderedDict
            traj_groups = OrderedDict()
            for inp, out, uid in zip(all_step_inputs, all_step_outputs, all_step_traj_uids):
                uid_str = str(uid)
                if uid_str not in traj_groups:
                    traj_groups[uid_str] = {"steps": []}
                traj_groups[uid_str]["steps"].append({"input": inp, "output": out})
            traj_uids_flat = np.concatenate(traj_uid_list, axis=0)
            scores_flat = np.array(sample_scores)
            uid_to_score = {}
            for uid, score in zip(traj_uids_flat, scores_flat):
                uid_str = str(uid)
                if uid_str not in uid_to_score:
                    uid_to_score[uid_str] = score
            with open(filename, "w") as f:
                for uid_str, traj_data in traj_groups.items():
                    entry = {
                        "traj_uid": uid_str,
                        "score": uid_to_score.get(uid_str, 0.0),
                        "num_steps": len(traj_data["steps"]),
                        "steps": traj_data["steps"],
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"Dumped {len(traj_groups)} OOD trajectories to {filename}")

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()
        data_sources = np.concatenate(data_source_lst, axis=0)
        tool_callings = np.concatenate(tool_calling_list, axis=0)
        traj_uids = np.concatenate(traj_uid_list, axis=0)
        success_rate = {k: np.mean(v) for k, v in success_rate_dict.items()}

        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        data_source_tool_calling = {}
        unique_traj_uid, unique_idx = np.unique(traj_uids, return_index=True)
        unique_data_sources = data_sources[unique_idx]
        unique_tool_callings = tool_callings[unique_idx]
        for i in range(unique_tool_callings.shape[0]):
            data_source = unique_data_sources[i]
            if data_source not in data_source_tool_calling:
                data_source_tool_calling[data_source] = []
            data_source_tool_calling[data_source].append(unique_tool_callings[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val_ood/{data_source}/test_score'] = np.mean(rewards)
        for data_source, tool_calls in data_source_tool_calling.items():
            metric_dict[f'val_ood/{data_source}/tool_call_count/mean'] = np.mean(tool_calls)
        for k, v in success_rate.items():
            metric_dict[f'val_ood/{k}'] = v
        metric_dict['val_ood/num_trajs'] = len(unique_traj_uid)

        return metric_dict

    def _update_skills_from_validation(
        self,
        sample_inputs: list,
        sample_outputs: list,
        sample_scores: list,
        success_rate: dict,
    ):
        """
        根据 validation 结果更新 skill bank。

        仅在特定任务类型成功率低于阈值时触发更新。
        """
        update_config = self.config.env.skills_only_memory
        threshold = update_config.get('update_threshold', 0.5)

        # 检查是否需要更新（某个任务类型成功率低于阈值）
        needs_update = False
        low_success_tasks = []
        for task_key, rate in success_rate.items():
            if rate < threshold:
                needs_update = True
                # 从 key 提取 task_type (e.g., "pick_and_place_success_rate" -> "pick_and_place")
                task_type = task_key.replace('_success_rate', '')
                low_success_tasks.append(task_type)

        if not needs_update:
            print(f"[SkillUpdate] All task success rates above {threshold}, skipping update")
            return

        print(f"[SkillUpdate] Low success tasks: {low_success_tasks}, triggering skill update...")

        # 收集失败 trajectories
        failed_trajectories = self._collect_failed_trajectories(
            sample_inputs, sample_outputs, sample_scores
        )

        if not failed_trajectories:
            print("[SkillUpdate] No failed trajectories found")
            return

        # 初始化 SkillUpdater (lazy init, 使用 Azure OpenAI o3)
        if not hasattr(self, 'skill_updater'):
            from agent_system.memory.skill_updater import SkillUpdater
            self.skill_updater = SkillUpdater(
                max_new_skills_per_update=update_config.get('max_new_skills', 3),
            )

        # 获取当前 skills —— 从 train envs 读取，因为新 skill 会加到 train envs，
        # 如果从 val_envs 读，_next_dyn_index 看不到已有的 dyn_* ID，导致 ID 冲突。
        train_memory = self.envs.retrieval_memory if (
            hasattr(self, 'envs') and hasattr(self.envs, 'retrieval_memory')
        ) else None
        if train_memory is None:
            print("[SkillUpdate] No retrieval_memory found in training envs")
            return

        # 分析失败并生成新 skills
        print(f"[SkillUpdate] Analyzing {len(failed_trajectories)} failed trajectories ...")
        new_skills = self.skill_updater.analyze_failures(
            failed_trajectories=failed_trajectories,
            current_skills=train_memory.skills,
        )

        if new_skills:
            # Add to both training and validation envs so that new skills
            # are used in subsequent training rollouts AND validation rollouts.
            added_train = train_memory.add_skills(new_skills, category='general')
            print(f"[SkillUpdate] Added {added_train} new skills to training envs")

            added_val = 0
            if hasattr(self, 'val_envs') and hasattr(self.val_envs, 'retrieval_memory') and self.val_envs.retrieval_memory:
                added_val = self.val_envs.retrieval_memory.add_skills(new_skills, category='general')
                print(f"[SkillUpdate] Added {added_val} new skills to validation envs")

            # Save updated skill bank to disk.
            if added_train > 0:
                save_dir = self.config.trainer.get('default_local_dir', './outputs')
                save_path = os.path.join(save_dir, f'updated_skills_step{self.global_steps}.json')
                train_memory.save_skills(save_path)
                print(f"[SkillUpdate] Saved updated skill bank to {save_path}")
            else:
                print("[SkillUpdate] All generated skills were duplicates, skipping save")
        else:
            print("[SkillUpdate] No new skills generated")

    def _collect_failed_trajectories(
        self,
        inputs: list,
        outputs: list,
        scores: list,
    ) -> list:
        """收集失败的 trajectories 用于分析"""
        failed = []
        for inp, out, score in zip(inputs, outputs, scores):
            if score <= 0:  # 失败的 trajectory
                task_type = self._detect_task_type_from_input(inp)
                task_desc = self._extract_task_description(inp)
                trajectory = self._parse_conversation_to_steps(inp, out)
                failed.append({
                    'task': task_desc,
                    'trajectory': trajectory,
                    'task_type': task_type,
                })
        return failed[:10]  # 限制数量，避免 prompt 过长

    def _extract_task_description(self, inp: str) -> str:
        """Extract the task description from a full conversation prompt."""
        import re
        # Common patterns used in ALFWorld, WebShop, OpenClaw, etc.
        patterns = [
            r'(?:Your task is to|Task:|task is to|you need to)[:\s]+(.*?)(?:\n|$)',
            r'(?:goal|objective)[:\s]+(.*?)(?:\n|$)',
        ]
        for pat in patterns:
            m = re.search(pat, inp, re.IGNORECASE)
            if m:
                return m.group(1).strip()[:1000]
        # Fallback: first user turn (skip system prompt)
        for marker in ('<|im_start|>user\n', '\nHuman: ', '\nUser: '):
            idx = inp.find(marker)
            if idx >= 0:
                start = idx + len(marker)
                return inp[start:start + 1000]
        return inp[:1000]

    def _parse_conversation_to_steps(self, inp: str, out: str) -> list:
        """
        Parse a full decoded conversation into a list of trajectory steps.

        Each step is ``{'action': str, 'observation': str}`` where
        ``observation`` is the environment feedback (user/tool turn) and
        ``action`` is the agent response (assistant turn).

        Falls back to treating the whole ``inp`` as the initial context when
        no structured turn markers are found.
        """
        import re
        steps = []

        # --- ChatML / Qwen format -------------------------------------------
        user_turns = re.findall(
            r'<\|im_start\|>user\n(.*?)<\|im_end\|>', inp, re.DOTALL
        )
        asst_turns = re.findall(
            r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', inp, re.DOTALL
        )
        if user_turns and asst_turns:
            for obs, act in zip(user_turns, asst_turns):
                steps.append({
                    'action': act.strip()[:1500],
                    'observation': obs.strip()[:800],
                })
            # Final (failed) action has no follow-up observation
            steps.append({'action': out[:2000], 'observation': ''})
            return steps

        # --- Human / Assistant format ----------------------------------------
        user_turns = re.findall(
            r'(?:Human|User):\s*(.*?)(?=(?:Human|User|Assistant):|$)',
            inp, re.DOTALL | re.IGNORECASE,
        )
        asst_turns = re.findall(
            r'Assistant:\s*(.*?)(?=(?:Human|User|Assistant):|$)',
            inp, re.DOTALL | re.IGNORECASE,
        )
        if user_turns and asst_turns:
            for obs, act in zip(user_turns, asst_turns):
                steps.append({
                    'action': act.strip()[:1500],
                    'observation': obs.strip()[:800],
                })
            steps.append({'action': out[:2000], 'observation': ''})
            return steps

        # --- Fallback: treat full inp as initial context ---------------------
        steps.append({'action': '', 'observation': inp[:3000]})
        steps.append({'action': out[:2000], 'observation': ''})
        return steps

    def _detect_task_type_from_input(self, inp: str) -> str:
        """从输入中检测任务类型"""
        inp_lower = inp.lower()
        if 'clean' in inp_lower:
            return 'clean'
        elif 'heat' in inp_lower:
            return 'heat'
        elif 'cool' in inp_lower:
            return 'cool'
        elif 'look at' in inp_lower and ('lamp' in inp_lower or 'light' in inp_lower):
            return 'look_at_obj_in_light'
        elif 'examine' in inp_lower:
            return 'examine'
        else:
            return 'pick_and_place'

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def _utilize_step(self, gen_batch, timing_raw, metrics):
        """Utilize-only ablation: contrastive GRPO (skill vs no-skill).

        Phase 1: Skill rollout (all envs get specific skills, context_type=top_k)
        Phase 2: No-skill probe (all envs get no specific skills, context_type=no_skill)
                 Same tasks replayed via reset_info, same uids via uid_base.
        Returns the batch and reward_extra_infos_dict.
        """
        rollout_n = self.config.actor_rollout_ref.rollout.n

        # ── Phase 1: Skill rollout ──
        with _timer("gen_skill", timing_raw):
            skill_output = self.traj_collector.multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=self.actor_rollout_wg,
                envs=self.envs,
                is_train=True,
            )

        # Capture reset_info and uid_base from Phase 1
        reset_info = self.envs.get_last_reset_info()

        # Extract unique uids (one per task) from Phase 1
        # skill_output has batch_size = train_data_size * rollout_n steps,
        # uid is repeated across steps of the same trajectory.
        # We need the unique uid per task (one uid shared by rollout_n trajectories).
        phase1_uids = skill_output.non_tensor_batch['uid']
        seen = set()
        uid_base = []
        for uid in phase1_uids:
            if uid not in seen:
                seen.add(uid)
                uid_base.append(uid)
        uid_base = np.array(uid_base, dtype=object)

        print(f"[Utilize] Phase 1 done: {len(skill_output.batch['input_ids'])} samples, "
              f"{len(uid_base)} tasks")

        # Dump Phase 1 (skill) trajectories immediately (before balance_batch reorders)
        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
        if rollout_data_dir:
            skill_inputs = self.tokenizer.batch_decode(skill_output.batch["prompts"], skip_special_tokens=True)
            skill_outputs_text = self.tokenizer.batch_decode(skill_output.batch["responses"], skip_special_tokens=True)
            skill_scores = [float(x) for x in skill_output.non_tensor_batch['episode_rewards']]
            self._dump_generations(
                inputs=skill_inputs, outputs=skill_outputs_text, scores=skill_scores,
                reward_extra_infos_dict={}, dump_path=rollout_data_dir,
                traj_uids=skill_output.non_tensor_batch.get('traj_uid', None),
                uids=skill_output.non_tensor_batch.get('uid', None),
            )

        # ── Phase 2: No-skill probe ──
        with _timer("gen_noskill", timing_raw):
            self.envs.set_contrastive_probe(True)
            noskill_output = self.traj_collector.multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=self.actor_rollout_wg,
                envs=self.envs,
                is_train=True,
                reset_info=reset_info,
                uid_base=uid_base,
            )
            self.envs.set_contrastive_probe(False)

        print(f"[Utilize] Phase 2 done: {len(noskill_output.batch['input_ids'])} samples")

        # Dump Phase 2 (noskill) trajectories
        if rollout_data_dir:
            noskill_dump_path = os.path.join(rollout_data_dir, "noskill")
            noskill_inputs = self.tokenizer.batch_decode(noskill_output.batch["prompts"], skip_special_tokens=True)
            noskill_outputs_text = self.tokenizer.batch_decode(noskill_output.batch["responses"], skip_special_tokens=True)
            noskill_scores = [float(x) for x in noskill_output.non_tensor_batch['episode_rewards']]
            self._dump_generations(
                inputs=noskill_inputs, outputs=noskill_outputs_text, scores=noskill_scores,
                reward_extra_infos_dict={}, dump_path=noskill_dump_path,
                traj_uids=noskill_output.non_tensor_batch.get('traj_uid', None),
                uids=noskill_output.non_tensor_batch.get('uid', None),
            )

        # ── Merge ──
        merged_batch = DataProto.concat([skill_output, noskill_output])

        print(f"[Utilize] Merged: {len(merged_batch.batch['input_ids'])} total samples")

        # ── Reward & Advantage on merged batch (need noskill rewards as baseline) ──
        merged_batch = adjust_batch(self.config, merged_batch)
        merged_batch.batch["response_mask"] = compute_response_mask(merged_batch)

        # Reward (on merged batch — need noskill reward values for advantage baseline)
        with _timer("reward", timing_raw):
            reward_tensor, reward_extra_infos_dict = compute_reward(merged_batch, self.reward_fn)

        # Advantage computation on merged batch
        with _timer("adv", timing_raw):
            merged_batch.batch["token_level_scores"] = reward_tensor

            # Apply invalid action penalty BEFORE advantage computation so that
            # both noskill_mean and skill scores include format penalties.
            # Metrics (valid_action_ratio) will be computed later on phase1 only.
            if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                merged_batch, _ = apply_invalid_action_penalty(
                    merged_batch,
                    invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                )

            print(f"{list(reward_extra_infos_dict.keys())=}")
            if reward_extra_infos_dict:
                merged_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

            # Contrastive metrics (on merged batch — these are already split by context_type)
            contrastive_context_types = merged_batch.non_tensor_batch.get('context_type', None)
            if contrastive_context_types is not None:
                episode_rewards_arr = merged_batch.non_tensor_batch.get('episode_rewards', None)
                bs = len(merged_batch.batch['token_level_scores'])
                traj_uids = merged_batch.non_tensor_batch.get('traj_uid', None)
                seen_trajs = {}
                for i in range(bs):
                    uid = traj_uids[i] if traj_uids is not None else i
                    if uid not in seen_trajs:
                        is_succ = float(episode_rewards_arr[i]) > 0 if episode_rewards_arr is not None else False
                        seen_trajs[uid] = (contrastive_context_types[i], is_succ)
                n_top_k_trajs = sum(1 for ct, _ in seen_trajs.values() if ct == 'top_k')
                n_top_k_success = sum(1 for ct, s in seen_trajs.values() if ct == 'top_k' and s)
                n_no_skill_trajs = sum(1 for ct, _ in seen_trajs.values() if ct == 'no_skill')
                n_no_skill_success = sum(1 for ct, s in seen_trajs.values() if ct == 'no_skill' and s)
                metrics['utilize/skill_success_rate'] = n_top_k_success / n_top_k_trajs if n_top_k_trajs > 0 else 0.0
                metrics['utilize/noskill_success_rate'] = n_no_skill_success / n_no_skill_trajs if n_no_skill_trajs > 0 else 0.0
                metrics['utilize/n_skill'] = n_top_k_trajs
                metrics['utilize/n_noskill'] = n_no_skill_trajs
                metrics['utilize/n_skill_success'] = n_top_k_success
                metrics['utilize/n_noskill_success'] = n_no_skill_success

                # Per-task delta for utilize mode
                group_uids_merged = merged_batch.non_tensor_batch.get('uid', None)
                if group_uids_merged is not None:
                    task_skill_rewards = defaultdict(list)
                    task_noskill_rewards = defaultdict(list)
                    for i, uid in enumerate(traj_uids):
                        if uid in seen_trajs:
                            ct, is_succ = seen_trajs[uid]
                            g_uid = group_uids_merged[i]
                            if ct == 'top_k':
                                task_skill_rewards[g_uid].append(float(is_succ))
                            elif ct == 'no_skill':
                                task_noskill_rewards[g_uid].append(float(is_succ))
                    task_deltas = []
                    for g_uid in task_skill_rewards:
                        skill_pr = np.mean(task_skill_rewards[g_uid])
                        noskill_pr = np.mean(task_noskill_rewards[g_uid]) if g_uid in task_noskill_rewards else 0.0
                        task_deltas.append(skill_pr - noskill_pr)
                    if task_deltas:
                        self._current_step_delta = float(np.mean(task_deltas))
                        metrics['utilize/delta'] = self._current_step_delta

            # token_level_rewards (no KL in reward for contrastive)
            if self.config.algorithm.use_kl_in_reward:
                merged_batch, kl_metrics = apply_kl_penalty(merged_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
            else:
                merged_batch.batch["token_level_rewards"] = merged_batch.batch["token_level_scores"]

            # Compute advantage (contrastive: noskill_mean as baseline)
            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
            utilize_cfg = self.config.env.get('utilize', {})
            use_decomposed = utilize_cfg.get('decomposed_contrastive', False)
            contrastive_omega = utilize_cfg.get('omega', 1.0)
            use_ema_delta = utilize_cfg.get('use_ema_delta', False)
            adv2_clip = utilize_cfg.get('adv2_clip', 3.0)
            effective_rollout_n = rollout_n * 2  # merged: skill + noskill per group
            # Compute delta baseline value based on mode
            if use_ema_delta:
                if self._delta_baseline_mode == 'window' and len(self._delta_window) > 0:
                    ema_delta_value = float(np.mean(list(self._delta_window)))
                elif self._delta_baseline_mode == 'ema' and self._ema_delta is not None:
                    ema_delta_value = self._ema_delta
                else:
                    ema_delta_value = None
            else:
                ema_delta_value = None
            merged_batch = compute_advantage(
                merged_batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=effective_rollout_n,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                use_pf_ppo=self.config.algorithm.use_pf_ppo,
                pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                gigpo_mode=self.config.algorithm.gigpo.mode,
                gigpo_enable_similarity=self.config.algorithm.gigpo.enable_similarity,
                gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                contrastive_context_types=contrastive_context_types,
                use_decomposed_contrastive=use_decomposed,
                contrastive_omega=contrastive_omega,
                ema_delta=ema_delta_value,
                adv2_clip=adv2_clip,
            )

        # Update delta baseline after advantage computation (use old value for adv2, then update)
        if hasattr(self, '_current_step_delta') and self._current_step_delta is not None:
            utilize_cfg = self.config.env.get('utilize', {})
            ema_delta_alpha = utilize_cfg.get('ema_delta_alpha', 0.1)
            if self._ema_delta is None:
                self._ema_delta = self._current_step_delta
            else:
                self._ema_delta = ema_delta_alpha * self._current_step_delta + (1 - ema_delta_alpha) * self._ema_delta
            self._delta_window.append(self._current_step_delta)
            # Log the effective baseline value
            if self._delta_baseline_mode == 'window' and len(self._delta_window) > 0:
                metrics['utilize/delta_baseline'] = float(np.mean(list(self._delta_window)))
            else:
                metrics['utilize/delta_baseline'] = self._ema_delta
            metrics['utilize/ema_delta'] = self._ema_delta
            self._current_step_delta = None

        # ── Filter: keep only Phase 1 (top_k) samples for training ──
        phase1_mask = np.array([ct == 'top_k' for ct in merged_batch.non_tensor_batch['context_type']])
        phase1_idxs = np.where(phase1_mask)[0]
        batch = merged_batch.select_idxs(phase1_idxs)
        batch.meta_info = merged_batch.meta_info.copy()
        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        # Pad to be divisible by n_gpus. Padding samples have advantages=0,
        # response_mask=0, and loss_mask=0 so they contribute zero loss and zero gradient.
        world_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        bs_phase1 = len(batch)
        remainder = bs_phase1 % world_size
        if remainder != 0:
            to_add = world_size - remainder
            dup_indices = np.random.choice(bs_phase1, to_add, replace=(to_add > bs_phase1))
            dup_proto = batch.select_idxs(dup_indices)
            # Zero out loss-related fields so padding has no gradient effect
            dup_proto.batch["advantages"] = torch.zeros_like(dup_proto.batch["advantages"])
            dup_proto.batch["response_mask"] = torch.zeros_like(dup_proto.batch["response_mask"])
            if "loss_mask" in dup_proto.batch:
                dup_proto.batch["loss_mask"] = torch.zeros_like(dup_proto.batch["loss_mask"])
            batch = DataProto.concat([batch, dup_proto])
            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
            print(f"[Utilize] Phase 1 padded: {bs_phase1} -> {len(batch)} (divisor={world_size})")

        print(f"[Utilize] Phase 1 only for training: {len(batch.batch['input_ids'])} samples")

        # ── Old log probs (Phase 1 only) ──
        with _timer("old_log_prob", timing_raw):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            # Compute entropy metric on real samples only (exclude padding)
            entropys_real = entropys[:bs_phase1]
            response_masks_real = batch.batch["response_mask"][:bs_phase1]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_loss = agg_loss(loss_mat=entropys_real, loss_mask=response_masks_real, loss_agg_mode=loss_agg_mode)
            metrics.update({"actor/entropy_loss_batch-grpo": entropy_loss.detach().item()})
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

        # ── Reference log probs (Phase 1 only) ──
        if self.use_reference_policy:
            with _timer("ref", timing_raw):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        # ── Invalid action penalty metrics (Phase 1 only, exclude padding) ──
        # NOTE: The actual penalty was already applied on merged_batch before advantage
        # computation (so both noskill_mean and skill scores include format penalties).
        # Here we only compute the valid_action_ratio metric on phase1 (exclude padding).
        if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
            valid_action_ratio = np.mean(batch.non_tensor_batch['is_action_valid'][:bs_phase1].astype(np.float32)).item()
            metrics['episode/valid_action_ratio'] = valid_action_ratio

        # ── Update actor (Phase 1 only) ──
        if self.config.trainer.critic_warmup <= self.global_steps:
            with _timer("update_actor", timing_raw):
                batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                guide_cfg = self.config.actor_rollout_ref.actor.get("policy_loss", {}).get("guide", {})
                if guide_cfg.get("enabled", False):
                    rollout_data_dir_ = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir_:
                        batch.meta_info["internalize_dump_path"] = os.path.join(rollout_data_dir_, "internalize_probs")
                        batch.meta_info["global_step"] = self.global_steps
                actor_output = self.actor_rollout_wg.update_actor(batch)
            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            if "actor/grad_norm" in actor_output_metrics:
                actor_output_metrics["actor/grad_norm_batch-grpo"] = actor_output_metrics.pop("actor/grad_norm")
            if "actor/pg_loss" in actor_output_metrics:
                actor_output_metrics["actor/batch-grpo_loss"] = actor_output_metrics.pop("actor/pg_loss")
            if "actor/kl_loss" in actor_output_metrics:
                actor_output_metrics["actor/kl_loss_batch-grpo"] = actor_output_metrics.pop("actor/kl_loss")
            metrics.update(actor_output_metrics)

        return batch, reward_extra_infos_dict

    @staticmethod
    def _subset_reset_info(reset_info: dict, indices: list) -> dict:
        """Extract a subset of reset_info by task indices.

        reset_info is typically {'game_files': [...]} (AlfWorld) or similar
        dict-of-lists.  This selects only the entries at the given indices.
        """
        if reset_info is None:
            return None
        result = {}
        for key, value in reset_info.items():
            if isinstance(value, (list, np.ndarray)):
                result[key] = [value[i] for i in indices]
            else:
                result[key] = value
        return result

    def _internalize_step(self, gen_batch, timing_raw, metrics):
        """Internalize-only ablation: JSD distillation for hard (cliff) tasks.

        Phase 1: Plain rollout (specific skills only) for all tasks
        Phase 2: Guided rollout (full skills) for cliff tasks only → R=1 filter → JSD
        Non-cliff tasks get standard GRPO.
        Each update is an independent optimizer step (no combined mode).
        Returns the batch and reward_extra_infos_dict.
        """
        internalize_cfg = self.config.env.get('internalize', {})
        jsd_lambda = internalize_cfg.get('jsd_lambda', 1.0)
        jsd_top_k = internalize_cfg.get('jsd_top_k', 64)
        jsd_temperature = internalize_cfg.get('jsd_temperature', 1.0)
        rollout_n = self.config.actor_rollout_ref.rollout.n

        # ── Phase 1: Plain rollout (all tasks, exclude general+common hints) ──
        # NOTE: multi_turn_loop internally handles gen_batch.repeat(env.rollout.n)
        # so we pass the original gen_batch (size = train_data_size).
        with _timer("gen_plain", timing_raw):
            self.envs.set_mode(plain=True)
            plain_output = self.traj_collector.multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=self.actor_rollout_wg,
                envs=self.envs,
                is_train=True,
            )

        # Capture reset_info from Phase 1 so Phase 2 replays the same tasks
        reset_info = self.envs.get_last_reset_info()

        # Dump Phase 1 (plain) trajectories immediately (before balance_batch reorders)
        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
        if rollout_data_dir:
            plain_inputs = self.tokenizer.batch_decode(plain_output.batch["prompts"], skip_special_tokens=True)
            plain_outputs_text = self.tokenizer.batch_decode(plain_output.batch["responses"], skip_special_tokens=True)
            plain_scores = [float(x) for x in plain_output.non_tensor_batch['episode_rewards']]
            self._dump_generations(
                inputs=plain_inputs, outputs=plain_outputs_text, scores=plain_scores,
                reward_extra_infos_dict={}, dump_path=rollout_data_dir,
                traj_uids=plain_output.non_tensor_batch.get('traj_uid', None),
                uids=plain_output.non_tensor_batch.get('uid', None),
            )

        # ── Cliff detection (task-level, grouped by uid) ──
        # uid = group uid shared by G trajectories of the same task
        # traj_uid = unique per trajectory (each trajectory has its own uuid)
        # NOTE: uid values are freshly generated per multi_turn_loop call, so they
        # cannot be compared across Phase 1 and Phase 2.  We use uid only for
        # within-phase grouping and convert to a stable *task_index* (0-based
        # ordinal determined by first-occurrence order of uid) for cross-phase
        # matching.
        episode_rewards = plain_output.non_tensor_batch['episode_rewards']
        group_uids = plain_output.non_tensor_batch['uid']
        traj_uids = plain_output.non_tensor_batch['traj_uid']

        # Build uid → task_index mapping (stable ordinal based on first occurrence)
        uid_to_task_idx = {}
        for uid in group_uids:
            if uid not in uid_to_task_idx:
                uid_to_task_idx[uid] = len(uid_to_task_idx)

        # Deduplicate: get one reward per trajectory (steps share the same value)
        traj_reward = {}   # traj_uid -> reward
        traj_to_task = {}  # traj_uid -> task_index
        for g_uid, t_uid, r in zip(group_uids, traj_uids, episode_rewards):
            traj_reward[t_uid] = float(r)
            traj_to_task[t_uid] = uid_to_task_idx[g_uid]

        # Group trajectory rewards by task_index
        task_rewards = defaultdict(list)
        for t_uid, r in traj_reward.items():
            task_rewards[traj_to_task[t_uid]].append(r)

        n_total_tasks = len(task_rewards)
        cliff_task_indices = {idx for idx, rewards in task_rewards.items() if sum(rewards) == 0}
        non_cliff_task_indices = set(task_rewards.keys()) - cliff_task_indices
        n_cliff = len(cliff_task_indices)
        n_non_cliff = n_total_tasks - n_cliff

        metrics['routing/hard_ratio'] = n_cliff / n_total_tasks if n_total_tasks > 0 else 0.0
        metrics['routing/medium_ratio'] = n_non_cliff / n_total_tasks if n_total_tasks > 0 else 0.0
        metrics['routing/batch_pass_rate'] = n_non_cliff / n_total_tasks if n_total_tasks > 0 else 0.0

        print(f"[Internalize] cliff={n_cliff}/{n_total_tasks} "
              f"(hard_ratio={metrics['routing/hard_ratio']:.3f})")

        # ── Separate non-cliff samples from plain rollout ──
        step_task_indices = np.array([uid_to_task_idx[uid] for uid in group_uids])
        non_cliff_mask = np.array([idx in non_cliff_task_indices for idx in step_task_indices])
        non_cliff_idxs = np.where(non_cliff_mask)[0]

        # ── Phase 2: Guided rollout for cliff tasks ──
        guided_r1_batch = None
        if n_cliff > 0:
            with _timer("gen_guided", timing_raw):
                self.envs.set_mode(plain=False)  # guided: full skills + guide_internalize
                guided_output = self.traj_collector.multi_turn_loop(
                    gen_batch=gen_batch,
                    actor_rollout_wg=self.actor_rollout_wg,
                    envs=self.envs,
                    is_train=True,
                    reset_info=reset_info,
                )

            # Dump Phase 2 (guided) trajectories
            if rollout_data_dir:
                guided_dump_path = os.path.join(rollout_data_dir, "guided")
                guided_inputs = self.tokenizer.batch_decode(guided_output.batch["prompts"], skip_special_tokens=True)
                guided_outputs_text = self.tokenizer.batch_decode(guided_output.batch["responses"], skip_special_tokens=True)
                guided_scores = [float(x) for x in guided_output.non_tensor_batch['episode_rewards']]
                self._dump_generations(
                    inputs=guided_inputs, outputs=guided_outputs_text, scores=guided_scores,
                    reward_extra_infos_dict={}, dump_path=guided_dump_path,
                    traj_uids=guided_output.non_tensor_batch.get('traj_uid', None),
                    uids=guided_output.non_tensor_batch.get('uid', None),
                )

            # R=1 filter: only keep successful guided trajectories for cliff tasks
            guided_rewards = guided_output.non_tensor_batch['episode_rewards']
            guided_group_uids = guided_output.non_tensor_batch['uid']
            guided_traj_uids = guided_output.non_tensor_batch['traj_uid']

            # Build uid → task_index mapping for guided output (same ordinal convention)
            guided_uid_to_task_idx = {}
            for uid in guided_group_uids:
                if uid not in guided_uid_to_task_idx:
                    guided_uid_to_task_idx[uid] = len(guided_uid_to_task_idx)

            # Deduplicate: one reward per trajectory
            guided_traj_reward = {}
            guided_traj_to_task = {}
            for g_uid, t_uid, r in zip(guided_group_uids, guided_traj_uids, guided_rewards):
                guided_traj_reward[t_uid] = float(r)
                guided_traj_to_task[t_uid] = guided_uid_to_task_idx[g_uid]

            # Per-step task index for guided output
            guided_step_task = np.array([guided_uid_to_task_idx[uid] for uid in guided_group_uids])

            # R=1 mask: step belongs to a cliff task AND its trajectory reward > 0
            guided_r1_mask = np.array([
                (guided_traj_to_task.get(t_uid, -1) in cliff_task_indices) and (float(r) > 0)
                for t_uid, r in zip(guided_traj_uids, guided_rewards)
            ])
            guided_r1_idxs = np.where(guided_r1_mask)[0]

            # Count trajectory-level stats (deduplicated by traj_uid)
            n_guided_total = sum(1 for t_uid in guided_traj_reward
                                 if guided_traj_to_task[t_uid] in cliff_task_indices)
            n_guided_pass = sum(1 for t_uid, r in guided_traj_reward.items()
                                if guided_traj_to_task[t_uid] in cliff_task_indices and r > 0)

            # Count no-signal tasks (cliff + all guided trajectories fail)
            guided_task_rewards = defaultdict(list)
            for t_uid, r in guided_traj_reward.items():
                tidx = guided_traj_to_task[t_uid]
                if tidx in cliff_task_indices:
                    guided_task_rewards[tidx].append(r)
            n_no_signal = sum(1 for tidx in cliff_task_indices
                              if sum(guided_task_rewards.get(tidx, [0])) == 0)

            # Phase 2 full rollout stats
            n_guided_rollout_total = len(guided_traj_reward)  # all trajectories in Phase 2
            n_guided_rollout_steps = len(guided_output.batch['input_ids'])  # all steps in Phase 2
            n_guided_r1_steps = int(guided_r1_mask.sum())  # steps actually used for JSD

            metrics['internalize/guided_rollout_total_traj'] = n_guided_rollout_total
            metrics['internalize/guided_rollout_total_steps'] = n_guided_rollout_steps
            metrics['internalize/guided_cliff_traj_count'] = n_guided_total
            metrics['internalize/guided_pass_rate'] = n_guided_pass / n_guided_total if n_guided_total > 0 else 0.0
            metrics['internalize/guided_pass_count'] = n_guided_pass
            metrics['internalize/jsd_token_count'] = n_guided_r1_steps
            metrics['internalize/no_signal_count'] = n_no_signal
            metrics['internalize/guided_utilization'] = n_guided_r1_steps / n_guided_rollout_steps if n_guided_rollout_steps > 0 else 0.0

            print(f"[Internalize] Phase 2: total_traj={n_guided_rollout_total}, "
                  f"cliff_traj={n_guided_total}, R=1={n_guided_pass} "
                  f"(pass_rate={metrics['internalize/guided_pass_rate']:.3f}), "
                  f"jsd_token_count={n_guided_r1_steps}/{n_guided_rollout_steps} "
                  f"(utilization={metrics['internalize/guided_utilization']:.3f}), "
                  f"no_signal={n_no_signal}")

            if n_guided_pass > 0:
                guided_r1_batch = guided_output.select_idxs(guided_r1_idxs)
        else:
            metrics['internalize/guided_rollout_total_traj'] = 0
            metrics['internalize/guided_rollout_total_steps'] = 0
            metrics['internalize/guided_cliff_traj_count'] = 0
            metrics['internalize/guided_pass_rate'] = 0.0
            metrics['internalize/guided_pass_count'] = 0
            metrics['internalize/jsd_token_count'] = 0
            metrics['internalize/no_signal_count'] = 0
            metrics['internalize/guided_utilization'] = 0.0

        # Restore env mode
        print("[Internalize] Restoring env mode...")
        self.envs.restore_mode()
        print("[Internalize] Env mode restored.")

        # ── Process non-cliff batch: standard GRPO pipeline ──
        batch = None
        reward_extra_infos_dict = {}

        if len(non_cliff_idxs) > 0:
            print(f"[Internalize] Processing non-cliff batch: {len(non_cliff_idxs)} samples")
            non_cliff_batch = plain_output.select_idxs(non_cliff_idxs)
            bs_real = len(non_cliff_batch)  # Track real sample count before padding
            non_cliff_batch = adjust_batch(self.config, non_cliff_batch)
            # Mark padding samples (adjust_batch appends copies at the end)
            is_padding = np.zeros(len(non_cliff_batch), dtype=bool)
            is_padding[bs_real:] = True
            non_cliff_batch.non_tensor_batch['_is_padding'] = is_padding
            non_cliff_batch.batch["response_mask"] = compute_response_mask(non_cliff_batch)

            # Zero out padding masks immediately so they never contribute to
            # entropy/kl gradients or metrics (dp_actor uses loss_mask in multi_turn)
            n_padding_early = int(is_padding.sum())
            if n_padding_early > 0:
                padding_indices_early = torch.tensor(np.where(is_padding)[0], dtype=torch.long)
                non_cliff_batch.batch["response_mask"][padding_indices_early] = 0.0
                if "loss_mask" in non_cliff_batch.batch:
                    non_cliff_batch.batch["loss_mask"][padding_indices_early] = 0.0

            if self.config.trainer.balance_batch:
                self._balance_batch(non_cliff_batch, metrics=metrics)

            non_cliff_batch.meta_info["global_token_num"] = torch.sum(
                non_cliff_batch.batch["attention_mask"], dim=-1).tolist()

            # Reward
            with _timer("reward", timing_raw):
                reward_tensor, reward_extra_infos_dict = compute_reward(non_cliff_batch, self.reward_fn)
            non_cliff_batch.batch["token_level_scores"] = reward_tensor

            if reward_extra_infos_dict:
                non_cliff_batch.non_tensor_batch.update(
                    {k: np.array(v) for k, v in reward_extra_infos_dict.items()})

            # Apply penalties (compute on full batch including padding for alignment,
            # but only report metrics from real samples)
            padding_mask = non_cliff_batch.non_tensor_batch['_is_padding']
            if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                non_cliff_batch, invalid_metrics = apply_invalid_action_penalty(
                    non_cliff_batch,
                    invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef)
                # Recompute valid_action_ratio on real samples only
                if 'valid_actions' in non_cliff_batch.non_tensor_batch:
                    real_valid = non_cliff_batch.non_tensor_batch['valid_actions'][~padding_mask]
                    invalid_metrics['episode/valid_action_ratio'] = float(np.mean(real_valid))
                metrics.update(invalid_metrics)

            if self.config.algorithm.use_kl_in_reward:
                non_cliff_batch, kl_metrics = apply_kl_penalty(
                    non_cliff_batch, kl_ctrl=self.kl_ctrl_in_reward,
                    kl_penalty=self.config.algorithm.kl_penalty)
                metrics.update(kl_metrics)
            else:
                non_cliff_batch.batch["token_level_rewards"] = non_cliff_batch.batch["token_level_scores"]

            # Old log probs (on-policy)
            n_padding = int(padding_mask.sum())
            print(f"[Internalize] Computing old_log_prob for non-cliff batch ({len(non_cliff_batch.batch['input_ids'])} samples, "
                  f"{bs_real} real + {n_padding} padding)...")
            with _timer("old_log_prob", timing_raw):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(non_cliff_batch)
                entropys = old_log_prob.batch["entropys"]
                # Only compute entropy metric on real samples
                real_mask_bool = ~padding_mask
                real_indices = torch.tensor(np.where(real_mask_bool)[0], dtype=torch.long)
                entropys_real = entropys[real_indices]
                response_masks_real = non_cliff_batch.batch["response_mask"][real_indices]
                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                entropy_loss = agg_loss(loss_mat=entropys_real, loss_mask=response_masks_real,
                                        loss_agg_mode=loss_agg_mode)
                metrics["actor/entropy_loss_grpo"] = entropy_loss.detach().item()
                old_log_prob.batch.pop("entropys")
                non_cliff_batch = non_cliff_batch.union(old_log_prob)

            # Reference log probs
            print("[Internalize] old_log_prob done. Computing ref_log_prob...")
            if self.use_reference_policy:
                with _timer("ref", timing_raw):
                    if not self.ref_in_actor:
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(non_cliff_batch)
                    else:
                        ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(non_cliff_batch)
                    non_cliff_batch = non_cliff_batch.union(ref_log_prob)

            # Advantages
            print("[Internalize] ref done. Computing advantages...")
            with _timer("adv", timing_raw):
                norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                non_cliff_batch = compute_advantage(
                    non_cliff_batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=rollout_n,
                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                    multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                    use_pf_ppo=self.config.algorithm.use_pf_ppo,
                    pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                    pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                    step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                    gigpo_mode=self.config.algorithm.gigpo.mode,
                    gigpo_enable_similarity=self.config.algorithm.gigpo.enable_similarity,
                    gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                )

            # Zero out padding advantages (response_mask/loss_mask already zeroed above)
            if n_padding > 0:
                padding_indices = torch.tensor(np.where(padding_mask)[0], dtype=torch.long)
                non_cliff_batch.batch["advantages"][padding_indices] = 0.0
                print(f"[Internalize] Zeroed out {n_padding} padding samples' advantages")

            # Mark as non-JSD samples
            non_cliff_batch.non_tensor_batch['is_jsd_sample'] = np.zeros(
                len(non_cliff_batch.batch['input_ids']), dtype=bool)

            batch = non_cliff_batch

        print(f"[Internalize] Non-cliff processing done. batch={'set' if batch is not None else 'None'}")

        # ── Prepare cliff R=1 batch for JSD ──
        # JSD batch only needs actor forward (no rollout/ref), so minimal padding:
        # just ensure divisible by (n_gpu * actor_micro_batch_size_per_gpu)
        jsd_batch = None
        if guided_r1_batch is not None and len(guided_r1_batch.batch['input_ids']) > 0:
            jsd_batch = guided_r1_batch
            world_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
            jsd_divisor = self.config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu * world_size
            bs_jsd_real = len(jsd_batch)
            remainder = bs_jsd_real % jsd_divisor
            if remainder != 0:
                to_add = jsd_divisor - remainder
                dup_indices = np.random.choice(bs_jsd_real, to_add, replace=(to_add > bs_jsd_real))
                dup_proto = jsd_batch.select_idxs(dup_indices)
                jsd_batch = DataProto.concat([jsd_batch, dup_proto])
            jsd_batch.batch["response_mask"] = compute_response_mask(jsd_batch)
            # Zero out masks for padding samples so they don't contribute to JSD loss.
            # Multi-turn JSD uses loss_mask[:, -response_length:] as response_mask.
            # We zero loss_mask for padding; attention_mask is kept intact for valid forward pass.
            if len(jsd_batch) > bs_jsd_real:
                jsd_batch.batch["response_mask"][bs_jsd_real:] = 0.0
                if "loss_mask" in jsd_batch.batch:
                    jsd_batch.batch["loss_mask"][bs_jsd_real:] = 0.0
            print(f"[Internalize] JSD batch: {bs_jsd_real} real samples -> {len(jsd_batch)} after padding (divisor={jsd_divisor})")

        # ── Fallback: if no GRPO data (all cliff), use plain as-is for GRPO ──
        if batch is None:
            print("[Internalize] WARNING: No non-cliff samples. Using plain rollout as-is for GRPO.")
            batch = plain_output
            bs_fallback_real = len(batch)
            batch = adjust_batch(self.config, batch)
            # Mark padding samples
            is_padding_fb = np.zeros(len(batch), dtype=bool)
            is_padding_fb[bs_fallback_real:] = True
            batch.non_tensor_batch['_is_padding'] = is_padding_fb
            batch.batch["response_mask"] = compute_response_mask(batch)
            # Zero out padding masks immediately
            if is_padding_fb.any():
                padding_indices_fb_early = torch.tensor(np.where(is_padding_fb)[0], dtype=torch.long)
                batch.batch["response_mask"][padding_indices_fb_early] = 0.0
                if "loss_mask" in batch.batch:
                    batch.batch["loss_mask"][padding_indices_fb_early] = 0.0
            if self.config.trainer.balance_batch:
                self._balance_batch(batch, metrics=metrics)
            batch.meta_info["global_token_num"] = torch.sum(
                batch.batch["attention_mask"], dim=-1).tolist()
            with _timer("reward", timing_raw):
                reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
            batch.batch["token_level_scores"] = reward_tensor
            if reward_extra_infos_dict:
                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
            with _timer("old_log_prob", timing_raw):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                old_log_prob.batch.pop("entropys")
                batch = batch.union(old_log_prob)
            if self.use_reference_policy:
                with _timer("ref", timing_raw):
                    if not self.ref_in_actor:
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                    else:
                        ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)
            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=rollout_n,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                use_pf_ppo=self.config.algorithm.use_pf_ppo,
                pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                gigpo_mode=self.config.algorithm.gigpo.mode,
                gigpo_enable_similarity=self.config.algorithm.gigpo.enable_similarity,
                gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
            )
            # Zero out padding advantages (response_mask/loss_mask already zeroed above)
            padding_mask_fb = batch.non_tensor_batch['_is_padding']
            if padding_mask_fb.any():
                padding_indices_fb = torch.tensor(np.where(padding_mask_fb)[0], dtype=torch.long)
                batch.batch["advantages"][padding_indices_fb] = 0.0

        # ── Update actor: independent GRPO step + independent JSD step ──
        if self.config.trainer.critic_warmup <= self.global_steps:
            # Step 1: GRPO update on non-cliff (or fallback) batch
            print(f"[Internalize] GRPO update_actor: {len(batch.batch['input_ids'])} samples")
            with _timer("update_actor", timing_raw):
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                batch.meta_info["global_token_num"] = torch.sum(
                    batch.batch["attention_mask"], dim=-1).tolist()
                batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                batch.meta_info["hdpo_mode"] = "grpo"
                actor_output = self.actor_rollout_wg.update_actor(batch)
            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            if "actor/grad_norm" in actor_output_metrics:
                actor_output_metrics["actor/grad_norm_grpo"] = actor_output_metrics.pop("actor/grad_norm")
            if "actor/pg_loss" in actor_output_metrics:
                actor_output_metrics["actor/grpo_loss"] = actor_output_metrics.pop("actor/pg_loss")
            if "actor/kl_loss" in actor_output_metrics:
                actor_output_metrics["actor/kl_loss_grpo"] = actor_output_metrics.pop("actor/kl_loss")
            metrics.update(actor_output_metrics)

            # Step 2: JSD update on cliff R=1 batch (independent step)
            if jsd_batch is not None:
                print(f"[Internalize] JSD update_actor: {len(jsd_batch.batch['input_ids'])} samples")
                with _timer("update_actor_jsd", timing_raw):
                    jsd_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                    jsd_batch.meta_info["global_token_num"] = torch.sum(
                        jsd_batch.batch["attention_mask"], dim=-1).tolist()
                    jsd_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    jsd_batch.meta_info["hdpo_mode"] = "jsd"
                    jsd_batch.meta_info["hdpo_config"] = {
                        'jsd_lambda': jsd_lambda,
                        'jsd_top_k': jsd_top_k,
                        'jsd_temperature': jsd_temperature,
                    }
                    jsd_output = self.actor_rollout_wg.update_actor(jsd_batch)
                jsd_output_metrics = reduce_metrics(jsd_output.meta_info["metrics"])
                if "actor/grad_norm" in jsd_output_metrics:
                    jsd_output_metrics["actor/grad_norm_jsd"] = jsd_output_metrics.pop("actor/grad_norm")
                metrics.update(jsd_output_metrics)

        return batch, reward_extra_infos_dict

    def _dual_distill_step(self, gen_batch, timing_raw, metrics):
        """Unified GRPO + Dual Adaptive Distillation.

        Phase 1: Plain rollout (specific skills only) → 128 samples, full GRPO
        Phase 2: Adaptive choice of guided OR noskill rollout → conditional JSD distillation

        Internalize branch: guided success → JSD distill to plain prompt
        Robustness branch: noskill success → JSD distill to skill prompt
        """
        dual_cfg = self.config.env.get('dual_distill', {})
        phase2_mode = dual_cfg.get('phase2_mode', 'adaptive')  # full / alternate / adaptive
        window_size = dual_cfg.get('window_size', 5)
        jsd_lambda_intern = dual_cfg.get('jsd_lambda_intern', 1.0)
        jsd_lambda_robust = dual_cfg.get('jsd_lambda_robust', 1.0)
        # When both branches fire in the same step, halve lambda to keep total JSD gradient stable
        # This applies to full mode (always both) and adaptive bootstrap (both during window fill)
        both_branches_this_step = (phase2_mode == 'full') or (
            phase2_mode == 'adaptive' and self.global_steps <= window_size)
        if both_branches_this_step:
            jsd_lambda_intern = jsd_lambda_intern * 0.5
            jsd_lambda_robust = jsd_lambda_robust * 0.5
        jsd_top_k = dual_cfg.get('jsd_top_k', 64)
        jsd_temperature = dual_cfg.get('jsd_temperature', 1.0)
        rollout_n = self.config.actor_rollout_ref.rollout.n
        env_rollout_n = self.config.env.rollout.n  # env workers per task (used for task_modes expansion)

        # ══════════════════════════════════════════════════════════════
        # Phase 1: Plain rollout (specific skills only)
        # ══════════════════════════════════════════════════════════════
        with _timer("gen_plain", timing_raw):
            self.envs.set_mode(plain=True)
            plain_output = self.traj_collector.multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=self.actor_rollout_wg,
                envs=self.envs,
                is_train=True,
            )

        # Capture reset_info for replaying same tasks in Phase 2
        reset_info = self.envs.get_last_reset_info()

        # Dump plain trajectories immediately (before balance_batch reorders)
        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
        if rollout_data_dir:
            plain_inputs = self.tokenizer.batch_decode(plain_output.batch["prompts"], skip_special_tokens=True)
            plain_outputs_text = self.tokenizer.batch_decode(plain_output.batch["responses"], skip_special_tokens=True)
            plain_scores = plain_output.non_tensor_batch['episode_rewards']
            plain_scores = [float(x) for x in plain_scores]
            self._dump_generations(
                inputs=plain_inputs, outputs=plain_outputs_text, scores=plain_scores,
                reward_extra_infos_dict={}, dump_path=rollout_data_dir,
                traj_uids=plain_output.non_tensor_batch.get('traj_uid', None),
                uids=plain_output.non_tensor_batch.get('uid', None),
            )

        # ══════════════════════════════════════════════════════════════
        # Task-level pass rate computation (Phase 1)
        # ══════════════════════════════════════════════════════════════
        episode_rewards = plain_output.non_tensor_batch['episode_rewards']
        group_uids = plain_output.non_tensor_batch['uid']
        traj_uids = plain_output.non_tensor_batch['traj_uid']

        uid_to_task_idx = {}
        for uid in group_uids:
            if uid not in uid_to_task_idx:
                uid_to_task_idx[uid] = len(uid_to_task_idx)

        traj_reward = {}
        traj_to_task = {}
        for g_uid, t_uid, r in zip(group_uids, traj_uids, episode_rewards):
            traj_reward[t_uid] = float(r)
            traj_to_task[t_uid] = uid_to_task_idx[g_uid]

        task_rewards = defaultdict(list)
        for t_uid, r in traj_reward.items():
            task_rewards[traj_to_task[t_uid]].append(r)

        n_total_tasks = len(task_rewards)
        task_pass_rates = {}
        for tidx, rewards in task_rewards.items():
            task_pass_rates[tidx] = sum(1 for r in rewards if r > 0) / len(rewards)

        plain_batch_pr = np.mean(list(task_pass_rates.values())) if task_pass_rates else 0.0

        # Extract unique uids for Phase 2 replay
        seen = set()
        uid_base = []
        for uid in group_uids:
            if uid not in seen:
                seen.add(uid)
                uid_base.append(uid)
        uid_base = np.array(uid_base, dtype=object)

        # Phase 1 episode metrics
        unique_traj_uids_p1, unique_idx_p1 = np.unique(plain_output.non_tensor_batch['traj_uid'], return_index=True)
        episode_rewards_p1 = plain_output.non_tensor_batch['episode_rewards'][unique_idx_p1]
        episode_lengths_p1 = plain_output.non_tensor_batch['episode_lengths'][unique_idx_p1]
        metrics['episode/reward/mean'] = float(episode_rewards_p1.mean())
        metrics['episode/reward/max'] = float(episode_rewards_p1.max())
        metrics['episode/reward/min'] = float(episode_rewards_p1.min())
        metrics['episode/length/mean'] = float(episode_lengths_p1.mean())
        metrics['episode/length/max'] = float(episode_lengths_p1.max())
        metrics['episode/length/min'] = float(episode_lengths_p1.min())
        for k, v in plain_output.non_tensor_batch.items():
            if "success_rate" in k:
                metrics[f'episode/{k}'] = float(v[0])

        # ══════════════════════════════════════════════════════════════
        # Phase 2: Adaptive scheduling — decide guided vs noskill
        # ══════════════════════════════════════════════════════════════
        run_guided = False
        run_noskill = False

        if phase2_mode == 'full':
            run_guided = True
            run_noskill = True
        elif phase2_mode == 'alternate':
            if self.global_steps % 2 == 1:
                run_guided = True
            else:
                run_noskill = True
        elif phase2_mode == 'adaptive':
            # Bootstrap: first window_size steps run both (fill windows)
            if self.global_steps <= window_size:
                run_guided = True
                run_noskill = True
            else:
                import random
                epsilon = dual_cfg.get('epsilon', 0.3)
                intern_mean = np.mean(self._dual_intern_window) if len(self._dual_intern_window) > 0 else 0.0
                robust_mean = np.mean(self._dual_robust_window) if len(self._dual_robust_window) > 0 else 0.0
                if random.random() < epsilon:
                    # Epsilon-greedy: explore the minority branch
                    if intern_mean > robust_mean:
                        run_noskill = True  # explore: pick the non-greedy branch
                    elif robust_mean > intern_mean:
                        run_guided = True
                    else:
                        if random.random() < 0.5:
                            run_guided = True
                        else:
                            run_noskill = True
                    metrics['dual/epsilon_explored'] = 1.0
                else:
                    # Greedy: pick the branch with higher mean delta
                    if intern_mean > robust_mean:
                        run_guided = True
                    elif robust_mean > intern_mean:
                        run_noskill = True
                    else:
                        if random.random() < 0.5:
                            run_guided = True
                        else:
                            run_noskill = True
                    metrics['dual/epsilon_explored'] = 0.0
                metrics['dual/intern_window_mean'] = float(intern_mean)
                metrics['dual/robust_window_mean'] = float(robust_mean)

        metrics['dual/phase2_choice'] = 0.0 if (run_guided and not run_noskill) else (1.0 if (run_noskill and not run_guided) else 0.5)

        print(f"[Dual] Step {self.global_steps}: phase2_mode={phase2_mode}, "
            f"run_guided={run_guided}, run_noskill={run_noskill}, plain_pr={plain_batch_pr:.4f}")

        # ══════════════════════════════════════════════════════════════
        # Phase 2a: Guided rollout (internalize branch)
        # ══════════════════════════════════════════════════════════════
        guided_output = None
        guided_task_pass_rates = {}

        if run_guided:
            with _timer("gen_guided", timing_raw):
                self.envs.set_mode(plain=False)  # guide_internalize mode
                guided_output = self.traj_collector.multi_turn_loop(
                    gen_batch=gen_batch,
                    actor_rollout_wg=self.actor_rollout_wg,
                    envs=self.envs,
                    is_train=True,
                    reset_info=reset_info,
                    uid_base=uid_base,
                )

            # Debug: verify guided prompt content
            _g_prompts = self.tokenizer.batch_decode(guided_output.batch["prompts"][:4], skip_special_tokens=True)
            for _gi, _gp in enumerate(_g_prompts):
                _has_general = "General Principles" in _gp
                _has_task = "Task-Relevant Skills" in _gp
                _has_retrieved = "Retrieved Relevant Experience" in _gp
                print(f"[Dual DEBUG] Guided sample {_gi}: len={len(_gp)}, has_general={_has_general}, has_task={_has_task}, has_retrieved={_has_retrieved}")

            # Compute guided per-task pass rates
            g_group_uids = guided_output.non_tensor_batch['uid']
            g_traj_uids = guided_output.non_tensor_batch['traj_uid']
            g_rewards = guided_output.non_tensor_batch['episode_rewards']

            g_uid_to_task_idx = {}
            for uid in g_group_uids:
                if uid not in g_uid_to_task_idx:
                    g_uid_to_task_idx[uid] = len(g_uid_to_task_idx)

            g_traj_reward = {}
            g_traj_to_task = {}
            for g_uid, t_uid, r in zip(g_group_uids, g_traj_uids, g_rewards):
                g_traj_reward[t_uid] = float(r)
                g_traj_to_task[t_uid] = g_uid_to_task_idx[g_uid]

            g_task_rewards = defaultdict(list)
            for t_uid, r in g_traj_reward.items():
                g_task_rewards[g_traj_to_task[t_uid]].append(r)

            for tidx, rewards in g_task_rewards.items():
                guided_task_pass_rates[tidx] = sum(1 for r in rewards if r > 0) / len(rewards)

            # Dump guided trajectories
            rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
            if rollout_data_dir:
                guided_dump_path = os.path.join(rollout_data_dir, "guided")
                guided_inputs = self.tokenizer.batch_decode(guided_output.batch["prompts"], skip_special_tokens=True)
                guided_outputs_text = self.tokenizer.batch_decode(guided_output.batch["responses"], skip_special_tokens=True)
                guided_scores = guided_output.batch["token_level_scores"].sum(-1).cpu().tolist() if "token_level_scores" in guided_output.batch else [float(x) for x in g_rewards]
                self._dump_generations(
                    inputs=guided_inputs,
                    outputs=guided_outputs_text,
                    scores=guided_scores,
                    reward_extra_infos_dict={},
                    dump_path=guided_dump_path,
                    traj_uids=g_traj_uids,
                    uids=g_group_uids,
                )

        # ══════════════════════════════════════════════════════════════
        # Phase 2b: Noskill rollout (robustness branch)
        # ══════════════════════════════════════════════════════════════
        noskill_output = None
        noskill_task_pass_rates = {}

        if run_noskill:
            with _timer("gen_noskill", timing_raw):
                # All tasks in noskill mode
                task_modes = ['noskill'] * (n_total_tasks * env_rollout_n)
                self.envs.set_mode(plain=False)
                self.envs.set_per_task_mode(task_modes)
                noskill_output = self.traj_collector.multi_turn_loop(
                    gen_batch=gen_batch,
                    actor_rollout_wg=self.actor_rollout_wg,
                    envs=self.envs,
                    is_train=True,
                    reset_info=reset_info,
                    uid_base=uid_base,
                )
                self.envs.clear_per_task_mode()

            # Debug: verify noskill prompt content
            _n_prompts = self.tokenizer.batch_decode(noskill_output.batch["prompts"][:4], skip_special_tokens=True)
            for _ni, _np in enumerate(_n_prompts):
                _has_general = "General Principles" in _np
                _has_task = "Task-Relevant Skills" in _np
                _has_retrieved = "Retrieved Relevant Experience" in _np
                print(f"[Dual DEBUG] Noskill sample {_ni}: len={len(_np)}, has_general={_has_general}, has_task={_has_task}, has_retrieved={_has_retrieved}")
            # Also check samples 16-19 (second task group)
            if len(noskill_output.batch["prompts"]) > 19:
                _n_prompts2 = self.tokenizer.batch_decode(noskill_output.batch["prompts"][16:20], skip_special_tokens=True)
                for _ni, _np in enumerate(_n_prompts2):
                    _has_general = "General Principles" in _np
                    _has_task = "Task-Relevant Skills" in _np
                    print(f"[Dual DEBUG] Noskill sample {_ni+16}: len={len(_np)}, has_general={_has_general}, has_task={_has_task}")

            # Compute noskill per-task pass rates
            ns_group_uids = noskill_output.non_tensor_batch['uid']
            ns_traj_uids = noskill_output.non_tensor_batch['traj_uid']
            ns_rewards = noskill_output.non_tensor_batch['episode_rewards']

            ns_uid_to_task_idx = {}
            for uid in ns_group_uids:
                if uid not in ns_uid_to_task_idx:
                    ns_uid_to_task_idx[uid] = len(ns_uid_to_task_idx)

            ns_traj_reward = {}
            ns_traj_to_task = {}
            for g_uid, t_uid, r in zip(ns_group_uids, ns_traj_uids, ns_rewards):
                ns_traj_reward[t_uid] = float(r)
                ns_traj_to_task[t_uid] = ns_uid_to_task_idx[g_uid]

            ns_task_rewards = defaultdict(list)
            for t_uid, r in ns_traj_reward.items():
                ns_task_rewards[ns_traj_to_task[t_uid]].append(r)

            for tidx, rewards in ns_task_rewards.items():
                noskill_task_pass_rates[tidx] = sum(1 for r in rewards if r > 0) / len(rewards)

            # Dump noskill trajectories
            rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
            if rollout_data_dir:
                noskill_dump_path = os.path.join(rollout_data_dir, "noskill")
                noskill_inputs = self.tokenizer.batch_decode(noskill_output.batch["prompts"], skip_special_tokens=True)
                noskill_outputs_text = self.tokenizer.batch_decode(noskill_output.batch["responses"], skip_special_tokens=True)
                noskill_scores = noskill_output.batch["token_level_scores"].sum(-1).cpu().tolist() if "token_level_scores" in noskill_output.batch else [float(x) for x in ns_rewards]
                self._dump_generations(
                    inputs=noskill_inputs,
                    outputs=noskill_outputs_text,
                    scores=noskill_scores,
                    reward_extra_infos_dict={},
                    dump_path=noskill_dump_path,
                    traj_uids=ns_traj_uids,
                    uids=ns_group_uids,
                )

        # Restore env mode
        self.envs.restore_mode()

        # ══════════════════════════════════════════════════════════════
        # Update sliding windows (delta for adaptive scheduling)
        # ══════════════════════════════════════════════════════════════
        if run_guided and guided_task_pass_rates:
            guided_mean_pr = np.mean(list(guided_task_pass_rates.values()))
            intern_delta = float(guided_mean_pr - plain_batch_pr)
            self._dual_intern_window.append(intern_delta)
            self._dual_intern_count += 1
            metrics['dual/intern_delta'] = intern_delta
            metrics['dual/guided_success_rate'] = float(guided_mean_pr)
            metrics['dual/intern_cumulative_count'] = self._dual_intern_count

        if run_noskill and noskill_task_pass_rates:
            noskill_mean_pr = np.mean(list(noskill_task_pass_rates.values()))
            robust_delta = float(noskill_mean_pr - plain_batch_pr)
            self._dual_robust_window.append(robust_delta)
            self._dual_robust_count += 1
            metrics['dual/robust_delta'] = robust_delta
            metrics['dual/noskill_success_rate'] = float(noskill_mean_pr)
            metrics['dual/robust_cumulative_count'] = self._dual_robust_count

        # ══════════════════════════════════════════════════════════════
        # Update 1: Full GRPO on all Phase 1 samples (128)
        # ══════════════════════════════════════════════════════════════
        batch = plain_output
        batch = adjust_batch(self.config, batch)
        batch.batch["response_mask"] = compute_response_mask(batch)

        if self.config.trainer.balance_batch:
            self._balance_batch(batch, metrics=metrics)

        batch.meta_info["global_token_num"] = torch.sum(
            batch.batch["attention_mask"], dim=-1).tolist()

        # Reward
        with _timer("reward", timing_raw):
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
        batch.batch["token_level_scores"] = reward_tensor
        if reward_extra_infos_dict:
            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

        # Apply invalid action penalty
        if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
            batch, invalid_metrics = apply_invalid_action_penalty(
                batch,
                invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef)
            metrics.update(invalid_metrics)

        # token_level_rewards
        if self.config.algorithm.use_kl_in_reward:
            batch, kl_metrics = apply_kl_penalty(
                batch, kl_ctrl=self.kl_ctrl_in_reward,
                kl_penalty=self.config.algorithm.kl_penalty)
            metrics.update(kl_metrics)
        else:
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

        # Old log probs
        with _timer("old_log_prob", timing_raw):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            metrics["actor/entropy_loss"] = entropy_loss.detach().item()
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

        # Ref log probs
        if self.use_reference_policy:
            with _timer("ref", timing_raw):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        # Advantage (standard GRPO, task-level z-score)
        with _timer("adv", timing_raw):
            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=rollout_n,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                use_pf_ppo=self.config.algorithm.use_pf_ppo,
                pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                gigpo_mode=self.config.algorithm.gigpo.mode,
                gigpo_enable_similarity=self.config.algorithm.gigpo.enable_similarity,
                gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
            )

        # Update actor: GRPO
        if self.config.trainer.critic_warmup <= self.global_steps:
            with _timer("update_actor_grpo", timing_raw):
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                batch.meta_info["hdpo_mode"] = "grpo"
                actor_output = self.actor_rollout_wg.update_actor(batch)
            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            if "actor/grad_norm" in actor_output_metrics:
                actor_output_metrics["actor/grad_norm_grpo"] = actor_output_metrics.pop("actor/grad_norm")
            if "actor/pg_loss" in actor_output_metrics:
                actor_output_metrics["actor/grpo_loss"] = actor_output_metrics.pop("actor/pg_loss")
            if "actor/kl_loss" in actor_output_metrics:
                actor_output_metrics["actor/kl_loss_grpo"] = actor_output_metrics.pop("actor/kl_loss")
            metrics.update(actor_output_metrics)

        print(f"[Dual] Update 1 (GRPO): {len(batch.batch['input_ids'])} samples")

        # ══════════════════════════════════════════════════════════════
        # Update 2a: Internalize JSD (guided branch, conditional)
        # Condition: per-task guided_pr > plain_pr, take guided R>0 trajectories
        # ══════════════════════════════════════════════════════════════
        if run_guided and guided_output is not None:
            # Find tasks where guided_pr > plain_pr
            intern_fire_tasks = set()
            for tidx in guided_task_pass_rates:
                g_pr = guided_task_pass_rates[tidx]
                p_pr = task_pass_rates.get(tidx, 0.0)
                if g_pr > p_pr:
                    intern_fire_tasks.add(tidx)

            metrics['dual/intern_fire_rate'] = len(intern_fire_tasks) / n_total_tasks if n_total_tasks > 0 else 0.0

            if len(intern_fire_tasks) > 0:
                # Select guided R>0 samples from tasks that fire
                g_group_uids = guided_output.non_tensor_batch['uid']
                g_traj_uids = guided_output.non_tensor_batch['traj_uid']
                g_rewards = guided_output.non_tensor_batch['episode_rewards']

                g_uid_to_task_idx_local = {}
                for uid in g_group_uids:
                    if uid not in g_uid_to_task_idx_local:
                        g_uid_to_task_idx_local[uid] = len(g_uid_to_task_idx_local)

                jsd_mask = np.array([
                    (g_uid_to_task_idx_local.get(g_uid, -1) in intern_fire_tasks) and (float(r) > 0)
                    for g_uid, r in zip(g_group_uids, g_rewards)
                ])
                jsd_idxs = np.where(jsd_mask)[0]

                if len(jsd_idxs) > 0:
                    jsd_batch = guided_output.select_idxs(jsd_idxs)
                    bs_jsd_real = len(jsd_batch)

                    # Pad to divisible by micro_batch
                    world_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
                    jsd_divisor = self.config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu * world_size
                    remainder = bs_jsd_real % jsd_divisor
                    if remainder != 0:
                        to_add = jsd_divisor - remainder
                        dup_indices = np.random.choice(bs_jsd_real, to_add, replace=(to_add > bs_jsd_real))
                        dup_proto = jsd_batch.select_idxs(dup_indices)
                        jsd_batch = DataProto.concat([jsd_batch, dup_proto])

                    jsd_batch.batch["response_mask"] = compute_response_mask(jsd_batch)
                    # Zero out padding masks
                    if len(jsd_batch) > bs_jsd_real:
                        jsd_batch.batch["response_mask"][bs_jsd_real:] = 0.0
                        if "loss_mask" in jsd_batch.batch:
                            jsd_batch.batch["loss_mask"][bs_jsd_real:] = 0.0

                    # Update actor: JSD internalize
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("update_actor_intern_jsd", timing_raw):
                            jsd_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                            jsd_batch.meta_info["global_token_num"] = torch.sum(
                                jsd_batch.batch["attention_mask"], dim=-1).tolist()
                            jsd_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            jsd_batch.meta_info["hdpo_mode"] = "jsd"
                            jsd_batch.meta_info["hdpo_config"] = {
                                'jsd_lambda': jsd_lambda_intern,
                                'jsd_top_k': jsd_top_k,
                                'jsd_temperature': jsd_temperature,
                            }
                            jsd_output = self.actor_rollout_wg.update_actor(jsd_batch)
                        jsd_output_metrics = reduce_metrics(jsd_output.meta_info["metrics"])
                        print(f"[Dual DEBUG] intern jsd_output_metrics keys: {list(jsd_output_metrics.keys())}")
                        # Rename keys: indicator first, _intern suffix for alphabetical grouping
                        intern_jsd_metrics = {}
                        jsd_rename_map = {
                            "actor/grad_norm": "actor/grad_norm_jsd_intern",
                            "actor/jsd_loss": "actor/jsd_loss_intern",
                            "actor/jsd_mean_per_token": "actor/jsd_mean_per_token_intern",
                            "actor/jsd_max_per_token": "actor/jsd_max_per_token_intern",
                            "actor/jsd_tail_mass": "actor/jsd_tail_mass_intern",
                            "actor/jsd_token_count": "actor/jsd_token_count_intern",
                        }
                        skip_keys = {"actor/jsd_lambda"}
                        for k, v in jsd_output_metrics.items():
                            if k in skip_keys:
                                continue
                            elif k in jsd_rename_map:
                                intern_jsd_metrics[jsd_rename_map[k]] = v
                            else:
                                intern_jsd_metrics[k] = v
                        metrics.update(intern_jsd_metrics)

                    print(f"[Dual] Update 2a (intern JSD): {bs_jsd_real} real samples, "
                        f"fire_tasks={len(intern_fire_tasks)}/{n_total_tasks}")
                else:
                    print(f"[Dual] Update 2a (intern JSD): skipped, no R>0 in fire tasks")
            else:
                print(f"[Dual] Update 2a (intern JSD): skipped, no task satisfies guided_pr > plain_pr")

        # ══════════════════════════════════════════════════════════════
        # Update 2b: Robustness JSD (noskill branch, conditional)
        # Condition: per-task noskill_pr > skill_pr (=plain_pr), take noskill R>0
        # Need to construct plain_* fields: skill prompt + noskill response
        # ══════════════════════════════════════════════════════════════
        if run_noskill and noskill_output is not None:
            # Find tasks where noskill_pr > plain_pr (skill_pr)
            robust_fire_tasks = set()
            for tidx in noskill_task_pass_rates:
                ns_pr = noskill_task_pass_rates[tidx]
                p_pr = task_pass_rates.get(tidx, 0.0)
                if ns_pr > p_pr:
                    robust_fire_tasks.add(tidx)

            metrics['dual/robust_fire_rate'] = len(robust_fire_tasks) / n_total_tasks if n_total_tasks > 0 else 0.0

            if len(robust_fire_tasks) > 0:
                # Select noskill R>0 samples from tasks that fire
                ns_group_uids = noskill_output.non_tensor_batch['uid']
                ns_traj_uids = noskill_output.non_tensor_batch['traj_uid']
                ns_rewards = noskill_output.non_tensor_batch['episode_rewards']

                ns_uid_to_task_idx_local = {}
                for uid in ns_group_uids:
                    if uid not in ns_uid_to_task_idx_local:
                        ns_uid_to_task_idx_local[uid] = len(ns_uid_to_task_idx_local)

                robust_mask = np.array([
                    (ns_uid_to_task_idx_local.get(g_uid, -1) in robust_fire_tasks) and (float(r) > 0)
                    for g_uid, r in zip(ns_group_uids, ns_rewards)
                ])
                robust_idxs = np.where(robust_mask)[0]

                if len(robust_idxs) > 0:
                    robust_jsd_batch = noskill_output.select_idxs(robust_idxs)
                    bs_robust_real = len(robust_jsd_batch)

                    # ── Construct plain_* fields: skill prompt + noskill response ──
                    # For each sample, find the corresponding Phase 1 prompt (skill version)
                    # and replace the prompt portion to create the "student" input
                    robust_uids = robust_jsd_batch.non_tensor_batch['uid']

                    # Build uid -> first sample index in plain_output (to get skill prompt)
                    p1_uid_to_first_idx = {}
                    for i, uid in enumerate(plain_output.non_tensor_batch['uid']):
                        if uid not in p1_uid_to_first_idx:
                            p1_uid_to_first_idx[uid] = i

                    # Get sequence length
                    seq_len = robust_jsd_batch.batch['input_ids'].shape[1]

                    # Construct plain_input_ids, plain_attention_mask, plain_position_ids
                    plain_input_ids_list = []
                    plain_attention_mask_list = []
                    plain_position_ids_list = []

                    for i in range(bs_robust_real):
                        sample_uid = robust_uids[i]
                        p1_idx = p1_uid_to_first_idx[sample_uid]

                        # Skill prompt from Phase 1
                        p1_input_ids = plain_output.batch['input_ids'][p1_idx]  # [seq_len]
                        p1_attention_mask = plain_output.batch['attention_mask'][p1_idx]

                        # Find prompt length in Phase 1 (non-response part)
                        # Use response_mask or prompt boundary from non_tensor_batch
                        if 'prompt_length' in plain_output.non_tensor_batch:
                            skill_prompt_len = int(plain_output.non_tensor_batch['prompt_length'][p1_idx])
                        else:
                            # Fallback: find first response token (where response starts)
                            p1_response_mask = plain_output.batch.get('response_mask', None)
                            if p1_response_mask is not None:
                                resp_positions = torch.where(p1_response_mask[p1_idx] > 0)[0]
                                skill_prompt_len = int(resp_positions[0]) if len(resp_positions) > 0 else int(p1_attention_mask.sum())
                            else:                                                                                                                             
                                skill_prompt_len = int(p1_attention_mask.sum()) // 2  # rough fallback                                                        
                                                                                                                                                            
                        # Noskill response from this sample                                                                                                   
                        ns_input_ids = robust_jsd_batch.batch['input_ids'][i]                                                                                 
                        ns_attention_mask = robust_jsd_batch.batch['attention_mask'][i]                                                                       
                                                                                                                                                            
                        if 'prompt_length' in robust_jsd_batch.non_tensor_batch:                                                                              
                            ns_prompt_len = int(robust_jsd_batch.non_tensor_batch['prompt_length'][i])                                                        
                        else:                                                                                                                                 
                            ns_resp_mask = robust_jsd_batch.batch.get('response_mask', None)                                                                  
                            if ns_resp_mask is not None:                                                                                                      
                                ns_resp_positions = torch.where(ns_resp_mask[i] > 0)[0]                                                                       
                                ns_prompt_len = int(ns_resp_positions[0]) if len(ns_resp_positions) > 0 else int(ns_attention_mask.sum())                     
                            else:                                                                                                                             
                                ns_prompt_len = int(ns_attention_mask.sum()) // 2                                                                             
                                                                                                                                                            
                        # Response tokens from noskill                                                                                                        
                        ns_response_ids = ns_input_ids[ns_prompt_len:]                                                                                        
                        ns_response_len = int(ns_attention_mask[ns_prompt_len:].sum())                                                                        
                                                                                                                                                            
                        # Skill prompt tokens                                                                                                                 
                        skill_prompt_ids = p1_input_ids[:skill_prompt_len]                                                                                    
                                                                                                                                                            
                        # Concatenate: skill_prompt + noskill_response, pad to seq_len                                                                        
                        total_len = skill_prompt_len + ns_response_len                                                                                        
                        pad_len = seq_len - total_len                                                                                                         
                                                                                                                                                            
                        if pad_len >= 0:
                            # Left-pad with zeros (consistent with training format)
                            new_input_ids = torch.cat([
                                torch.zeros(pad_len, dtype=p1_input_ids.dtype, device=p1_input_ids.device),
                                skill_prompt_ids[:skill_prompt_len],
                                ns_response_ids[:ns_response_len],
                            ])
                            new_attention_mask = torch.cat([
                                torch.zeros(pad_len, dtype=p1_attention_mask.dtype, device=p1_attention_mask.device),
                                torch.ones(total_len, dtype=p1_attention_mask.dtype, device=p1_attention_mask.device),
                            ])
                        else:
                            # Truncate from left (skill prompt) if too long
                            overflow = -pad_len
                            new_input_ids = torch.cat([
                                skill_prompt_ids[overflow:skill_prompt_len],
                                ns_response_ids[:ns_response_len],
                            ])
                            new_attention_mask = torch.ones(seq_len, dtype=p1_attention_mask.dtype, device=p1_attention_mask.device)

                        # position_ids must match attention_mask: cumsum(mask)-1, clamp min=0
                        new_position_ids = torch.clamp(torch.cumsum(new_attention_mask.long(), dim=-1) - 1, min=0)

                        plain_input_ids_list.append(new_input_ids)
                        plain_attention_mask_list.append(new_attention_mask)
                        plain_position_ids_list.append(new_position_ids)

                    # Stack into batch tensors
                    robust_jsd_batch.batch['plain_input_ids'] = torch.stack(plain_input_ids_list)
                    robust_jsd_batch.batch['plain_attention_mask'] = torch.stack(plain_attention_mask_list)
                    robust_jsd_batch.batch['plain_position_ids'] = torch.stack(plain_position_ids_list)

                    # Pad batch to divisible by micro_batch
                    world_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
                    jsd_divisor = self.config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu * world_size
                    remainder = bs_robust_real % jsd_divisor
                    if remainder != 0:
                        to_add = jsd_divisor - remainder
                        dup_indices = np.random.choice(bs_robust_real, to_add, replace=(to_add > bs_robust_real))
                        dup_proto = robust_jsd_batch.select_idxs(dup_indices)
                        robust_jsd_batch = DataProto.concat([robust_jsd_batch, dup_proto])

                    robust_jsd_batch.batch["response_mask"] = compute_response_mask(robust_jsd_batch)
                    # Zero out padding masks
                    if len(robust_jsd_batch) > bs_robust_real:
                        robust_jsd_batch.batch["response_mask"][bs_robust_real:] = 0.0
                        if "loss_mask" in robust_jsd_batch.batch:
                            robust_jsd_batch.batch["loss_mask"][bs_robust_real:] = 0.0

                    # Update actor: JSD robustness
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("update_actor_robust_jsd", timing_raw):
                            robust_jsd_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                            robust_jsd_batch.meta_info["global_token_num"] = torch.sum(
                                robust_jsd_batch.batch["attention_mask"], dim=-1).tolist()
                            robust_jsd_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            robust_jsd_batch.meta_info["hdpo_mode"] = "jsd"
                            robust_jsd_batch.meta_info["hdpo_config"] = {
                                'jsd_lambda': jsd_lambda_robust,
                                'jsd_top_k': jsd_top_k,
                                'jsd_temperature': jsd_temperature,
                            }
                            robust_output = self.actor_rollout_wg.update_actor(robust_jsd_batch)
                        robust_output_metrics = reduce_metrics(robust_output.meta_info["metrics"])
                        # Rename keys: indicator first, _robust suffix for alphabetical grouping
                        robust_jsd_metrics = {}
                        skip_keys = {"actor/jsd_lambda"}
                        jsd_rename_map_r = {
                            "actor/grad_norm": "actor/grad_norm_jsd_robust",
                            "actor/jsd_loss": "actor/jsd_loss_robust",
                            "actor/jsd_mean_per_token": "actor/jsd_mean_per_token_robust",
                            "actor/jsd_max_per_token": "actor/jsd_max_per_token_robust",
                            "actor/jsd_tail_mass": "actor/jsd_tail_mass_robust",
                            "actor/jsd_token_count": "actor/jsd_token_count_robust",
                        }
                        for k, v in robust_output_metrics.items():
                            if k in skip_keys:
                                continue
                            elif k in jsd_rename_map_r:
                                robust_jsd_metrics[jsd_rename_map_r[k]] = v
                            else:
                                robust_jsd_metrics[k] = v
                        metrics.update(robust_jsd_metrics)

                    print(f"[Dual] Update 2b (robust JSD): {bs_robust_real} real samples, "
                        f"fire_tasks={len(robust_fire_tasks)}/{n_total_tasks}")
                else:
                    print(f"[Dual] Update 2b (robust JSD): skipped, no R>0 in fire tasks")
            else:
                print(f"[Dual] Update 2b (robust JSD): skipped, no task satisfies noskill_pr > plain_pr")

        return batch, reward_extra_infos_dict

    def _ours_step(self, gen_batch, timing_raw, metrics):
        """Three-tier adaptive routing: hard(JSD) / medium(GRPO) / easy(contrastive).

        Phase 1: Plain rollout (specific skills only) → per-task pass_rate
        Routing (EMA-based):
          - hard:   pass_rate == 0          → Phase 2a: Guided rollout → R=1 → JSD
          - medium: 0 < pass_rate < EMA     → standard GRPO on Phase 1 data
          - easy:   pass_rate >= EMA        → Phase 2b: No-skill probe → contrastive GRPO

        Three independent update_actor calls (each full optimizer cycle).
        Returns the batch (for logging/dumping) and reward_extra_infos_dict.
        """
        internalize_cfg = self.config.env.get('internalize', {})
        jsd_lambda = internalize_cfg.get('jsd_lambda', 1.0)
        jsd_top_k = internalize_cfg.get('jsd_top_k', 64)
        jsd_temperature = internalize_cfg.get('jsd_temperature', 1.0)
        ours_cfg = self.config.env.get('ours', {})
        warmup_steps = ours_cfg.get('warmup_steps', 10)
        rollout_n = self.config.actor_rollout_ref.rollout.n
        env_rollout_n = self.config.env.rollout.n  # env workers per task (used for task_modes expansion)

        # ══════════════════════════════════════════════════════════════
        # Phase 1: Plain rollout (specific skills only, exclude general+common)
        # ══════════════════════════════════════════════════════════════
        with _timer("gen_plain", timing_raw):
            self.envs.set_mode(plain=True)
            plain_output = self.traj_collector.multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=self.actor_rollout_wg,
                envs=self.envs,
                is_train=True,
            )

        # Capture reset_info for replaying same tasks in Phase 2
        reset_info = self.envs.get_last_reset_info()

        # Dump plain trajectories immediately (before balance_batch reorders)
        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
        if rollout_data_dir:
            plain_inputs = self.tokenizer.batch_decode(plain_output.batch["prompts"], skip_special_tokens=True)
            plain_outputs_text = self.tokenizer.batch_decode(plain_output.batch["responses"], skip_special_tokens=True)
            plain_scores = [float(x) for x in plain_output.non_tensor_batch['episode_rewards']]
            self._dump_generations(
                inputs=plain_inputs, outputs=plain_outputs_text, scores=plain_scores,
                reward_extra_infos_dict={}, dump_path=rollout_data_dir,
                traj_uids=plain_output.non_tensor_batch.get('traj_uid', None),
                uids=plain_output.non_tensor_batch.get('uid', None),
            )

        # ══════════════════════════════════════════════════════════════
        # Task-level pass rate computation (grouped by uid)
        # ══════════════════════════════════════════════════════════════
        episode_rewards = plain_output.non_tensor_batch['episode_rewards']
        group_uids = plain_output.non_tensor_batch['uid']
        traj_uids = plain_output.non_tensor_batch['traj_uid']

        # Build uid -> task_index mapping (stable ordinal based on first occurrence)
        uid_to_task_idx = {}
        for uid in group_uids:
            if uid not in uid_to_task_idx:
                uid_to_task_idx[uid] = len(uid_to_task_idx)

        # Deduplicate: get one reward per trajectory
        traj_reward = {}   # traj_uid -> reward
        traj_to_task = {}  # traj_uid -> task_index
        for g_uid, t_uid, r in zip(group_uids, traj_uids, episode_rewards):
            traj_reward[t_uid] = float(r)
            traj_to_task[t_uid] = uid_to_task_idx[g_uid]

        # Group trajectory rewards by task_index -> compute per-task pass_rate
        task_rewards = defaultdict(list)
        for t_uid, r in traj_reward.items():
            task_rewards[traj_to_task[t_uid]].append(r)

        n_total_tasks = len(task_rewards)
        task_pass_rates = {}
        for tidx, rewards in task_rewards.items():
            task_pass_rates[tidx] = sum(1 for r in rewards if r > 0) / len(rewards)

        # Batch-level pass rate
        batch_pass_rate = np.mean(list(task_pass_rates.values())) if task_pass_rates else 0.0

        # Non-zero task statistics for sliding window update
        non_zero_pass_rates = [pr for pr in task_pass_rates.values() if pr > 0]
        non_zero_mean = float(np.mean(non_zero_pass_rates)) if non_zero_pass_rates else 0.0

        # ══════════════════════════════════════════════════════════════
        # Three-tier routing (sliding-window mean)
        # ══════════════════════════════════════════════════════════════
        is_warmup = self.global_steps <= warmup_steps

        # Update sliding window with current step's non-zero mean
        if non_zero_pass_rates:
            self._routing_window.append(non_zero_mean)

        # Compute threshold from window (mean of recent non-zero means)
        if len(self._routing_window) > 0:
            self._routing_threshold = float(np.mean(list(self._routing_window)))
        else:
            self._routing_threshold = 0.5  # fallback before any data

        # Three-tier routing
        ema = self._routing_threshold
        hard_task_indices = {idx for idx, pr in task_pass_rates.items() if pr == 0}
        medium_task_indices = {idx for idx, pr in task_pass_rates.items() if 0 < pr <= ema}
        easy_task_indices = {idx for idx, pr in task_pass_rates.items() if pr > ema}

        # During warmup: merge easy into medium (hard stays independent)
        if is_warmup and easy_task_indices:
            medium_task_indices = medium_task_indices | easy_task_indices
            easy_task_indices = set()

        # Safety: if all tasks are hard (no medium/easy), fall back to all-medium
        # to ensure GRPO training signal is always available
        if len(medium_task_indices) == 0 and len(easy_task_indices) == 0:
            print(f"[Ours] All tasks hard (pr=0), falling back to all-medium for GRPO signal")
            medium_task_indices = hard_task_indices
            hard_task_indices = set()

        n_hard = len(hard_task_indices)
        n_medium = len(medium_task_indices)
        n_easy = len(easy_task_indices)

        metrics['routing/hard_ratio'] = n_hard / n_total_tasks if n_total_tasks > 0 else 0.0
        metrics['routing/medium_ratio'] = n_medium / n_total_tasks if n_total_tasks > 0 else 0.0
        metrics['routing/easy_ratio'] = n_easy / n_total_tasks if n_total_tasks > 0 else 0.0
        metrics['routing/threshold'] = self._routing_threshold
        metrics['routing/batch_pass_rate'] = batch_pass_rate
        metrics['routing/non_zero_mean'] = non_zero_mean
        metrics['routing/window_len'] = float(len(self._routing_window))
        metrics['routing/is_warmup'] = float(is_warmup)

        # Per-task pass rates sorted descending (e.g. "8/8, 7/8, 5/8, 0/8, ...")
        group_size = len(next(iter(task_rewards.values()))) if task_rewards else 8
        sorted_prs = sorted(task_pass_rates.values(), reverse=True)
        pr_str = ", ".join(f"{int(pr * group_size)}/{group_size}" for pr in sorted_prs)

        print(f"[Ours] Step {self.global_steps}: hard={n_hard}, medium={n_medium}, easy={n_easy}, "
              f"threshold={self._routing_threshold:.4f} (window={len(self._routing_window)}/{self._routing_window_size}), "
              f"non_zero_mean={non_zero_mean:.4f}, batch_pr={batch_pass_rate:.4f}, warmup={is_warmup}")
        print(f"[Ours] Step {self.global_steps} pass_rates: [{pr_str}]")

        # Phase 1 full-batch episode metrics (all tasks, not just medium/easy sub-batch)
        unique_traj_uids_p1, unique_idx_p1 = np.unique(plain_output.non_tensor_batch['traj_uid'], return_index=True)
        episode_rewards_p1 = plain_output.non_tensor_batch['episode_rewards'][unique_idx_p1]
        episode_lengths_p1 = plain_output.non_tensor_batch['episode_lengths'][unique_idx_p1]
        metrics['episode/reward/mean'] = float(episode_rewards_p1.mean())
        metrics['episode/reward/max'] = float(episode_rewards_p1.max())
        metrics['episode/reward/min'] = float(episode_rewards_p1.min())
        metrics['episode/length/mean'] = float(episode_lengths_p1.mean())
        metrics['episode/length/max'] = float(episode_lengths_p1.max())
        metrics['episode/length/min'] = float(episode_lengths_p1.min())
        for k, v in plain_output.non_tensor_batch.items():
            if "success_rate" in k:
                metrics[f'episode/{k}'] = float(v[0])

        # Per-step task index array (for selecting samples by tier)
        step_task_indices = np.array([uid_to_task_idx[uid] for uid in group_uids])

        # Extract unique uids (one per task) from Phase 1 for uid_base
        seen = set()
        uid_base = []
        for uid in group_uids:
            if uid not in seen:
                seen.add(uid)
                uid_base.append(uid)
        uid_base = np.array(uid_base, dtype=object)

        # ══════════════════════════════════════════════════════════════
        # Phase 2: Unified rollout (per-task mode: guided/noskill/plain)
        # Single multi_turn_loop replaces separate Phase 2a + Phase 2b.
        # ══════════════════════════════════════════════════════════════
        guided_r1_batch = None
        noskill_output = None
        phase2_needed = (n_hard > 0 or n_easy > 0)

        if phase2_needed:
            # Build per-task mode list (one entry per task)
            task_modes_base = []
            for tidx in range(n_total_tasks):
                if tidx in hard_task_indices:
                    task_modes_base.append('guided')
                elif tidx in easy_task_indices:
                    task_modes_base.append('noskill')
                else:
                    task_modes_base.append('plain')
            # Expand to match env batch (each task repeated env_rollout_n times, interleaved)
            task_modes = [m for m in task_modes_base for _ in range(env_rollout_n)]

            with _timer("gen_phase2", timing_raw):
                # Set mode: guide_internalize=True (needed for hard tasks to get dual text)
                self.envs.set_mode(plain=False)
                self.envs.set_per_task_mode(task_modes)
                phase2_output = self.traj_collector.multi_turn_loop(
                    gen_batch=gen_batch,
                    actor_rollout_wg=self.actor_rollout_wg,
                    envs=self.envs,
                    is_train=True,
                    reset_info=reset_info,
                    uid_base=uid_base,
                )
                self.envs.clear_per_task_mode()

            # Build uid -> task_index for Phase 2 output
            p2_group_uids = phase2_output.non_tensor_batch['uid']
            p2_traj_uids = phase2_output.non_tensor_batch['traj_uid']
            p2_rewards = phase2_output.non_tensor_batch['episode_rewards']

            p2_uid_to_task_idx = {}
            for uid in p2_group_uids:
                if uid not in p2_uid_to_task_idx:
                    p2_uid_to_task_idx[uid] = len(p2_uid_to_task_idx)

            p2_traj_reward = {}
            p2_traj_to_task = {}
            for g_uid, t_uid, r in zip(p2_group_uids, p2_traj_uids, p2_rewards):
                p2_traj_reward[t_uid] = float(r)
                p2_traj_to_task[t_uid] = p2_uid_to_task_idx[g_uid]

            # ── Extract hard task data (for JSD) ──
            if n_hard > 0:
                # R=1 mask: step belongs to hard task AND trajectory reward > 0
                guided_r1_mask = np.array([
                    (p2_traj_to_task.get(t_uid, -1) in hard_task_indices) and (float(r) > 0)
                    for t_uid, r in zip(p2_traj_uids, p2_rewards)
                ])
                guided_r1_idxs = np.where(guided_r1_mask)[0]

                # Metrics
                n_guided_total = sum(1 for t_uid in p2_traj_reward
                                     if p2_traj_to_task[t_uid] in hard_task_indices)
                n_guided_pass = sum(1 for t_uid, r in p2_traj_reward.items()
                                    if p2_traj_to_task[t_uid] in hard_task_indices and r > 0)
                n_guided_r1_steps = int(guided_r1_mask.sum())

                metrics['internalize/guided_pass_rate'] = n_guided_pass / n_guided_total if n_guided_total > 0 else 0.0
                metrics['internalize/guided_pass_count'] = n_guided_pass
                metrics['internalize/jsd_token_count'] = n_guided_r1_steps

                print(f"[Ours] Phase 2 (hard): guided_traj={n_guided_total}, "
                      f"R=1={n_guided_pass} (rate={metrics['internalize/guided_pass_rate']:.3f}), "
                      f"jsd_token_count={n_guided_r1_steps}")

                if n_guided_pass > 0:
                    guided_r1_batch = phase2_output.select_idxs(guided_r1_idxs)
            else:
                metrics['internalize/guided_pass_rate'] = 0.0
                metrics['internalize/guided_pass_count'] = 0
                metrics['internalize/jsd_token_count'] = 0

            # ── Extract easy task data (for contrastive) ──
            if n_easy > 0:
                # Select samples belonging to easy tasks
                p2_step_task_indices = np.array([p2_uid_to_task_idx[uid] for uid in p2_group_uids])
                easy_mask_p2 = np.array([idx in easy_task_indices for idx in p2_step_task_indices])
                easy_idxs_p2 = np.where(easy_mask_p2)[0]
                noskill_output = phase2_output.select_idxs(easy_idxs_p2)
                # Mark all as no_skill context_type
                noskill_output.non_tensor_batch['context_type'] = np.array(
                    ['no_skill'] * len(easy_idxs_p2), dtype=object)
                print(f"[Ours] Phase 2 (easy): noskill_output={len(noskill_output.batch['input_ids'])} samples")

            # ── Dump Phase 2 branch trajectories ──
            rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
            if rollout_data_dir:
                p2_step_task_indices_all = np.array([p2_uid_to_task_idx[uid] for uid in p2_group_uids])
                # Dump guided (hard) trajectories
                if n_hard > 0:
                    guided_mask_all = np.array([idx in hard_task_indices for idx in p2_step_task_indices_all])
                    guided_idxs_all = np.where(guided_mask_all)[0]
                    if len(guided_idxs_all) > 0:
                        guided_subset = phase2_output.select_idxs(guided_idxs_all)
                        guided_dump_path = os.path.join(rollout_data_dir, "guided")
                        guided_inputs = self.tokenizer.batch_decode(guided_subset.batch["prompts"], skip_special_tokens=True)
                        guided_outputs_text = self.tokenizer.batch_decode(guided_subset.batch["responses"], skip_special_tokens=True)
                        guided_scores = [float(x) for x in guided_subset.non_tensor_batch['episode_rewards']]
                        self._dump_generations(
                            inputs=guided_inputs,
                            outputs=guided_outputs_text,
                            scores=guided_scores,
                            reward_extra_infos_dict={},
                            dump_path=guided_dump_path,
                            traj_uids=guided_subset.non_tensor_batch['traj_uid'],
                            uids=guided_subset.non_tensor_batch['uid'],
                            extra_meta={"tier": "hard"},
                        )
                # Dump noskill (easy) trajectories
                if n_easy > 0 and noskill_output is not None:
                    noskill_dump_path = os.path.join(rollout_data_dir, "noskill")
                    noskill_inputs = self.tokenizer.batch_decode(noskill_output.batch["prompts"], skip_special_tokens=True)
                    noskill_outputs_text = self.tokenizer.batch_decode(noskill_output.batch["responses"], skip_special_tokens=True)
                    noskill_scores = [float(x) for x in noskill_output.non_tensor_batch['episode_rewards']]
                    self._dump_generations(
                        inputs=noskill_inputs,
                        outputs=noskill_outputs_text,
                        scores=noskill_scores,
                        reward_extra_infos_dict={},
                        dump_path=noskill_dump_path,
                        traj_uids=noskill_output.non_tensor_batch['traj_uid'],
                        uids=noskill_output.non_tensor_batch['uid'],
                        extra_meta={"tier": "easy"},
                    )

        else:
            metrics['internalize/guided_pass_rate'] = 0.0
            metrics['internalize/guided_pass_count'] = 0
            metrics['internalize/jsd_token_count'] = 0

        # Restore env mode
        self.envs.restore_mode()

        # ══════════════════════════════════════════════════════════════
        # Update 1: Easy tasks → Contrastive GRPO (independent step)
        # ══════════════════════════════════════════════════════════════
        reward_extra_infos_dict = {}
        batch = None  # will hold the "main" batch for return (use medium or easy)

        if n_easy > 0 and noskill_output is not None:
            # Select easy task samples from plain_output (Phase 1)
            easy_mask = np.array([idx in easy_task_indices for idx in step_task_indices])
            easy_idxs = np.where(easy_mask)[0]
            easy_skill_batch = plain_output.select_idxs(easy_idxs)

            # Remove extra keys from noskill_output that don't exist in easy_skill_batch
            # (Phase 2 may produce plain_* fields for guided tasks; noskill inherits them)
            extra_keys = set(noskill_output.batch.keys()) - set(easy_skill_batch.batch.keys())
            for k in extra_keys:
                del noskill_output.batch[k]

            # Merge skill (easy from Phase 1) + noskill (Phase 2)
            merged_easy = DataProto.concat([easy_skill_batch, noskill_output])
            merged_easy = adjust_batch(self.config, merged_easy)
            merged_easy.batch["response_mask"] = compute_response_mask(merged_easy)

            # Reward on merged batch
            with _timer("reward_easy", timing_raw):
                reward_tensor_easy, reward_extra_easy = compute_reward(merged_easy, self.reward_fn)
            merged_easy.batch["token_level_scores"] = reward_tensor_easy
            if reward_extra_easy:
                merged_easy.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_easy.items()})

            # Apply invalid action penalty
            if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                merged_easy, _ = apply_invalid_action_penalty(
                    merged_easy,
                    invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                )

            # token_level_rewards
            if self.config.algorithm.use_kl_in_reward:
                merged_easy, _ = apply_kl_penalty(merged_easy, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
            else:
                merged_easy.batch["token_level_rewards"] = merged_easy.batch["token_level_scores"]

            # Contrastive advantage (noskill_mean as baseline)
            contrastive_context_types = merged_easy.non_tensor_batch.get('context_type', None)
            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
            utilize_cfg = self.config.env.get('utilize', {})
            use_decomposed = utilize_cfg.get('decomposed_contrastive', False)
            contrastive_omega = utilize_cfg.get('omega', 1.0)
            use_ema_delta = utilize_cfg.get('use_ema_delta', False)
            adv2_clip = utilize_cfg.get('adv2_clip', 3.0)
            effective_rollout_n = rollout_n * 2
            # Ablation: disable_utilize forces omega=0 (no contrastive adv2, standard GRPO only)
            ablation_disable_utilize = ours_cfg.get('disable_utilize', False)
            # During warmup: disable adv2 (omega=0), only adv1 (standard GRPO per task)
            # EMA delta still accumulates but doesn't affect training
            if is_warmup or ablation_disable_utilize:
                effective_omega = 0.0
                ema_delta_value = None
            else:
                effective_omega = contrastive_omega
                # Pass delta baseline if enabled and available (None on first easy step → falls back to batch mode)
                if use_ema_delta:
                    if self._delta_baseline_mode == 'window' and len(self._delta_window) > 0:
                        ema_delta_value = float(np.mean(list(self._delta_window)))
                    elif self._delta_baseline_mode == 'ema' and self._ema_delta is not None:
                        ema_delta_value = self._ema_delta
                    else:
                        ema_delta_value = None
                else:
                    ema_delta_value = None
            merged_easy = compute_advantage(
                merged_easy,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=effective_rollout_n,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                use_pf_ppo=self.config.algorithm.use_pf_ppo,
                pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                gigpo_mode=self.config.algorithm.gigpo.mode,
                gigpo_enable_similarity=self.config.algorithm.gigpo.enable_similarity,
                gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                contrastive_context_types=contrastive_context_types,
                use_decomposed_contrastive=use_decomposed,
                contrastive_omega=effective_omega,
                ema_delta=ema_delta_value,
                adv2_clip=adv2_clip,
            )

            # Filter: keep only skill (top_k) samples for training
            phase1_mask = np.array([ct == 'top_k' for ct in merged_easy.non_tensor_batch['context_type']])
            phase1_idxs_easy = np.where(phase1_mask)[0]
            easy_batch = merged_easy.select_idxs(phase1_idxs_easy)
            easy_batch.meta_info = merged_easy.meta_info.copy()
            easy_batch.meta_info["global_token_num"] = torch.sum(easy_batch.batch["attention_mask"], dim=-1).tolist()

            # Pad to divisible by world_size
            world_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
            bs_easy = len(easy_batch)
            remainder = bs_easy % world_size
            if remainder != 0:
                to_add = world_size - remainder
                dup_indices = np.random.choice(bs_easy, to_add, replace=(to_add > bs_easy))
                dup_proto = easy_batch.select_idxs(dup_indices)
                dup_proto.batch["advantages"] = torch.zeros_like(dup_proto.batch["advantages"])
                dup_proto.batch["response_mask"] = torch.zeros_like(dup_proto.batch["response_mask"])
                if "loss_mask" in dup_proto.batch:
                    dup_proto.batch["loss_mask"] = torch.zeros_like(dup_proto.batch["loss_mask"])
                easy_batch = DataProto.concat([easy_batch, dup_proto])
                easy_batch.meta_info["global_token_num"] = torch.sum(easy_batch.batch["attention_mask"], dim=-1).tolist()

            # Contrastive metrics
            if contrastive_context_types is not None:
                episode_rewards_arr = merged_easy.non_tensor_batch.get('episode_rewards', None)
                bs_merged = len(merged_easy.batch['token_level_scores'])
                traj_uids_merged = merged_easy.non_tensor_batch.get('traj_uid', None)
                seen_trajs = {}
                for i in range(bs_merged):
                    uid = traj_uids_merged[i] if traj_uids_merged is not None else i
                    if uid not in seen_trajs:
                        is_succ = float(episode_rewards_arr[i]) > 0 if episode_rewards_arr is not None else False
                        seen_trajs[uid] = (contrastive_context_types[i], is_succ)
                n_top_k_success = sum(1 for ct, s in seen_trajs.values() if ct == 'top_k' and s)
                n_top_k_trajs = sum(1 for ct, _ in seen_trajs.values() if ct == 'top_k')
                n_no_skill_success = sum(1 for ct, s in seen_trajs.values() if ct == 'no_skill' and s)
                n_no_skill_trajs = sum(1 for ct, _ in seen_trajs.values() if ct == 'no_skill')
                metrics['utilize/skill_success_rate'] = n_top_k_success / n_top_k_trajs if n_top_k_trajs > 0 else 0.0
                metrics['utilize/noskill_success_rate'] = n_no_skill_success / n_no_skill_trajs if n_no_skill_trajs > 0 else 0.0

                # Per-task delta: mean(skill_pass_rate - noskill_pass_rate) across easy tasks
                group_uids_merged = merged_easy.non_tensor_batch.get('uid', None)
                if group_uids_merged is not None:
                    task_skill_rewards = defaultdict(list)
                    task_noskill_rewards = defaultdict(list)
                    for i, uid in enumerate(traj_uids_merged):
                        if uid in seen_trajs:
                            ct, is_succ = seen_trajs[uid]
                            g_uid = group_uids_merged[i]
                            if ct == 'top_k':
                                task_skill_rewards[g_uid].append(float(is_succ))
                            elif ct == 'no_skill':
                                task_noskill_rewards[g_uid].append(float(is_succ))

                    # Compute per-task delta
                    task_deltas = []
                    for g_uid in task_skill_rewards:
                        skill_pr = np.mean(task_skill_rewards[g_uid])
                        noskill_pr = np.mean(task_noskill_rewards[g_uid]) if g_uid in task_noskill_rewards else 0.0
                        task_deltas.append(skill_pr - noskill_pr)

                    if task_deltas:
                        mean_delta = float(np.mean(task_deltas))
                        metrics['utilize/delta'] = mean_delta
                        # Update delta baseline (after computing adv2 with old value)
                        # Always update both EMA and window so switching mode mid-run is seamless
                        ema_delta_alpha = utilize_cfg.get('ema_delta_alpha', 0.1)
                        if self._ema_delta is None:
                            self._ema_delta = mean_delta
                        else:
                            self._ema_delta = ema_delta_alpha * mean_delta + (1 - ema_delta_alpha) * self._ema_delta
                        self._delta_window.append(mean_delta)
                        # Log the effective baseline value
                        if self._delta_baseline_mode == 'window' and len(self._delta_window) > 0:
                            metrics['utilize/delta_baseline'] = float(np.mean(list(self._delta_window)))
                        else:
                            metrics['utilize/delta_baseline'] = self._ema_delta
                        metrics['utilize/ema_delta'] = self._ema_delta

            # Old log probs + ref log probs
            with _timer("old_log_prob_easy", timing_raw):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(easy_batch)
                entropys = old_log_prob.batch["entropys"]
                # Compute entropy metric on real samples only (exclude padding)
                entropys_real = entropys[:bs_easy]
                response_masks_real = easy_batch.batch["response_mask"][:bs_easy]
                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                entropy_loss = agg_loss(loss_mat=entropys_real, loss_mask=response_masks_real, loss_agg_mode=loss_agg_mode)
                metrics["actor/entropy_loss_easy"] = entropy_loss.detach().item()
                old_log_prob.batch.pop("entropys")
                easy_batch = easy_batch.union(old_log_prob)

            if self.use_reference_policy:
                with _timer("ref_easy", timing_raw):
                    if not self.ref_in_actor:
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(easy_batch)
                    else:
                        ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(easy_batch)
                    easy_batch = easy_batch.union(ref_log_prob)

            # Update actor (independent step)
            if self.config.trainer.critic_warmup <= self.global_steps:
                with _timer("update_actor_easy", timing_raw):
                    easy_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    easy_batch.meta_info["hdpo_mode"] = "grpo"
                    easy_output = self.actor_rollout_wg.update_actor(easy_batch)
                easy_output_metrics = reduce_metrics(easy_output.meta_info["metrics"])
                if "actor/grad_norm" in easy_output_metrics:
                    easy_output_metrics["actor/grad_norm_easy"] = easy_output_metrics.pop("actor/grad_norm")
                if "actor/pg_loss" in easy_output_metrics:
                    easy_output_metrics["actor/grpo_loss_easy"] = easy_output_metrics.pop("actor/pg_loss")
                if "actor/kl_loss" in easy_output_metrics:
                    easy_output_metrics["actor/kl_loss_easy"] = easy_output_metrics.pop("actor/kl_loss")
                metrics.update(easy_output_metrics)

            print(f"[Ours] Update 1 (easy/utilize): {bs_easy} samples")
            batch = easy_batch  # use as return batch if no medium


        # ══════════════════════════════════════════════════════════════
        # Update 2: Medium tasks → Standard GRPO (independent step)
        # ══════════════════════════════════════════════════════════════
        if n_medium > 0:
            medium_mask = np.array([idx in medium_task_indices for idx in step_task_indices])
            medium_idxs = np.where(medium_mask)[0]
            medium_batch = plain_output.select_idxs(medium_idxs)
            bs_medium_real = len(medium_batch)

            medium_batch = adjust_batch(self.config, medium_batch)
            # Mark padding
            is_padding = np.zeros(len(medium_batch), dtype=bool)
            is_padding[bs_medium_real:] = True
            medium_batch.non_tensor_batch['_is_padding'] = is_padding
            medium_batch.batch["response_mask"] = compute_response_mask(medium_batch)

            # Zero out padding masks immediately so they never contribute to
            # entropy/kl gradients or metrics (dp_actor uses loss_mask in multi_turn)
            n_padding = int(is_padding.sum())
            if n_padding > 0:
                padding_indices = torch.tensor(np.where(is_padding)[0], dtype=torch.long)
                medium_batch.batch["response_mask"][padding_indices] = 0.0
                if "loss_mask" in medium_batch.batch:
                    medium_batch.batch["loss_mask"][padding_indices] = 0.0

            if self.config.trainer.balance_batch:
                self._balance_batch(medium_batch, metrics=metrics)

            medium_batch.meta_info["global_token_num"] = torch.sum(
                medium_batch.batch["attention_mask"], dim=-1).tolist()

            # Reward
            with _timer("reward_medium", timing_raw):
                reward_tensor_med, reward_extra_med = compute_reward(medium_batch, self.reward_fn)
            medium_batch.batch["token_level_scores"] = reward_tensor_med
            if reward_extra_med:
                medium_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_med.items()})
            if not reward_extra_infos_dict:
                reward_extra_infos_dict = reward_extra_med

            # Apply invalid action penalty
            if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                medium_batch, invalid_metrics = apply_invalid_action_penalty(
                    medium_batch,
                    invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef)
                # Metrics from real samples only
                padding_mask = medium_batch.non_tensor_batch['_is_padding']
                if 'valid_actions' in medium_batch.non_tensor_batch:
                    real_valid = medium_batch.non_tensor_batch['valid_actions'][~padding_mask]
                    invalid_metrics['episode/valid_action_ratio'] = float(np.mean(real_valid))
                metrics.update(invalid_metrics)

            # KL / token_level_rewards
            if self.config.algorithm.use_kl_in_reward:
                medium_batch, kl_metrics = apply_kl_penalty(
                    medium_batch, kl_ctrl=self.kl_ctrl_in_reward,
                    kl_penalty=self.config.algorithm.kl_penalty)
                metrics.update(kl_metrics)
            else:
                medium_batch.batch["token_level_rewards"] = medium_batch.batch["token_level_scores"]

            # Old log probs
            with _timer("old_log_prob_medium", timing_raw):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(medium_batch)
                entropys = old_log_prob.batch["entropys"]
                # Compute entropy metric on real samples only (exclude padding)
                real_mask_bool = ~medium_batch.non_tensor_batch['_is_padding']
                real_indices = torch.tensor(np.where(real_mask_bool)[0], dtype=torch.long)
                entropys_real = entropys[real_indices]
                response_masks_real = medium_batch.batch["response_mask"][real_indices]
                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                entropy_loss = agg_loss(loss_mat=entropys_real, loss_mask=response_masks_real, loss_agg_mode=loss_agg_mode)
                metrics["actor/entropy_loss_medium"] = entropy_loss.detach().item()
                old_log_prob.batch.pop("entropys")
                medium_batch = medium_batch.union(old_log_prob)

            # Ref log probs
            if self.use_reference_policy:
                with _timer("ref_medium", timing_raw):
                    if not self.ref_in_actor:
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(medium_batch)
                    else:
                        ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(medium_batch)
                    medium_batch = medium_batch.union(ref_log_prob)

            # Advantage
            with _timer("adv_medium", timing_raw):
                norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                medium_batch = compute_advantage(
                    medium_batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=rollout_n,
                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                    multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                    use_pf_ppo=self.config.algorithm.use_pf_ppo,
                    pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                    pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                    step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                    gigpo_mode=self.config.algorithm.gigpo.mode,
                    gigpo_enable_similarity=self.config.algorithm.gigpo.enable_similarity,
                    gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                )

            # Zero out padding advantages (response_mask/loss_mask already zeroed above)
            padding_mask = medium_batch.non_tensor_batch.get('_is_padding', np.zeros(len(medium_batch), dtype=bool))
            n_padding = int(padding_mask.sum())
            if n_padding > 0:
                padding_indices = torch.tensor(np.where(padding_mask)[0], dtype=torch.long)
                medium_batch.batch["advantages"][padding_indices] = 0.0

            # Update actor (independent step)
            if self.config.trainer.critic_warmup <= self.global_steps:
                with _timer("update_actor_medium", timing_raw):
                    medium_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                    medium_batch.meta_info["global_token_num"] = torch.sum(
                        medium_batch.batch["attention_mask"], dim=-1).tolist()
                    medium_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    medium_batch.meta_info["hdpo_mode"] = "grpo"
                    medium_output = self.actor_rollout_wg.update_actor(medium_batch)
                medium_output_metrics = reduce_metrics(medium_output.meta_info["metrics"])
                if "actor/grad_norm" in medium_output_metrics:
                    medium_output_metrics["actor/grad_norm_medium"] = medium_output_metrics.pop("actor/grad_norm")
                if "actor/pg_loss" in medium_output_metrics:
                    medium_output_metrics["actor/grpo_loss_medium"] = medium_output_metrics.pop("actor/pg_loss")
                if "actor/kl_loss" in medium_output_metrics:
                    medium_output_metrics["actor/kl_loss_medium"] = medium_output_metrics.pop("actor/kl_loss")
                metrics.update(medium_output_metrics)

            print(f"[Ours] Update 2 (medium/grpo): {bs_medium_real} real + {n_padding} padding samples")
            batch = medium_batch  # prefer medium as return batch

        # ══════════════════════════════════════════════════════════════
        # Update 3: Hard tasks → Standard GRPO (format signal)
        # Even all-fail tasks get gradient from invalid_action_penalty
        # differences (score=0 vs score=-0.1 across steps).
        # ══════════════════════════════════════════════════════════════
        if n_hard > 0:
            hard_mask = np.array([idx in hard_task_indices for idx in step_task_indices])
            hard_idxs = np.where(hard_mask)[0]
            hard_batch = plain_output.select_idxs(hard_idxs)
            bs_hard_real = len(hard_batch)

            hard_batch = adjust_batch(self.config, hard_batch)
            # Mark padding
            is_padding_hard = np.zeros(len(hard_batch), dtype=bool)
            is_padding_hard[bs_hard_real:] = True
            hard_batch.non_tensor_batch['_is_padding'] = is_padding_hard
            hard_batch.batch["response_mask"] = compute_response_mask(hard_batch)

            # Zero out padding masks
            n_padding_hard = int(is_padding_hard.sum())
            if n_padding_hard > 0:
                padding_indices_hard = torch.tensor(np.where(is_padding_hard)[0], dtype=torch.long)
                hard_batch.batch["response_mask"][padding_indices_hard] = 0.0
                if "loss_mask" in hard_batch.batch:
                    hard_batch.batch["loss_mask"][padding_indices_hard] = 0.0

            hard_batch.meta_info["global_token_num"] = torch.sum(
                hard_batch.batch["attention_mask"], dim=-1).tolist()

            # Reward
            with _timer("reward_hard", timing_raw):
                reward_tensor_hard, reward_extra_hard = compute_reward(hard_batch, self.reward_fn)
            hard_batch.batch["token_level_scores"] = reward_tensor_hard
            if reward_extra_hard:
                hard_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_hard.items()})

            # Apply invalid action penalty (this creates the reward variance for GRPO)
            if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                hard_batch, _ = apply_invalid_action_penalty(
                    hard_batch,
                    invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef)

            # token_level_rewards
            if self.config.algorithm.use_kl_in_reward:
                hard_batch, _ = apply_kl_penalty(
                    hard_batch, kl_ctrl=self.kl_ctrl_in_reward,
                    kl_penalty=self.config.algorithm.kl_penalty)
            else:
                hard_batch.batch["token_level_rewards"] = hard_batch.batch["token_level_scores"]

            # Old log probs
            with _timer("old_log_prob_hard", timing_raw):
                old_log_prob_hard = self.actor_rollout_wg.compute_log_prob(hard_batch)
                old_log_prob_hard.batch.pop("entropys")
                hard_batch = hard_batch.union(old_log_prob_hard)

            # Ref log probs
            if self.use_reference_policy:
                with _timer("ref_hard", timing_raw):
                    if not self.ref_in_actor:
                        ref_log_prob_hard = self.ref_policy_wg.compute_ref_log_prob(hard_batch)
                    else:
                        ref_log_prob_hard = self.actor_rollout_wg.compute_ref_log_prob(hard_batch)
                    hard_batch = hard_batch.union(ref_log_prob_hard)

            # Advantage (standard GRPO, task-level z-score)
            with _timer("adv_hard", timing_raw):
                norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                hard_batch = compute_advantage(
                    hard_batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=rollout_n,
                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                    multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                    use_pf_ppo=self.config.algorithm.use_pf_ppo,
                    pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                    pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                    step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                    gigpo_mode=self.config.algorithm.gigpo.mode,
                    gigpo_enable_similarity=self.config.algorithm.gigpo.enable_similarity,
                    gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                )

            # Zero out padding advantages
            if n_padding_hard > 0:
                hard_batch.batch["advantages"][padding_indices_hard] = 0.0

            # Update actor (independent step)
            if self.config.trainer.critic_warmup <= self.global_steps:
                with _timer("update_actor_hard", timing_raw):
                    hard_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                    hard_batch.meta_info["global_token_num"] = torch.sum(
                        hard_batch.batch["attention_mask"], dim=-1).tolist()
                    hard_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    hard_batch.meta_info["hdpo_mode"] = "grpo"
                    hard_output = self.actor_rollout_wg.update_actor(hard_batch)
                hard_output_metrics = reduce_metrics(hard_output.meta_info["metrics"])
                if "actor/grad_norm" in hard_output_metrics:
                    hard_output_metrics["actor/grad_norm_hard"] = hard_output_metrics.pop("actor/grad_norm")
                if "actor/pg_loss" in hard_output_metrics:
                    hard_output_metrics["actor/grpo_loss_hard"] = hard_output_metrics.pop("actor/pg_loss")
                if "actor/kl_loss" in hard_output_metrics:
                    hard_output_metrics["actor/kl_loss_hard"] = hard_output_metrics.pop("actor/kl_loss")
                metrics.update(hard_output_metrics)

            print(f"[Ours] Update 3 (hard/grpo): {bs_hard_real} real + {n_padding_hard} padding samples")
            if batch is None:
                batch = hard_batch

        # ══════════════════════════════════════════════════════════════
        # Update 4: Hard tasks → JSD (independent step, if guided succeeded)
        # ══════════════════════════════════════════════════════════════
        if guided_r1_batch is not None and len(guided_r1_batch.batch['input_ids']) > 0:
            jsd_batch = guided_r1_batch
            world_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
            jsd_divisor = self.config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu * world_size
            bs_jsd_real = len(jsd_batch)
            remainder = bs_jsd_real % jsd_divisor
            if remainder != 0:
                to_add = jsd_divisor - remainder
                dup_indices = np.random.choice(bs_jsd_real, to_add, replace=(to_add > bs_jsd_real))
                dup_proto = jsd_batch.select_idxs(dup_indices)
                jsd_batch = DataProto.concat([jsd_batch, dup_proto])
            jsd_batch.batch["response_mask"] = compute_response_mask(jsd_batch)
            # Zero out padding masks
            if len(jsd_batch) > bs_jsd_real:
                jsd_batch.batch["response_mask"][bs_jsd_real:] = 0.0
                if "loss_mask" in jsd_batch.batch:
                    jsd_batch.batch["loss_mask"][bs_jsd_real:] = 0.0

            # Update actor (independent JSD step)
            # Ablation: disable_jsd skips the actual update but keeps all metrics above
            ablation_disable_jsd = ours_cfg.get('disable_jsd', False)
            if self.config.trainer.critic_warmup <= self.global_steps and not ablation_disable_jsd:
                with _timer("update_actor_jsd", timing_raw):
                    jsd_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                    jsd_batch.meta_info["global_token_num"] = torch.sum(
                        jsd_batch.batch["attention_mask"], dim=-1).tolist()
                    jsd_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    jsd_batch.meta_info["hdpo_mode"] = "jsd"
                    jsd_batch.meta_info["hdpo_config"] = {
                        'jsd_lambda': jsd_lambda,
                        'jsd_top_k': jsd_top_k,
                        'jsd_temperature': jsd_temperature,
                    }
                    jsd_output = self.actor_rollout_wg.update_actor(jsd_batch)
                jsd_output_metrics = reduce_metrics(jsd_output.meta_info["metrics"])
                if "actor/grad_norm" in jsd_output_metrics:
                    jsd_output_metrics["actor/grad_norm_jsd"] = jsd_output_metrics.pop("actor/grad_norm")
                metrics.update(jsd_output_metrics)

            print(f"[Ours] Update 4 (hard/jsd): {bs_jsd_real} real samples"
                  f"{' [ABLATION: JSD disabled]' if ablation_disable_jsd else ''}")

        # ══════════════════════════════════════════════════════════════
        # Fallback: if no batch set (degenerate edge case)
        # ══════════════════════════════════════════════════════════════
        if batch is None:
            # Use plain_output as-is for return value (no GRPO update happened)
            print("[Ours] WARNING: No medium/easy samples. Using plain_output as fallback batch.")
            batch = plain_output
            batch = adjust_batch(self.config, batch)
            batch.batch["response_mask"] = compute_response_mask(batch)
            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
            with _timer("reward_fallback", timing_raw):
                reward_tensor_fb, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
            batch.batch["token_level_scores"] = reward_tensor_fb
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
            # Fill advantages with zeros (no GRPO update happened, but fit() needs this field)
            batch.batch["advantages"] = torch.zeros_like(batch.batch["response_mask"])

        return batch, reward_extra_infos_dict


    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        # breakpoint()
        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            if self.val_envs_ood is not None:
                val_ood_metrics = self._validate_ood()
                val_metrics.update(val_ood_metrics)
                pprint(f"Initial OOD validation metrics: {val_ood_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                # Save merged metrics (val + val_ood) to val_dump_path
                # Merge strategy: per-domain metrics (different names) go under val/
                # Aggregate metrics (success_rate, test_score) that exist in both
                # val/ and val_ood/ are kept as val/id/X and val/ood/X, with a
                # weighted val/X computed from both.
                val_dump_path = self.config.trainer.get("val_dump_path", None)
                if val_dump_path:
                    os.makedirs(val_dump_path, exist_ok=True)

                    # Identify conflicting keys (exist in both val/ and val_ood/)
                    val_keys = {k for k in val_metrics if k.startswith('val/') and not k.startswith('val_ood/')}
                    ood_keys = {k for k in val_metrics if k.startswith('val_ood/')}
                    ood_as_val = {k.replace('val_ood/', 'val/'): k for k in ood_keys}
                    conflicting = val_keys & set(ood_as_val.keys())

                    normalized = {}
                    for k, v in val_metrics.items():
                        if k.startswith('val_ood/'):
                            new_key = k.replace('val_ood/', 'val/')
                            if new_key in conflicting:
                                # Conflicting aggregate: store under val/ood/
                                normalized[k.replace('val_ood/', 'val/ood/')] = v
                            else:
                                normalized[new_key] = v
                        elif k in conflicting:
                            # ID side of conflicting aggregate: store under val/id/
                            normalized[k.replace('val/', 'val/id/')] = v
                        else:
                            normalized[k] = v

                    # Compute weighted overall success_rate / test_score
                    id_sr = val_metrics.get('val/success_rate')
                    ood_sr = val_metrics.get('val_ood/success_rate')
                    id_ts = val_metrics.get('val/text/test_score')
                    ood_ts = val_metrics.get('val_ood/text/test_score')
                    n_id = val_metrics.get('val/num_trajs', 0)
                    n_ood = val_metrics.get('val_ood/num_trajs', 0)

                    if id_sr is not None and ood_sr is not None and n_id + n_ood > 0:
                        normalized['val/success_rate'] = (id_sr * n_id + ood_sr * n_ood) / (n_id + n_ood)
                    if id_ts is not None and ood_ts is not None and n_id + n_ood > 0:
                        normalized['val/text/test_score'] = (id_ts * n_id + ood_ts * n_ood) / (n_id + n_ood)

                    metrics_file = os.path.join(val_dump_path, "metrics.json")
                    with open(metrics_file, "w") as f:
                        json.dump(normalized, f, indent=2, ensure_ascii=False)
                    print(f"Saved merged validation metrics to {metrics_file}")
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                dual_distill_mode = self.config.env.get('dual_distill_mode', False)
                ours_mode = self.config.env.get('ours_mode', False)
                internalize_mode = self.config.env.get('internalize_mode', False)
                utilize_mode = self.config.env.get('utilize_mode', False)

                with _timer("step", timing_raw):
                  if dual_distill_mode:
                    # ═══════════ Unified GRPO + Dual Adaptive Distillation ═══════════
                    batch, reward_extra_infos_dict = self._dual_distill_step(
                        gen_batch, timing_raw, metrics)
                  elif ours_mode:
                    # ═══════════ Three-tier adaptive routing ═══════════
                    batch, reward_extra_infos_dict = self._ours_step(
                        gen_batch, timing_raw, metrics)
                  elif internalize_mode:
                    # ═══════════ Internalize only (JSD distillation) ═══════════
                    batch, reward_extra_infos_dict = self._internalize_step(
                        gen_batch, timing_raw, metrics)
                  elif utilize_mode:
                    # ═══════════ Utilize only (contrastive GRPO) ═══════════
                    batch, reward_extra_infos_dict = self._utilize_step(
                        gen_batch, timing_raw, metrics)
                  else:
                    raise ValueError(
                        "Skill0.5 requires one of dual_distill_mode, ours_mode, internalize_mode, or utilize_mode to be True. "
                        "For baseline GRPO, use SkillRL directly.")

                  # Note: plain rollout dump is now done inside each _*_step method
                  # (before balance_batch reorders data), so we skip the post-update dump here.

                  # validate
                  if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                      with _timer("testing", timing_raw):
                          val_metrics: dict = self._validate()
                          if is_last_step:
                              last_val_metrics = val_metrics
                      metrics.update(val_metrics)
                      # OOD validation
                      if self.val_envs_ood is not None:
                          with _timer("testing_ood", timing_raw):
                              val_ood_metrics = self._validate_ood()
                          metrics.update(val_ood_metrics)

                  if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                      with _timer("save_checkpoint", timing_raw):
                          self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                # Preserve Phase 1 full-batch episode metrics (set in _ours_step) before
                # compute_data_metrics overwrites them with sub-batch values
                episode_keys_override = {k: v for k, v in metrics.items() if k.startswith("episode/")} if ours_mode else {}
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                if episode_keys_override:
                    metrics.update(episode_keys_override)
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

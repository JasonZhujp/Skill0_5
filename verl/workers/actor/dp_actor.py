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
Single Process Actor
"""

import itertools
import json
import time
import logging
import os
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_jsd_loss, compute_policy_loss, compute_policy_loss_gspo, compute_policy_loss_with_guide_reshaping, compute_policy_loss_with_luffy_shaping, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, ulysses_pad
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _forward_micro_batch_logits(self, micro_batch, temperature) -> torch.Tensor:
        """Forward pass that returns full logits for response tokens (for JSD computation).

        Returns:
            logits: (bs, response_length, vocab_size)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad, position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size)
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad, position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size)

                output = self.actor_module(
                    input_ids=input_ids_rmpad, attention_mask=None,
                    position_ids=position_ids_rmpad, **multi_modal_inputs,
                    use_cache=False)

                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                logits_rmpad.div_(temperature)

                if self.use_ulysses_sp:
                    # For logits we need vocab_size dim — gather across SP ranks
                    logits_rmpad = gather_outpus_and_unpad(
                        logits_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # pad back to (bsz, seqlen, vocab_size)
                full_logits = pad_input(
                    hidden_states=logits_rmpad, indices=indices,
                    batch=batch_size, seqlen=seqlen)
                # response part only
                logits = full_logits[:, -response_length - 1: -1, :]  # (bs, resp_len, V)

            else:
                output = self.actor_module(
                    input_ids=input_ids, attention_mask=attention_mask,
                    position_ids=position_ids, **multi_modal_inputs,
                    use_cache=False)
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1: -1, :]  # (bs, resp_len, V)

            return logits

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        # ==================== HDPO Mode ====================
        hdpo_mode = data.meta_info.get("hdpo_mode", None)
        if hdpo_mode == "jsd":
            hdpo_cfg = data.meta_info["hdpo_config"]
            return self._update_policy_hdpo_jsd(data, hdpo_cfg)
        elif hdpo_mode == "grpo":
            # GRPO path: falls through to standard update_policy below
            pass

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        # Internalize prob dump config (set by trainer when guide is enabled)
        internalize_dump_path = data.meta_info.get("internalize_dump_path", None)
        internalize_global_step = data.meta_info.get("global_step", None)
        _internalize_dumped = False  # only dump once per update_policy call

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        guide_cfg = self.config.policy_loss.get("guide", {})
        guide_active = guide_cfg.get("enabled", False)  # config says guide mode is on
        guide_warmup_steps = guide_cfg.get("warmup_steps", 0)
        guide_use_reshaping = guide_cfg.get("use_reshaping", True)  # False = always use PPO clip (no reshaping)
        current_step = internalize_global_step if internalize_global_step is not None else 0
        guide_loss_enabled = guide_active and guide_use_reshaping and (current_step >= guide_warmup_steps)  # reshaping loss active
        is_warmup = guide_active and (not guide_loss_enabled)
        if guide_active:
            select_keys.extend(["plain_input_ids", "plain_attention_mask", "plain_position_ids"])

        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        # During guide warmup, use 1 epoch (standard on-policy); internalization phase uses configured ppo_epochs
        effective_ppo_epochs = 1 if is_warmup else self.config.ppo_epochs
        for epoch in range(effective_ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True

                    if guide_active and not is_warmup:
                        # Forward on plain prompt for all sequences (internalization target)
                        plain_micro = {**data}
                        plain_micro["input_ids"] = data["plain_input_ids"]
                        plain_micro["attention_mask"] = data["plain_attention_mask"]
                        plain_micro["position_ids"] = data["plain_position_ids"]
                        entropy, log_prob = self._forward_micro_batch(micro_batch=plain_micro, temperature=temperature, calculate_entropy=calculate_entropy)
                        log_prob_plain = log_prob  # alias for metrics
                    else:
                        entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                    if guide_loss_enabled:
                        # Guide mode: shaping for all sequences (luffy or guide-reshaping)
                        guide_shaping_fn = guide_cfg.get("shaping_fn", "guide")
                        reshaping_quantile = self.config.policy_loss.guide.get("reshaping_quantile", 0.9)
                        luffy_gamma = self.config.policy_loss.guide.get("luffy_gamma", 0.2)
                        if guide_shaping_fn == "luffy":
                            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss_with_luffy_shaping(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask,
                                cliprange=clip_ratio,
                                cliprange_low=clip_ratio_low,
                                cliprange_high=clip_ratio_high,
                                clip_ratio_c=clip_ratio_c,
                                loss_agg_mode=loss_agg_mode,
                                gamma=luffy_gamma,
                            )
                        elif guide_shaping_fn == "none":
                            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask,
                                cliprange=clip_ratio,
                                cliprange_low=clip_ratio_low,
                                cliprange_high=clip_ratio_high,
                                clip_ratio_c=clip_ratio_c,
                                loss_agg_mode=loss_agg_mode,
                            )
                        else:
                            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss_with_guide_reshaping(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask,
                                cliprange=clip_ratio,
                                cliprange_low=clip_ratio_low,
                                cliprange_high=clip_ratio_high,
                                clip_ratio_c=clip_ratio_c,
                                loss_agg_mode=loss_agg_mode,
                                reshaping_quantile=reshaping_quantile,
                            )
                    else:
                        if loss_mode == "vanilla":
                            policy_loss_fn = compute_policy_loss
                        elif loss_mode == "gspo":
                            policy_loss_fn = compute_policy_loss_gspo
                        else:
                            raise ValueError(f"Unsupported loss_mode: {loss_mode}")

                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                    }

                    data["actor/pg_clipfrac"] = pg_clipfrac.detach().item()
                    data["actor/pg_clipfrac_lower"] = pg_clipfrac_lower.detach().item()

                    if guide_active and not is_warmup:
                        with torch.no_grad():
                            # Log-space diff: log π_θ(plain) - log π_θold(guided)
                            # This is the true signal of off-policy gap (negative = not yet internalized)
                            log_ratio = log_prob_plain - old_log_prob  # (bs, resp_len)
                            valid_log_ratio = log_ratio[response_mask.bool()]
                            data["actor/guide_log_ratio_mean"] = valid_log_ratio.mean().item()
                            data["actor/guide_log_ratio_median"] = valid_log_ratio.median().item()
                            data["actor/guide_log_ratio_min"] = valid_log_ratio.min().item()
                            data["actor/guide_log_ratio_max"] = valid_log_ratio.max().item()
                            # Ratio-space stats (for compatibility / comparison)
                            guide_ratio = torch.exp(log_ratio)
                            valid_ratio = guide_ratio[response_mask.bool()]
                            data["actor/guide_ratio_mean"] = valid_ratio.mean().item()
                            # P90 — the reshaping anchor used in loss
                            bs = guide_ratio.shape[0]
                            p90_vals = []
                            for i in range(bs):
                                valid = guide_ratio[i][response_mask[i].bool()]
                                if len(valid) > 0:
                                    p90_vals.append(torch.quantile(valid.float(), 0.9).item())
                            data["actor/guide_ratio_p90"] = sum(p90_vals) / max(len(p90_vals), 1)
                            data["actor/guide_is_warmup"] = 1.0 if is_warmup else 0.0

                            # Dump token-level prob comparison for internalization analysis
                            if internalize_dump_path and not _internalize_dumped:
                                _internalize_dumped = True
                                try:
                                    os.makedirs(internalize_dump_path, exist_ok=True)
                                    dump_file = os.path.join(internalize_dump_path, f"step_{internalize_global_step}.jsonl")
                                    with open(dump_file, "w") as f:
                                        for i in range(bs):
                                            mask_i = response_mask[i].bool()
                                            valid_len = mask_i.sum().item()
                                            record = {
                                                "sample_idx": i,
                                                "is_warmup": is_warmup,
                                                "token_ids": responses[i][:valid_len].cpu().tolist(),
                                                "log_prob_plain": log_prob_plain[i][:valid_len].cpu().tolist(),
                                                "log_prob_guided": old_log_prob[i][:valid_len].cpu().tolist(),
                                                "advantages": advantages[i][:valid_len].cpu().tolist(),
                                            }
                                            f.write(json.dumps(record) + "\n")
                                    logger.info(f"[Internalize] Saved token-level probs to {dump_file} ({bs} samples, warmup={is_warmup})")
                                except Exception as e:
                                    logger.warning(f"[Internalize] Failed to save probs: {e}")

                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics

    def _update_policy_hdpo_jsd(self, data: DataProto, hdpo_cfg: dict):
        """JSD distillation update: all samples are cliff R=1, use JSD (guided→plain).

        Called as a separate update_actor pass so all GPUs execute the same operations
        (avoids FSDP deadlock from asymmetric forward passes).
        Each call is an independent optimizer cycle (zero_grad → backward → clip → step).

        Args:
            data: DataProto with JSD samples only (cliff R=1 trajectories)
            hdpo_cfg: dict with keys: jsd_lambda, jsd_top_k, jsd_temperature
        """
        multi_turn = data.meta_info.get("multi_turn", False)
        jsd_lambda = hdpo_cfg.get("jsd_lambda", 1.0)
        jsd_top_k = hdpo_cfg.get("jsd_top_k", 64)
        jsd_temperature = hdpo_cfg.get("jsd_temperature", 1.0)

        jsd_keys = ["responses", "input_ids", "attention_mask", "position_ids",
                     "plain_input_ids", "plain_attention_mask", "plain_position_ids"]
        if multi_turn:
            jsd_keys.append("loss_mask")

        metrics = {}
        self.actor_module.train()

        jsd_batch = data.select(batch_keys=jsd_keys).batch
        # JSD uses a single mini-batch (all samples) to avoid over-updating on few R=1 samples.
        # gradient_accumulation normalizes across micro-batches within this single pass.
        jsd_batch_size = jsd_batch.batch_size[0]
        jsd_mini_batch_size = min(self.config.ppo_mini_batch_size, jsd_batch_size)
        mini_batches = jsd_batch.split(jsd_mini_batch_size)

        for mini_batch in mini_batches:
            micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
            self.gradient_accumulation = jsd_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
            self.actor_optimizer.zero_grad()

            for mb in micro_batches:
                mb = mb.to(get_torch_device().current_device())
                responses = mb["responses"]
                response_length = responses.size(1)
                attention_mask = mb["attention_mask"]
                if multi_turn:
                    response_mask = mb["loss_mask"][:, -response_length:]
                else:
                    response_mask = attention_mask[:, -response_length:]

                # Teacher forward: guided prompt (input_ids = guided + response) — no gradient
                with torch.no_grad():
                    teacher_logits = self._forward_micro_batch_logits(
                        micro_batch=mb, temperature=jsd_temperature)

                # Student forward: plain prompt (plain_input_ids + response) — with gradient
                plain_mb = {**mb}
                plain_mb["input_ids"] = mb["plain_input_ids"]
                plain_mb["attention_mask"] = mb["plain_attention_mask"]
                plain_mb["position_ids"] = mb["plain_position_ids"]
                student_logits = self._forward_micro_batch_logits(
                    micro_batch=plain_mb, temperature=jsd_temperature)

                jsd_loss, jsd_metrics = compute_jsd_loss(
                    teacher_logits=teacher_logits.detach(),
                    student_logits=student_logits,
                    response_mask=response_mask,
                    top_k=jsd_top_k,
                    temperature=jsd_temperature,
                    loss_agg_mode=self.config.loss_agg_mode)

                loss = (jsd_lambda * jsd_loss) / self.gradient_accumulation
                loss.backward()

                append_to_dict(metrics, {
                    "actor/jsd_loss": jsd_loss.detach().item(),
                    "actor/jsd_lambda": jsd_lambda,
                    "actor/jsd_mean_per_token": jsd_metrics["jsd/mean_per_token"],
                    "actor/jsd_max_per_token": jsd_metrics["jsd/max_per_token"],
                    "actor/jsd_tail_mass": jsd_metrics["jsd/tail_mass_mean"],
                    "actor/jsd_token_count": jsd_metrics["jsd/token_count"],
                })

            grad_norm = self._optimizer_step()
            append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics

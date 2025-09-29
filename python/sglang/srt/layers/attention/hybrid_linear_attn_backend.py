from dataclasses import astuple, dataclass
from functools import lru_cache
from typing import Optional, Union

from sglang.srt.layers.lightning_attn import lightning_attention, linear_decode_forward_triton
import torch
import torch.nn.functional as F
import einops

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
from sglang.srt.layers.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_update,
)
from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, MinimaxReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.qwen3_next import Qwen3HybridLinearDecoderLayer, fused_gdn_gating
from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput
from sglang.srt.utils import is_cuda, is_npu

if is_cuda():
    from sglang.srt.layers.attention.mamba.causal_conv1d import (
        causal_conv1d_fn as causal_conv1d_fn_cuda,
    )

    causal_conv1d_fn = causal_conv1d_fn_cuda
elif is_npu():
    from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu
    from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update_npu,
    )
    from sgl_kernel_npu.mamba.causal_conv1d import (
        causal_conv1d_fn_npu,
        causal_conv1d_update_npu,
    )

    chunk_gated_delta_rule = chunk_gated_delta_rule_npu
    fused_sigmoid_gating_delta_rule_update = fused_sigmoid_gating_delta_rule_update_npu
    causal_conv1d_fn = causal_conv1d_fn_npu
    causal_conv1d_update = causal_conv1d_update_npu


@dataclass
class ForwardMetadata:
    query_start_loc: Optional[torch.Tensor]
    mamba_cache_indices: torch.Tensor


class MambaAttnBackend(AttentionBackend):
    pass


class LightningBackend(AttentionBackend):
    """Attention backend using Lightning kernel."""
    @staticmethod
    def _get_num_prefills(forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_extend():
            return forward_batch.batch_size
        elif forward_batch.forward_mode.is_mixed():
            return (
                len(forward_batch.extend_seq_lens)
                if forward_batch.extend_seq_lens is not None
                else 0
            )
        else:
            return 0
    
    @staticmethod
    def _get_num_prefill_tokens(forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_extend() or forward_batch.forward_mode.is_mixed():
            if forward_batch.extend_num_tokens is not None:
                return forward_batch.extend_num_tokens
            elif forward_batch.extend_seq_lens is not None:
                return int(forward_batch.extend_seq_lens.sum().item())
            else:
                return 0
        else:
            return 0

    @staticmethod
    def _get_num_decode_tokens(forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_decode():
            return forward_batch.batch_size
        elif forward_batch.forward_mode.is_mixed():
            num_prefills = LightningBackend._get_num_prefills(forward_batch)
            return max(0, forward_batch.batch_size - num_prefills)
        else:
            return 0

    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.pad_slot_id = -1  # Default pad slot id
        self.device = model_runner.device
        self.req_to_token_pool: MinimaxReqToTokenPool = model_runner.req_to_token_pool
        self.forward_metadata: ForwardMetadata = None
        self.state_indices_list = []
        self.query_start_loc_list = []

    @classmethod
    @lru_cache(maxsize=128)
    def _get_cached_arange(cls, bs: int, device_str: str) -> torch.Tensor:
        """Cache torch.arange tensors for common batch sizes to avoid repeated allocation."""
        device = torch.device(device_str)
        return torch.arange(0, bs + 1, dtype=torch.int32, device=device)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size
        if forward_batch.forward_mode.is_decode_or_idle():
            query_start_loc = self._get_cached_arange(bs, str(self.device))
        elif forward_batch.forward_mode.is_extend():
            if forward_batch.forward_mode.is_target_verify():
                raise ValueError("Target verify mode is not implemented")
            else:
                query_start_loc = torch.empty(
                    (bs + 1,), dtype=torch.int32, device=self.device
                )
                query_start_loc[:bs] = forward_batch.extend_start_loc
                query_start_loc[bs] = (
                    forward_batch.extend_start_loc[-1]
                    + forward_batch.extend_seq_lens[-1]
                )
        else:
            raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode}")

        cache_indices = self.req_to_token_pool.get_minimax_indices(
            forward_batch.req_pool_indices
        )
        self.forward_metadata = ForwardMetadata(
            query_start_loc=query_start_loc,
            mamba_cache_indices=cache_indices,
        )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for i in range(max_bs):
            self.state_indices_list.append(
                torch.full(
                    (i + 1,), self.pad_slot_id, dtype=torch.int32, device=self.device
                )
            )
            self.query_start_loc_list.append(
                torch.empty((i + 2,), dtype=torch.int32, device=self.device)
            )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        if forward_mode.is_decode_or_idle():
            self.query_start_loc_list[bs - 1].copy_(self._get_cached_arange(bs, "cuda"))
        elif forward_mode.is_target_verify():
            raise ValueError("Target verify mode is not implemented")
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode}")

        indices = self.req_to_token_pool.get_minimax_indices(req_pool_indices)
        self.state_indices_list[bs - 1][: len(indices)].copy_(indices)
        self.forward_metadata = ForwardMetadata(
            query_start_loc=self.query_start_loc_list[bs - 1],
            mamba_cache_indices=self.state_indices_list[bs - 1],
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode,
        spec_info,
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        # 处理 padding 请求
        num_padding = torch.count_nonzero(
            seq_lens_cpu == self.get_cuda_graph_seq_len_fill_value()
        )
        req_pool_indices[bs - num_padding :] = 0
        indices = self.req_to_token_pool.get_minimax_indices(req_pool_indices)
        indices[bs - num_padding :] = -1

        self.state_indices_list[bs - 1][: len(indices)].copy_(indices)

        if forward_mode.is_decode_or_idle():
            self.query_start_loc_list[bs - 1].copy_(self._get_cached_arange(bs, "cuda"))
            if num_padding > 0:
                self.query_start_loc_list[bs - 1][bs - num_padding :] = bs - num_padding
        elif forward_mode.is_target_verify():
            raise ValueError("Target verify mode is not implemented")
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")

        self.forward_metadata = ForwardMetadata(
            query_start_loc=self.query_start_loc_list[bs - 1],
            mamba_cache_indices=self.state_indices_list[bs - 1],
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1  # 和 Qwen3 保持一致

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """
        decode 路径：一 token 一 token 追加。
        直接调用你现有的 linear_decode_forward_triton。
        """
        layer_id = kwargs["layer_id"]
        slope_rate = kwargs["slope_rate"]
        block_size = kwargs.get("block_size", 32)


        num_prefill_tokens = LightningBackend._get_num_prefill_tokens(forward_batch)
        num_prefills = LightningBackend._get_num_prefills(forward_batch)
        q = q[num_prefill_tokens:].unsqueeze(2).contiguous()
        k = k[num_prefill_tokens:].unsqueeze(2).contiguous()
        v = v[num_prefill_tokens:].unsqueeze(2).contiguous()

        (state,) = self.req_to_token_pool.get_minimax_params(layer_id)
        slot_id = self.forward_metadata.mamba_cache_indices
        assert len(slot_id) == q.shape[0], (
            f"slot_id length {len(slot_id)} does not match decode batch size {q.shape[0]}. "
            "This indicates a bug in the upstream logic that should be investigated."
        )

        hidden = linear_decode_forward_triton(
            q, k, v, state, slope_rate, slot_id, 32
        )
        return hidden

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        layer_id = kwargs["layer_id"]
        slope_rate = kwargs["slope_rate"]
        block_size = kwargs.get("BLOCK", 256)

        (state,) = self.req_to_token_pool.get_minimax_params(layer_id)
        cache_indices = self.forward_metadata.mamba_cache_indices
        query_start_loc = self.forward_metadata.query_start_loc

        hidden = []
        for _prefill_idx in range(self._get_num_prefills(forward_batch)):
            if _prefill_idx >= len(forward_batch.extend_start_loc):
                break
            if _prefill_idx >= len(query_start_loc):
                break

            _start = forward_batch.extend_start_loc[_prefill_idx]

            if _prefill_idx + 1 < len(forward_batch.extend_start_loc):
                _end = forward_batch.extend_start_loc[_prefill_idx + 1]
            else:
                if forward_batch.extend_seq_lens is not None and _prefill_idx < len(
                    forward_batch.extend_seq_lens
                ):
                    seq_len = forward_batch.extend_seq_lens[_prefill_idx]
                    _end = _start + seq_len
                else:
                    _end = q.shape[0]

            slot_id = self.forward_metadata.mamba_cache_indices[_prefill_idx]
            qs = q[_start:_end].transpose(0, 1).contiguous()
            ks = k[_start:_end].transpose(0, 1).contiguous()
            vs = v[_start:_end].transpose(0, 1).contiguous()
            slice_layer_cache = state[slot_id, ...]


            def jit_linear_forward_prefix(
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                kv_caches: torch.Tensor,
                slope_rate: torch.Tensor,
                block_size: int,
                layer_idx: int = None,
                **kwargs,
            ) -> torch.Tensor:

                slope_rate = slope_rate.to(torch.float32)
                should_pad_dim = q.dim() == 3
                if should_pad_dim:
                    q = q.unsqueeze(0)
                    k = k.unsqueeze(0)
                    v = v.unsqueeze(0)
                b, h, n, d = q.shape
                e = d
                kv_history = kv_caches.reshape(1, h, d, e).contiguous()
                output, kv_history = lightning_attention(
                    q, k, v, slope_rate, block_size=block_size, kv_history=kv_history
                )
                kv_caches.copy_(kv_history[:, :, -1, :, :].reshape(h, d, e))
                assert output.shape[0] == 1, "batch size must be 1"
                return einops.rearrange(output.squeeze(0), "h n d -> n (h d)")


            out_slice = jit_linear_forward_prefix(
                qs,
                ks,
                vs,
                slice_layer_cache,
                slope_rate,
                block_size,
                layer_id,
            )
            hidden.append(out_slice.contiguous())
        if self._get_num_decode_tokens(forward_batch) > 0:
            hidden.append(
                self._decode_infer(
                    q, k, v, state, self.forward_metadata.mamba_cache_indices, forward_batch
                )
            )

        if not hidden:
            return torch.empty((0, q.size(-1)), device=q.device, dtype=q.dtype)

        hidden = torch.concat(hidden, dim=0).contiguous()
        return hidden

        # # lightning_attention 的输入需要 [B, H, N, D]
        # # 你的 q,k,v 可能是 [N, H, D] 或 [H, N, D]，请确保在 Layer 侧 reshape 成 [B, H, N, D]
        # # 这里直接调用
        # output, updated_state = lightning_attention(
        #     q, k, v, slope_rate, block_size=block_size,
        #     kv_history=state[cache_indices]
        # )

        # # 把最新状态写回 pool
        # state[cache_indices] = updated_state[:, :, -1, :, :]
        # return output.reshape(-1, output.shape[-1])

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """
        对外统一接口：根据 forward_mode 路由。
        """
        mode = forward_batch.forward_mode
        if mode.is_idle():
            return torch.empty_like(q)
        elif mode.is_decode():
            return self.forward_decode(q, k, v, layer, forward_batch, save_kv_cache, **kwargs)
        elif mode.is_extend():
            return self.forward_extend(q, k, v, layer, forward_batch, save_kv_cache, **kwargs)
        else:
            raise ValueError(f"Unsupported forward mode: {mode}")


class HybridLinearAttnBackend(AttentionBackend):
    """Support different backends for prefill and decode."""

    def __init__(
        self,
        full_attn_backend: AttentionBackend,
        linear_attn_backend: AttentionBackend,
        full_attn_layers: list[int],
    ):
        self.full_attn_layers = full_attn_layers
        self.attn_backend_list = [full_attn_backend, linear_attn_backend]

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata_capture_cuda_graph(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_mode,
                spec_info,
            )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata_replay_cuda_graph(
                bs,
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                encoder_lens,
                forward_mode,
                spec_info,
                seq_lens_cpu,
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return self.attn_backend_list[0].get_cuda_graph_seq_len_fill_value()

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        if layer_id in self.full_attn_layers:
            return self.attn_backend_list[0].forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        return self.attn_backend_list[1].forward_decode(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        if layer_id in self.full_attn_layers:
            return self.attn_backend_list[0].forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        return self.attn_backend_list[1].forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Run forward on an attention layer."""
        if forward_batch.forward_mode.is_idle():
            if layer is None:
                return torch.empty_like(kwargs["z"])
            return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
        elif forward_batch.forward_mode.is_decode():
            return self.forward_decode(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )
        else:
            return self.forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )

    def update_mamba_state_after_mtp_verify(self, accepted_length, model):
        request_number = accepted_length.shape[0]

        state_indices_tensor = self.attn_backend_list[
            1
        ].forward_metadata.mamba_cache_indices[:request_number]

        mamba_caches = self.attn_backend_list[
            1
        ].req_to_token_pool.get_mamba_params_all_layers()

        (
            conv_states,
            ssm_states,
            intermediate_state_cache,
            intermediate_conv_window_cache,
        ) = mamba_caches

        # SSM state updates (chunked to reduce peak memory)
        valid_mask = accepted_length > 0

        # Compute common indices once to avoid duplication
        last_steps_all = (accepted_length - 1).to(torch.int64)
        valid_state_indices = state_indices_tensor[valid_mask].to(torch.int64)  # [N]
        last_steps = last_steps_all[valid_mask].to(torch.int64)  # [N]

        # scatter into ssm_states at the chosen cache lines
        ssm_states[:, valid_state_indices, :] = intermediate_state_cache[
            :, valid_state_indices, last_steps
        ].to(ssm_states.dtype, copy=False)

        # Scatter into conv_states at the chosen cache lines
        conv_states[:, valid_state_indices, :, :] = intermediate_conv_window_cache[
            :, valid_state_indices, last_steps
        ].to(conv_states.dtype, copy=False)

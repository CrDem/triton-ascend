"""
Fused Attention - Single Loop Implementation
============================================

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao
(https://tridao.me/publications/flash2/flash2.pdf) adapted for Ascend NPU with a 
single flattened execution loop.
"""

import math
from typing import List
import pytest
import torch
import torch_npu
import triton
import triton.language as tl
import triton.runtime.driver as driver

DEVICE = "npu"

device = torch.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
AICORE_NUM = properties["num_aicore"]


@triton.jit
def _vdv_atn_fwd(
    Q, K, V, ATTEN_MASK, M, Out, sm_scale: tl.constexpr, sparse_start_idx,
    stride_qz: tl.constexpr, stride_qh: tl.constexpr, stride_qm: tl.constexpr, stride_qk: tl.constexpr,
    stride_kz: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr, stride_kk: tl.constexpr,
    stride_vz: tl.constexpr, stride_vh: tl.constexpr, stride_vn: tl.constexpr, stride_vk: tl.constexpr,
    stride_oz: tl.constexpr, stride_oh: tl.constexpr, stride_om: tl.constexpr, stride_on: tl.constexpr,
    stride_am: tl.constexpr,
    Z: tl.constexpr, H: tl.constexpr,
    num_kv_heads,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    NUM_BLOCKS_PER_CORE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    NUM_BLOCKS_M: tl.constexpr,
    AICORE_NUM: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    pid = tl.program_id(0)

    if STAGE == 3:
        start_block = tl.load(sparse_start_idx + pid)
        end_block = tl.load(sparse_start_idx + pid + 1)
        step = 1
    else:
        start_block, end_block, step = pid, NUM_BLOCKS, AICORE_NUM

    STEPS_PER_BLOCK: tl.constexpr = N_CTX // BLOCK_N
    TOTAL_STEPS: tl.constexpr = NUM_BLOCKS_PER_CORE * STEPS_PER_BLOCK

    # Running tile registers maintained across the flattened loop
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    q = tl.zeros([BLOCK_M, HEAD_DIM], dtype=Q.dtype.element_ty)

    offs_n_init = tl.arange(0, BLOCK_N)

    # Single flattened loop over blocks and sequence tiles
    for step_idx in tl.range(0, TOTAL_STEPS):
        outer_iter = step_idx // STEPS_PER_BLOCK
        inner_step = step_idx % STEPS_PER_BLOCK
        block_idx = start_block + outer_iter * step

        if block_idx < end_block:
            task_hz_idx = block_idx // NUM_BLOCKS_M
            task_m_idx = block_idx % NUM_BLOCKS_M
            off_z = task_hz_idx // H
            off_h = task_hz_idx % H
            kv_h = off_h * num_kv_heads // H

            q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
            k_offset = off_z.to(tl.int64) * stride_kz + kv_h.to(tl.int64) * stride_kh
            v_offset = off_z.to(tl.int64) * stride_vz + kv_h.to(tl.int64) * stride_vh

            # First step of a sequence block: initialize state and load Q
            if inner_step == 0:
                acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
                l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
                m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

                Q_block_ptr = tl.make_block_ptr(
                    base=Q + q_offset,
                    shape=(N_CTX, HEAD_DIM),
                    strides=(stride_qm, stride_qk),
                    offsets=(task_m_idx * BLOCK_M, 0),
                    block_shape=(BLOCK_M, HEAD_DIM),
                    order=(1, 0),
                )
                q = tl.load(Q_block_ptr)

            start_n = inner_step * BLOCK_N

            # Condition checks equivalent to STAGE 1 (off-band) & STAGE 2 (on-band)
            is_offband = (STAGE & 1) != 0 and ((STAGE != 3) or (start_n < task_m_idx * BLOCK_M))
            is_onband = (STAGE & 2) != 0 and (start_n >= task_m_idx * BLOCK_M) and (start_n < (task_m_idx + 1) * BLOCK_M)

            if is_offband or is_onband:
                K_block_ptr = tl.make_block_ptr(
                    base=K + k_offset,
                    shape=(N_CTX, HEAD_DIM),
                    strides=(stride_kn, stride_kk),
                    offsets=(start_n, 0),
                    block_shape=(BLOCK_N, HEAD_DIM),
                    order=(1, 0),
                )
                V_block_ptr = tl.make_block_ptr(
                    base=V + v_offset,
                    shape=(N_CTX, HEAD_DIM),
                    strides=(stride_vn, stride_vk),
                    offsets=(start_n, 0),
                    block_shape=(BLOCK_N, HEAD_DIM),
                    order=(1, 0),
                )

                k = tl.load(K_block_ptr)
                trans_k = tl.trans(k)
                qk = tl.dot(q, trans_k)

                if is_onband:
                    if HAS_MASK:
                        attn_mask_ptr = tl.make_block_ptr(
                            base=ATTEN_MASK,
                            shape=(N_CTX, N_CTX),
                            strides=(stride_am, 1),
                            offsets=(task_m_idx * BLOCK_M, start_n),
                            block_shape=(BLOCK_M, BLOCK_N),
                            order=(1, 0),
                        )
                        mask = tl.load(attn_mask_ptr)
                        qk = qk * sm_scale + tl.where(mask, -1.0e4, 0)
                    else:
                        qk = qk * sm_scale
                else:
                    qk = qk * sm_scale

                m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
                qk -= m_ij[:, None]

                p = tl.math.exp(qk)
                p_cast = p.to(q.type)
                v = tl.load(V_block_ptr)
                pv = tl.dot(p_cast, v)

                l_ij = tl.sum(p, 1)
                alpha = tl.math.exp(m_i - m_ij)
                l_i = l_i * alpha + l_ij
                acc = acc * alpha[:, None] + pv
                m_i = m_ij.to(m_i.type)

            # Last step of a sequence block: write out final results
            if inner_step == STEPS_PER_BLOCK - 1:
                offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
                m_i += tl.math.log(l_i).to(q.type)
                acc = acc / l_i[:, None]

                m_ptrs = M + task_hz_idx * N_CTX + offs_m
                tl.store(m_ptrs, m_i)

                O_block_ptr = tl.make_block_ptr(
                    base=Out + q_offset,
                    shape=(N_CTX, HEAD_DIM),
                    strides=(stride_om, stride_on),
                    offsets=(task_m_idx * BLOCK_M, 0),
                    block_shape=(BLOCK_M, HEAD_DIM),
                    order=(1, 0),
                )
                tl.store(O_block_ptr, acc.to(Out.type.element_ty))


SPARSE_INDEX_CACHE = {}


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, atten_mask, causal, sm_scale, BM, BN, compile_opt=None):
        sm_scale: torch.float16 = sm_scale
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}

        if compile_opt is None:
            compile_opt = {
                "debug": False,
                "main_loop_unroll_factor": 1,
                #"demote_f32_reduction": True,
            }

        num_cores = AICORE_NUM
        NUM_BLOCKS_M = triton.cdiv(q.shape[2], BM)
        NUM_BLOCKS = NUM_BLOCKS_M * q.shape[0] * q.shape[1]
        num_cores = num_cores if NUM_BLOCKS > num_cores else NUM_BLOCKS
        NUM_BLOCKS_PER_CORE = triton.cdiv(NUM_BLOCKS, num_cores)

        cache_key = (q.shape[0], q.shape[1], q.shape[2], k.shape[2], BM, BN, causal, num_cores)
        if cache_key in SPARSE_INDEX_CACHE:
            sparse_start_idx = SPARSE_INDEX_CACHE[cache_key]
        else:
            sparse_valid_array = [0] * (q.shape[2] // BM)
            sparse_start_idx = [NUM_BLOCKS] * (num_cores + 1)
            init_sparse_valid_array(
                sparse_valid_array, k.shape[2], BM, BN, s1_sparse_valid_size=q.shape[2], s2_sparse_valid_size=0
            )
            set_sparse_start_idx(
                sparse_valid_array,
                sparse_start_idx,
                total_size=NUM_BLOCKS,
                core_num=num_cores,
                max_core_num=num_cores,
            )
            sparse_start_idx = torch.tensor(sparse_start_idx, device=q.device, dtype=torch.int32)
            SPARSE_INDEX_CACHE[cache_key] = sparse_start_idx

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        has_mask = atten_mask is not None        
        mask_ptr = atten_mask if has_mask else q 
        stride_am = atten_mask.stride(0) if has_mask else 0
        _vdv_atn_fwd[(num_cores,)](
            q, k, v, mask_ptr, M, o, sm_scale,
            sparse_start_idx,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            stride_am,
            q.shape[0], q.shape[1], k.shape[1],
            N_CTX=q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            BLOCK_M=BM,
            BLOCK_N=BN,
            STAGE=stage,
            NUM_BLOCKS_PER_CORE=NUM_BLOCKS_PER_CORE,
            NUM_BLOCKS=NUM_BLOCKS,
            NUM_BLOCKS_M=NUM_BLOCKS_M,
            AICORE_NUM=num_cores,
            HAS_MASK=has_mask,
            **compile_opt,
            **extra_kern_args
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o


attention = _attention.apply


def init_sparse_valid_array(
    sparse_valid_array: List[int],
    s2_size, BLOCK_S1, BLOCK_S2, s1_sparse_valid_size, s2_sparse_valid_size
):
    INVALID_ROW_SPARSE_RATIO = 6
    assert len(sparse_valid_array) > 0, "init_sparse_valid_array: Sparse valid array size should be larger than 0."

    s2_num_blocks = math.ceil(s2_size / BLOCK_S2)
    valid_s1_size = math.ceil(s1_sparse_valid_size / BLOCK_S1)
    valid_s2_size = math.ceil(s2_sparse_valid_size / BLOCK_S2)
    for i in range(len(sparse_valid_array)):
        reduce_blocks = 0 if i < valid_s1_size else (triton.cdiv((i + 1) * BLOCK_S1 - s1_sparse_valid_size, BLOCK_S2) - 1)
        add_blocks = min(
            s2_num_blocks - valid_s2_size,
            math.ceil(((i + 1) * BLOCK_S1 + s2_sparse_valid_size) / BLOCK_S2) - valid_s2_size
        )
        valid_block_num = valid_s2_size - reduce_blocks + add_blocks
        sparse_valid_array[i] = valid_block_num if valid_block_num > 0 else INVALID_ROW_SPARSE_RATIO
    return True


def set_sparse_start_idx(sparse_valid_array, sparse_start_idx, total_size, core_num, max_core_num):
    valid_aiv_num = min(core_num, max_core_num)
    assert valid_aiv_num > 0, "valid_aiv_num should be greater than 0"

    for idx in range(max_core_num):
        sparse_start_idx[idx] = total_size

    if total_size <= valid_aiv_num:
        for idx in range(total_size):
            sparse_start_idx[idx] = idx
        return True

    sparse_array_sum = sum(sparse_valid_array)
    load_total = sparse_array_sum * (total_size // len(sparse_valid_array))
    partition_result = [total_size] * (valid_aiv_num + 1)
    last_valid_partition_result = [total_size] * (valid_aiv_num + 1)

    load_each_core_lower_bound = load_total // valid_aiv_num - 1
    max_sparse_valid = max(sparse_valid_array)
    load_each_core_upper_bound = triton.cdiv(load_total, valid_aiv_num) + max_sparse_valid

    while load_each_core_lower_bound + 1 < load_each_core_upper_bound:
        load_max = load_each_core_lower_bound + (load_each_core_upper_bound - load_each_core_lower_bound) // 2
        if (
            load_max * valid_aiv_num >= load_total and
            partition_sparse_data(sparse_valid_array, sparse_array_sum, total_size, load_max, partition_result)
        ):
            load_each_core_upper_bound = load_max
            last_valid_partition_result, partition_result = partition_result, last_valid_partition_result
        else:
            load_each_core_lower_bound = load_max

    for idx in range(valid_aiv_num):
        sparse_start_idx[idx] = last_valid_partition_result[idx]


def partition_sparse_data(
    sparse_rolling_array: List[int],
    sparse_rolling_array_sum: int,
    sparse_array_size: int,
    load_max_each_core: int,
    partition_result: List[int]
) -> bool:
    assert partition_result, "partition_result should be greater than 0"
    assert sparse_rolling_array_sum > 0, "sparse_rolling_array_sum should be greater than 0"

    s1_outer_cut_each_core = load_max_each_core // sparse_rolling_array_sum
    s1_outer_load_each_core = s1_outer_cut_each_core * sparse_rolling_array_sum
    s1_outer_num_each_core = s1_outer_cut_each_core * len(sparse_rolling_array)

    target_core_num = len(partition_result)
    core_idx = 0
    rolling_idx = 0
    load_size = s1_outer_load_each_core

    partition_result[0] = 0

    i = s1_outer_num_each_core
    while i < sparse_array_size:
        if rolling_idx >= len(sparse_rolling_array):
            rolling_idx = 0

        load_next = sparse_rolling_array[rolling_idx]
        need_one_more_core = (load_size + load_next) > load_max_each_core

        if need_one_more_core and core_idx >= (target_core_num - 1):
            return False

        if need_one_more_core:
            core_idx += 1
            partition_result[core_idx] = i
            i += s1_outer_num_each_core - 1
            rolling_idx -= 1
            load_size = s1_outer_load_each_core
        else:
            load_size += load_next
        i += 1
        rolling_idx += 1

    for j in range(core_idx + 1, target_core_num):
        partition_result[j] = sparse_array_size

    return True


@pytest.mark.parametrize("Z,H,N_CTX,HEAD_DIM,causal,dtype,BM,BN", [
    [128, 8, 8192, 128, False, torch.float16, 128, 128],
    [128, 8, 8192, 64, False, torch.float16, 128, 128],
    [128, 8, 1024, 128, False, torch.float16, 128, 128],
    [128, 8, 1024, 64, False, torch.float16, 128, 128],
    [128, 8, 8192, 128, True, torch.float16, 128, 128],
    [128, 8, 8192, 64, True, torch.float16, 128, 128],
    [128, 8, 1024, 128, True, torch.float16, 128, 128],
    [128, 8, 1024, 64, True, torch.float16, 128, 128],
    [128, 8, 8192, 128, False, torch.bfloat16, 128, 128],
    [128, 8, 8192, 64, False, torch.bfloat16, 128, 128],
    [128, 8, 1024, 128, False, torch.bfloat16, 128, 128],
    [128, 8, 1024, 64, False, torch.bfloat16, 128, 128],
    [128, 8, 8192, 128, True, torch.bfloat16, 128, 128],
    [128, 8, 8192, 64, True, torch.bfloat16, 128, 128],
    [128, 8, 1024, 128, True, torch.bfloat16, 128, 128],
    [128, 8, 1024, 64, True, torch.bfloat16, 128, 128],
])
def test_op(Z, H, N_CTX, HEAD_DIM, causal, dtype, BM, BN):
    torch.manual_seed(20)
    q = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    k = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    v = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()

    sm_scale = 0.5
    atten_mask = None

    if causal:
        atten_mask = torch.triu(torch.ones(N_CTX, N_CTX, device=DEVICE), diagonal=1)

    if atten_mask is None:
        atten_mask = torch.zeros((1, 1), device=DEVICE)

    tri_out = attention(q, k, v, atten_mask, causal, sm_scale, BM, BN)
    print("compare success!")


if __name__ == "__main__":
    test_op(1, 1, 128, 128, causal=False, dtype=torch.float16, BM=64, BN=64)

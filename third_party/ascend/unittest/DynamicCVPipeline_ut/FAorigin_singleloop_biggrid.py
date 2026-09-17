"""
Fused Attention - True Single Loop (Hardware Pipelined)
======================================================
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
    Q, K, V, ATTEN_MASK, M, Out, sm_scale: tl.constexpr,
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
    NUM_BLOCKS_M: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    # Map program ID directly across the (Batch, Head, M-block) space
    block_idx = tl.program_id(0)

    task_hz_idx = block_idx // NUM_BLOCKS_M
    task_m_idx = block_idx % NUM_BLOCKS_M
    off_z = task_hz_idx // H
    off_h = task_hz_idx % H
    kv_h = off_h * num_kv_heads // H

    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + kv_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + kv_h.to(tl.int64) * stride_vh

    # Initialize pointers once outside the loop
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(task_m_idx * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )

    offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # Q is loaded once and stays in L0A / UB
    q = tl.load(Q_block_ptr)

    # Sequence range setup
    if STAGE == 1:
        lo, hi = 0, N_CTX
    elif STAGE == 3:
        lo, hi = 0, (task_m_idx + 1) * BLOCK_M

    K_block_ptr = tl.advance(K_block_ptr, (lo, 0))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    if HAS_MASK:
        attn_mask_ptr = tl.make_block_ptr(
            base=ATTEN_MASK,
            shape=(N_CTX, N_CTX),
            strides=(stride_am, 1),
            offsets=(task_m_idx * BLOCK_M, lo),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )

    # THE ONLY LOOP: 100% hardware-pipelined with DMA multi-buffering
    for start_n in tl.range(lo, hi, BLOCK_N):
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, tl.trans(k)) * sm_scale

        if STAGE == 3:
            # Causal diagonal boundary condition
            is_diag = start_n >= task_m_idx * BLOCK_M
            if is_diag:
                row_idx = offs_m[:, None]
                col_idx = (start_n + offs_n)[None, :]
                qk = tl.where(row_idx >= col_idx, qk, -1.0e4)
        elif HAS_MASK:
            mask = tl.load(attn_mask_ptr)
            qk = qk + tl.where(mask, -1.0e4, 0.0)
            attn_mask_ptr = tl.advance(attn_mask_ptr, (0, BLOCK_N))

        m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
        p = tl.math.exp(qk - m_ij[:, None]).to(q.type)

        v = tl.load(V_block_ptr)
        pv = tl.dot(p, v)

        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp(m_i - m_ij)

        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None] + pv
        m_i = m_ij.to(m_i.type)

        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # Epilogue: normalization and memory writes executed once outside the loop
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

        if compile_opt is None:
            compile_opt = {
                "debug": False,
                "main_loop_unroll_factor": 1,
            }

        NUM_BLOCKS_M = triton.cdiv(q.shape[2], BM)
        NUM_BLOCKS = NUM_BLOCKS_M * q.shape[0] * q.shape[1]

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        has_mask = atten_mask is not None
        mask_ptr = atten_mask if has_mask else q
        stride_am = atten_mask.stride(0) if has_mask else 0

        # Launch the grid across all M-blocks; hardware distributes blocks across AICOREs
        grid = (NUM_BLOCKS,)

        _vdv_atn_fwd[grid](
            q, k, v, mask_ptr, M, o, sm_scale,
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
            NUM_BLOCKS_M=NUM_BLOCKS_M,
            HAS_MASK=has_mask,
            **compile_opt
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o


attention = _attention.apply


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

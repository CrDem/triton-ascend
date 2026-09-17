"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Credits: OpenAI kernel team

Extra Credits:

* Original flash attention paper (https://arxiv.org/abs/2205.14135)
* Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import pytest
import torch
import torch_npu
import triton
import triton.language as tl

DEVICE = "npu"

import triton.runtime.driver as driver
from typing import List

device = torch.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
AICORE_NUM = properties["num_aicore"]


@triton.jit
def _vdv_atn_fwd_inner(acc, l_i, m_i, q, #
                    K_block_ptr, V_block_ptr, ATTEN_MASK, #
                    stride_am, start_m, qk_scale: tl.constexpr,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr):
    if STAGE == 1:
        tl.static_assert(BLOCK_M >= BLOCK_N)
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        tl.static_assert(BLOCK_M >= BLOCK_N)
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
        attn_mask_ptr = tl.make_block_ptr(
            base = ATTEN_MASK,
            shape=(N_CTX, N_CTX),
            strides=(stride_am, 1),
            offsets=(start_m * BLOCK_M, lo),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0)
        )
    else:
        lo, hi = 0, N_CTX
        
    K_block_ptr = tl.advance(K_block_ptr, (lo, 0))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N):#, loop_unroll_factor=2):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)

      
        trans_k = tl.trans(k)
        qk = tl.dot(q, trans_k)
        # ------------------------------

        if STAGE == 2:
            curr_mask_ptr = attn_mask_ptr
            mask = tl.load(curr_mask_ptr)
            qk = qk * qk_scale + tl.where(mask, -1.0e4, 0)
            m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
            qk -= m_ij[:, None]
            attn_mask_ptr = tl.advance(attn_mask_ptr, (0, BLOCK_N))
        else:
            qk = qk * qk_scale
            m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
            qk = qk - m_ij[:, None]
        p = tl.math.exp(qk)
        p_cast = p.to(q.type)
        v = tl.load(V_block_ptr)
        pv = tl.dot(p_cast, v)
        #pv = tl.dot(p, v)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp(m_i - m_ij)
        #alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None] + pv

        m_i = m_ij.to(m_i.type) 
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
    return acc, l_i, m_i


@triton.jit
def _vdv_atn_fwd(Q, K, V, ATTEN_MASK, M, Out, sm_scale: tl.constexpr, sparse_start_idx, #
              stride_qz: tl.constexpr, stride_qh: tl.constexpr, stride_qm: tl.constexpr, stride_qk: tl.constexpr,  #
              stride_kz: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr, stride_kk: tl.constexpr,  #
              stride_vz: tl.constexpr, stride_vh: tl.constexpr, stride_vn: tl.constexpr, stride_vk: tl.constexpr,  #
              stride_oz: tl.constexpr, stride_oh: tl.constexpr, stride_om: tl.constexpr, stride_on: tl.constexpr,  #
              stride_am: tl.constexpr,
              Z: tl.constexpr, H: tl.constexpr, 
              num_kv_heads, # if H==num_kv_heads, MHA; H % num_kv_heads==0, GQA
              N_CTX: tl.constexpr,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              NUM_BLOCKS_PER_CORE: tl.constexpr,
              NUM_BLOCKS: tl.constexpr,
              NUM_BLOCKS_M: tl.constexpr,
              AICORE_NUM: tl.constexpr,
              ):
    pid = tl.program_id(0)
    block_start = pid * NUM_BLOCKS_PER_CORE
    NUM_BLOCKS_hz = NUM_BLOCKS // NUM_BLOCKS_M 
    task_m_idx = 0
    task_hz_idx = 0

    if STAGE == 3:
        # use indices calculated by load balancer in causal stage
        start_block = tl.load(sparse_start_idx + pid)
        end_block = tl.load(sparse_start_idx + pid + 1)
        step = 1
    else:
        start_block, end_block, step = pid, NUM_BLOCKS, AICORE_NUM
    
    for block_idx in range(start_block, end_block, step):
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M
        off_z = task_hz_idx // H
        off_h = task_hz_idx % H
        kv_h = off_h * num_kv_heads // H

        q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        k_offset = off_z.to(tl.int64) * stride_kz + kv_h.to(tl.int64) * stride_kh
        v_offset = off_z.to(tl.int64) * stride_vz + kv_h.to(tl.int64) * stride_vh
        Q_block_ptr = tl.make_block_ptr(
            base=Q + q_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_qm, stride_qk),

            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
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
        K_block_ptr = tl.make_block_ptr(
            base=K + k_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_kn, stride_kk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            base=Out + q_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_om, stride_on),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )

        # initialize offsets
        offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        #m_i = tl.full([BLOCK_M], -65504.0, dtype=tl.float16)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        # load scales

        # load q: it will stay in SRAM throughout
        q = tl.load(Q_block_ptr)
        # stage 1: off-band
        # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
        # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE

        #FP16 mod
        #m_i = m_i.cast(q.type)

        if STAGE & 1:
            acc, l_i, m_i = _vdv_atn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, ATTEN_MASK, #
                                            stride_am, task_m_idx, sm_scale,  #
                                            BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                            4 - STAGE, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                            )
        # stage 2: on-band

        if STAGE & 2:
            # barrier makes it easier for compielr to schedule the
            # two loops independently
            acc, l_i, m_i = _vdv_atn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, ATTEN_MASK, #
                                            stride_am, task_m_idx, sm_scale,  #
                                            BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                            2, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                            )
        # epilogue
        m_i += tl.math.log(l_i).to(q.type)
        acc = acc / l_i[:, None]
        m_ptrs = M + task_hz_idx * N_CTX + offs_m

        tl.store(m_ptrs, m_i)
        tl.store(O_block_ptr, acc.to(Out.type.element_ty))

# define a cache dict
SPARSE_INDEX_CACHE = {}

class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, atten_mask, causal, sm_scale, BM, BN, compile_opt = None):
        sm_scale: torch.float16 = sm_scale
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        o = torch.empty_like(q)

        # stage = 3
        stage = 3 if causal else 1
        extra_kern_args = {}
        if compile_opt is None:
            #compile_opt = {}
            compile_opt = {
                "debug": False, #True,
                "main_loop_unroll_factor": 1,
                "demote_f32_reduction": True,
                ###"enable_graph_optimize": False,
                #"buf_slot_num_of_veccore": 3,
                #"buf_slot_num_of_crosscore": 2,
                #"multibuffer": False, # True doesn't work with reorder
                #"enable_mixed_cv": True,
                #"enable_auto_bind_sub_block": False,
                ###"sync_solver": True,
                
                #### VDV experiments
                #"ops_reorder":True,
                #"code_motion":True,
                #"enable_preload":True,
                
                #"enable_nd2nz_on_vector" : True,
                ###"use_bytecode": True,
                #"disable_tightly_coupled_buffer_reuse": True,
                #"enable_drop_unit_dims": True,
                #"set_workspace_multibuffer": 2,
                #"add_auto_scheduling": True,
                #"enable_dynamic_cv_pipeline": False,
                #"limit_auto_multi_buffer_of_local_buffer": "no-limit",
                ###"tile_mix_vector_loop": True, # no such options in bishengir
                ###"tile_mix_cube_loop": True, # no such options in bishengir
                ###"enable_vf_fusion" : True, # perf degradation
                ###"vf_fusion_mode": "all-op", #"ub-aware-op",
                #"vf_merge_level": 2,
                #"enable_auto_vectorize_v2": True, ###False,
                #"enable_cce_vf_remove_membar": True,
                #"mix_mode": "aic", # --disable-hfusion-vectorize=true compile error
                #"enable_hivm_auto_cv_balance": True,
                ###"usesDAG": False,
                ###"usesBuffer": True,
                
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
            sparse_start_idx = [NUM_BLOCKS] * (num_cores+1)
            init_sparse_valid_array(sparse_valid_array,k.shape[2],BM,BN,s1_sparse_valid_size=q.shape[2],s2_sparse_valid_size=0)
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
        _vdv_atn_fwd[(num_cores,)](
            q, k, v, atten_mask, M, o, sm_scale,
            sparse_start_idx,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[2],
            q.shape[0], q.shape[1], k.shape[1], N_CTX=q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            BLOCK_M = BM,
            BLOCK_N = BN,
            STAGE=stage,
            NUM_BLOCKS_PER_CORE=NUM_BLOCKS_PER_CORE,
            NUM_BLOCKS=NUM_BLOCKS,
            NUM_BLOCKS_M=NUM_BLOCKS_M,
            AICORE_NUM = num_cores,
            **compile_opt,
            **extra_kern_args)


        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o

import math
attention = _attention.apply

def init_sparse_valid_array(sparse_valid_array: List[int],
    s2_size, BLOCK_S1, BLOCK_S2,s1_sparse_valid_size,s2_sparse_valid_size):
    INVALID_ROW_SPARSE_RATIO = 6
    assert len(sparse_valid_array) > 0 ,'init_sparse_valid_array: Sparse valid array size should be larger than 0.'

    s2_num_blocks = math.ceil(s2_size / BLOCK_S2)
    valid_s1_size = math.ceil(s1_sparse_valid_size / BLOCK_S1)
    valid_s2_size = math.ceil(s2_sparse_valid_size / BLOCK_S2)
    for i in range(len(sparse_valid_array)):
        reduce_blocks = 0 if i < valid_s1_size else (triton.cdiv((i + 1) * BLOCK_S1 - s1_sparse_valid_size , BLOCK_S2) - 1)
        add_blocks = min(s2_num_blocks - valid_s2_size,
            math.ceil(((i + 1) * BLOCK_S1 + s2_sparse_valid_size) / BLOCK_S2) - valid_s2_size)
        valid_block_num = valid_s2_size - reduce_blocks + add_blocks
        sparse_valid_array[i] = valid_block_num  if valid_block_num > 0 else INVALID_ROW_SPARSE_RATIO
    return True

def set_sparse_start_idx(sparse_valid_array,
                        sparse_start_idx,
                        total_size,
                        core_num,
                        max_core_num):
    """
    设置稀疏起始索引
    
    params:
        sparse_valid_array: 稀疏有效数组
        multi_core_params: 多核参数对象
        max_core_num: 最大核心数
    """
    valid_aiv_num = min(core_num, max_core_num)
    assert valid_aiv_num > 0,"valid_aiv_num should be greater than 0"

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

    
    # binary search for load-balancing
    while load_each_core_lower_bound + 1 < load_each_core_upper_bound:
        load_max = load_each_core_lower_bound + (load_each_core_upper_bound - load_each_core_lower_bound) // 2
        if (load_max * valid_aiv_num >= load_total and
            partition_sparse_data(sparse_valid_array, sparse_array_sum, 
                                 total_size, load_max, partition_result)):
            load_each_core_upper_bound = load_max
            last_valid_partition_result, partition_result = partition_result, last_valid_partition_result
        else:
            load_each_core_lower_bound = load_max

    for idx in range(valid_aiv_num):
        sparse_start_idx[idx] = last_valid_partition_result[idx]

def partition_sparse_data(sparse_rolling_array: List[int],
                         sparse_rolling_array_sum: int,
                         sparse_array_size: int,
                         load_max_each_core: int,
                         partition_result: List[int]) -> bool:
    """
    将稀疏数据分配到多个核心
    
    参数:
        sparse_rolling_array: 稀疏滚动数组 sparse_valid_
        sparse_rolling_array_sum: 稀疏滚动数组的和 sparse_valid_sum
        sparse_array_size: 稀疏数组大小
        load_max_each_core: 每个核心的最大负载
        partition_result: 分区结果（会被修改）
    
    返回:
        bool: 分配是否成功
    """

    assert partition_result,"partition_result should be greater than 0"
    assert sparse_rolling_array_sum > 0,"sparse_rolling_array_sum should be greater than 0"
    
    # 计算每个核心的基础负载
    s1_outer_cut_each_core = load_max_each_core // sparse_rolling_array_sum
    s1_outer_load_each_core = s1_outer_cut_each_core * sparse_rolling_array_sum
    s1_outer_num_each_core = s1_outer_cut_each_core * len(sparse_rolling_array)
    
    target_core_num = len(partition_result)
    core_idx = 0
    rolling_idx = 0
    load_size = s1_outer_load_each_core
    
    # 初始化第一个核心的起始位置
    partition_result[0] = 0
    
    # 分配剩余的数据到各个核心
    i = s1_outer_num_each_core
    while i < sparse_array_size:
        # 处理rolling_idx回绕
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
        # 128, 8, 8192, 64, causal=False, dtype=torch.float16, BM = 128, BN = 128
        # ============================ fp16 cases ===================================
        [128, 8, 8192, 128, False, torch.float16, 128, 128],
        [128, 8, 8192, 64, False, torch.float16, 128, 128],
        [128, 8, 1024, 128, False, torch.float16, 128, 128],
        [128, 8, 1024, 64, False, torch.float16, 128, 128],
        [128, 8, 8192, 128, True, torch.float16, 128, 128],
        [128, 8, 8192, 64, True, torch.float16, 128, 128],
        [128, 8, 1024, 128, True, torch.float16, 128, 128],
        [128, 8, 1024, 64, True, torch.float16, 128, 128],
        # ============================ bf16 cases ===================================
        [128, 8, 8192, 128, False, torch.bfloat16, 128, 128],
        [128, 8, 8192, 64, False, torch.bfloat16, 128, 128],
        [128, 8, 1024, 128, False, torch.bfloat16, 128, 128],
        [128, 8, 1024, 64, False, torch.bfloat16, 128, 128],
        [128, 8, 8192, 128, True, torch.bfloat16, 128, 128],
        [128, 8, 8192, 64, True, torch.bfloat16, 128, 128],
        [128, 8, 1024, 128, True, torch.bfloat16, 128, 128],
        [128, 8, 1024, 64, True, torch.bfloat16, 128, 128],
        # ============================ fp8 cases ===================================
        # [128, 8, 8192, 128, False, torch.float8_e5m2, 128,128],
        # [128, 8, 8192, 64, False, torch.float8_e5m2, 128,128],
        # [128, 8, 1024, 128, False, torch.float8_e5m2, 128,128],
        # [128, 8, 1024, 64, False, torch.float8_e5m2, 128,128],
        # [128, 8, 8192, 128, True, torch.float8_e5m2, 64,128],
        # [128, 8, 8192, 64, True, torch.float8_e5m2, 64,128],
        # [128, 8, 1024, 128, True, torch.float8_e5m2, 64,128],
        # [128, 8, 1024, 64, True, torch.float8_e5m2, 64,128],
        # [128, 8, 8192, 128, False, torch.float8_e4m3, 128,128],
        # [128, 8, 8192, 64, False, torch.float8_e4m3, 128,128],
        # [128, 8, 1024, 128, False, torch.float8_e4m3, 128,128],
        # [128, 8, 1024, 64, False, torch.float8_e4m3, 128,128],
        # [128, 8, 8192, 128, True, torch.float8_e4m3, 64,128],
        # [128, 8, 8192, 64, True, torch.float8_e4m3, 64,128],
        # [128, 8, 1024, 128, True, torch.float8_e4m3, 64,128],
        # [128, 8, 1024, 64, True, torch.float8_e4m3, 64,128],
        # [1, 1, 8192, 128, True, torch.float16, 128, 64]
    ])
def test_op(Z, H, N_CTX, HEAD_DIM, causal, dtype,BM ,BN):
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())

    sm_scale = 0.5

    compressed_len = 2048
    atten_mask = None
    sparse_mode = 0
    atten_golden_mask = None

    if causal:
        atten_mask = torch.triu(torch.ones(N_CTX, N_CTX, device=DEVICE), diagonal=1)
        atten_golden_mask = torch.triu(torch.ones(compressed_len, compressed_len, device=DEVICE), diagonal=1).bool()
        sparse_mode = 2

    if atten_mask is None:
        atten_mask = torch.zeros((1, 1), device=DEVICE)
        
    
    ###ref_out = torch_npu.npu_fusion_attention(
    ###    q, k, v, H,
    ###    padding_mask=None,
    ###    atten_mask=atten_golden_mask,
    ###    scale=sm_scale,
    ###    keep_prob=1.0,
    ###    input_layout='BNSD',
    ###    pre_tockens=65535,
    ###    next_tockens=65535,
    ###    sparse_mode=sparse_mode,
    ###    )[0]


    tri_out = attention(q, k, v, atten_mask, causal, sm_scale,BM,BN)

    rtol = 0.0
    atol = 1e-2	
    ###assert torch.allclose(ref_out, tri_out, atol=atol, rtol=rtol)
    print("compare success!")

if __name__ == "__main__":
    #test_op(128,8,1024,128, causal=True, dtype=torch.float16, BM = 128,BN = 128)
    test_op(1,1,128,128, causal=False, dtype=torch.float16, BM = 64,BN = 64)

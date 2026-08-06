"""
AOT-компиляция Triton-кернела ДО TTIR (без NPU/GPU, только CPU),
плюс отдельная точка вызова backend-компилятора (ttir -> linalg).

Идея: то, что обычно скрыто внутри triton.runtime.jit.JITFunction.run() /
triton.compiler.compiler.compile(), здесь развёрнуто руками
"""

import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from collections import namedtuple

# Модуль вашего Ascend backend-а (тот самый файл, который вы прислали)
from triton.backends.ascend.compiler import (
    AscendBackend,
    make_ttir,        # stages["ttir"]     -- то, до чего хотим дойти
    ttir_to_linalg,   # stages["ttadapter"] -- "backend compiler", вызываем отдельно
)


@triton.jit
def _vdv_atn_fwd_inner_opt(acc, l_i, m_i, q, #
                    K_block_ptr, V_block_ptr, ATTEN_MASK, #
                    stride_am, start_m, qk_scale: tl.constexpr,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr):
    lo, hi = 0, N_CTX
    # 0, 256, step 128
    # 0, 8192, step 128
    q_type : tl.constexpr = q.type
    K_block_ptr = tl.advance(K_block_ptr, (lo, 0))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N):#, loop_unroll_factor=2):
        # -- compute qk ----
        k = tl.load(K_block_ptr)      
        trans_k = tl.trans(k)
        qk = tl.dot(q, trans_k)
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
        # ------------------------------

        qk = qk * qk_scale
        m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
        qk = qk - m_ij[:, None]
        ###p = tl.exp(qk.cast(q_type))
        p = tl.math.exp(qk)
        p = p.cast(q_type)
        #orig_shape = p.shape
        ###p_cast = p.view([BLOCK_M * BLOCK_N]).cast(q.type) ###.to(q.type)
        ###p_cast = p_cast.view(p.shape)
        #p_cast = p.cast(q_type)
        ###
        v = tl.load(V_block_ptr)
        #pv = tl.dot(p_cast, v)
        pv = tl.dot(p, v)
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp(m_i - m_ij)
        m_i = m_ij 

        # -- update output accumulator --
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None] + pv
    return acc, l_i, m_i

@triton.jit
def _vdv_atn_fwd(Q, K, V, ATTEN_MASK, M, Out, sm_scale: tl.constexpr,  #
              stride_qz: tl.constexpr, stride_qh: tl.constexpr, stride_qm: tl.constexpr, stride_qk: tl.constexpr,  #
              stride_kz: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr, stride_kk: tl.constexpr,  #
              stride_vz: tl.constexpr, stride_vh: tl.constexpr, stride_vn: tl.constexpr, stride_vk: tl.constexpr,  #
              stride_oz: tl.constexpr, stride_oh: tl.constexpr, stride_om: tl.constexpr, stride_on: tl.constexpr,  #
              stride_am: tl.constexpr,
              Z: tl.constexpr,
              H: tl.constexpr, 
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
    
    start_block, end_block, step = pid, NUM_BLOCKS, AICORE_NUM
    #start_block, end_block, step = pid*NUM_BLOCKS_PER_CORE,pid*NUM_BLOCKS_PER_CORE + NUM_BLOCKS_PER_CORE,1
    #if end_block > NUM_BLOCKS:
    #    end_block = NUM_BLOCKS    
    ###for block_idx in range(start_block, end_block, step):
    for block_idx in tl.range(start_block, end_block, step):
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M
        off_z = task_hz_idx // H
        off_h = task_hz_idx % H
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        Q_block_ptr = tl.make_block_ptr(
            base=Q + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_qm, stride_qk),

            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + qvk_offset,

            shape=(N_CTX, HEAD_DIM),
            strides=(stride_vn, stride_vk),

            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_kn, stride_kk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            base=Out + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_om, stride_on),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )

        # initialize offsets
        offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        # load scales

        # load q: it will stay in SRAM throughout
        q = tl.load(Q_block_ptr)
        acc, l_i, m_i = _vdv_atn_fwd_inner_opt(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, ATTEN_MASK, #
                                        stride_am, task_m_idx, sm_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                        )
        
        # epilogue
        m_i += tl.math.log(l_i)
        acc = acc / l_i[:, None]
        m_ptrs = M + task_hz_idx * N_CTX + offs_m

        ###tl.store(m_ptrs, m_i)
        tl.store(O_block_ptr, acc.to(Out.type.element_ty))


def main():
    # -----------------------------------------------------------------------
    # 2. Явная специализация кернела (то, что обычно JIT делает сам по
    #    типам реальных аргументов при вызове add_kernel[grid](x, y, ...))
    #    Здесь мы задаём это руками, без единого реального тензора/девайса.
    # -----------------------------------------------------------------------
    signature = {
        "Q": "*fp16",
        "K": "*fp16",
        "V": "*fp16",
        "ATTEN_MASK": "*fp32",
        "M": "*fp32",
        "Out": "*fp16",
        "sm_scale": "constexpr",
        "stride_qz": "constexpr",
        "stride_qh": "constexpr",
        "stride_qm": "constexpr",
        "stride_qk": "constexpr",
        "stride_kz": "constexpr",
        "stride_kh": "constexpr",
        "stride_kn": "constexpr",
        "stride_kk": "constexpr",
        "stride_vz": "constexpr",
        "stride_vh": "constexpr",
        "stride_vn": "constexpr", 
        "stride_vk": "constexpr",
        "stride_oz": "constexpr",
        "stride_oh": "constexpr",
        "stride_om": "constexpr",
        "stride_on": "constexpr",
        "stride_am": "constexpr",
        "Z": "constexpr",
        "H": "constexpr", 
        "N_CTX": "constexpr",  #
        "HEAD_DIM": "constexpr",  #
        "BLOCK_M": "constexpr",  #
        "BLOCK_N": "constexpr",  #
        "STAGE": "constexpr",  #
        "NUM_BLOCKS_PER_CORE": "constexpr",
        "NUM_BLOCKS": "constexpr",
        "NUM_BLOCKS_M": "constexpr",
        "AICORE_NUM": "constexpr",
    }
    Z=128
    H=8
    N_CTX=8192
    HEAD_DIM=128
    sm_scale=0.5
    AICORE_NUM=28
    BM=128
    BN=128
    
    num_cores=AICORE_NUM
    NUM_BLOCKS_M=triton.cdiv(N_CTX, BM)
    NUM_BLOCKS=NUM_BLOCKS_M*Z*H
    num_cores = num_cores if NUM_BLOCKS > num_cores else NUM_BLOCKS
    NUM_BLOCKS_PER_CORE=triton.cdiv(NUM_BLOCKS, num_cores)
    
    constexprs = {
        "sm_scale": sm_scale,
        "stride_qz": H * N_CTX * HEAD_DIM,
        "stride_qh": N_CTX * HEAD_DIM,
        "stride_qm": HEAD_DIM,
        "stride_qk": 1,
        "stride_kz": H * N_CTX * HEAD_DIM,
        "stride_kh": N_CTX * HEAD_DIM,
        "stride_kn": HEAD_DIM,
        "stride_kk": 1,
        "stride_vz": H * N_CTX * HEAD_DIM,
        "stride_vh": N_CTX * HEAD_DIM,
        "stride_vn": HEAD_DIM, 
        "stride_vk": 1,
        "stride_oz": H * N_CTX * HEAD_DIM,
        "stride_oh": N_CTX * HEAD_DIM,
        "stride_om": HEAD_DIM,
        "stride_on": 1,
        "stride_am": N_CTX,
        "Z": Z,
        "H": H, 
        "N_CTX": N_CTX,  #
        "HEAD_DIM": HEAD_DIM,  #
        "BLOCK_M": BM,  #
        "BLOCK_N": BN,  #
        "STAGE": 1,  #
        "NUM_BLOCKS_PER_CORE": NUM_BLOCKS_PER_CORE,
        "NUM_BLOCKS": NUM_BLOCKS,
        "NUM_BLOCKS_M": NUM_BLOCKS_M,
        "AICORE_NUM": num_cores,
    }

    # 1. Get the exact ordered list of arguments from your signature
    signature_keys = list(signature.keys())
    
    # 2. Define which tensors need the 16-byte alignment attribute
    target_pointers = ["Q", "K", "V", "ATTEN_MASK", "M", "Out"]
    
    # 3. Create a dictionary mapped exactly to Triton 3.6's structural layout requirements
    # Path is a tuple index (e.g., (0,) for the first parameter)
    # The value MUST be a list of (attribute_name, attribute_value) tuples
    structured_attrs_map = {}
    for name in target_pointers:
        if name in signature_keys:
            arg_index = signature_keys.index(name)
            # Triton 3.6 maps flat function positions to tuple paths: (index,)
            # It expects MLIR dialect attribute strings like "tt.divisible_by"
            structured_attrs_map[(arg_index,)] = [("tt.divisible_by", 16)]

    # 4. Construct a unified mock object that satisfies both direct lookups 
    # and .get(path) style code queries without breaking.
    class Triton36CompilerAttrsMock:
        def __init__(self, mapping_dict):
            self._mapping = mapping_dict
            # Fallback direct fields just in case a secondary pass checks them
            self.divisible_by_16 = tuple(path[0] for path in mapping_dict.keys())
            self.equal_to_1 = ()

        def get(self, key, default=None):
            # Triton 3.6 calls code_generator.py: self.attrs.get(path, [])
            # where key is the tuple path (e.g., (0,))
            return self._mapping.get(key, default if default is not None else [])

    # Instantiate the working mock object
    attrs_mock = Triton36CompilerAttrsMock(structured_attrs_map)    
    
    # "arch" здесь может быть произвольной строкой на этом этапе — она нужна
    # реальному bishengir-compile'у позже, а не фронтенду/пассам ttir.
    target = GPUTarget(backend="npu", arch="Ascend950PR_958b", warp_size=32)

    backend = AscendBackend(target)
    options = backend.parse_options({"debug": True, "compile_on_910_95":True, "enable_dynamic_cv_pipeline": True})  # NPUOptions с дефолтами
    for key, value in vars(options).items():
        print(f"{key}: {value}")

    src = triton.compiler.ASTSource(
        fn=_vdv_atn_fwd,          # именно JITFunction, не "голая" python-функция
        signature=signature,
        constexprs=constexprs,
        attrs=attrs_mock,
    )

    # -----------------------------------------------------------------------
    # 3. То, что compile() делает перед стадиями: контекст + диалекты
    # -----------------------------------------------------------------------
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    codegen_fns = backend.get_codegen_implementation(options)
    module_map = backend.get_module_map()

    # AST -> "черновой" ttir (ещё без общих оптимизационных пассов)
    module = src.make_ir(target, options, codegen_fns, module_map, context)

    # -----------------------------------------------------------------------
    # 4. Ровно stages["ttir"] — общий для всех backend-ов проход пассов.
    #    Это и есть финальный TTIR, который в штатном compile() ушёл бы
    #    дальше в backend. Мы просто вызываем его руками и останавливаемся.
    # -----------------------------------------------------------------------
    metadata = {}
    module = make_ttir(module, metadata, options)

    ttir_text = str(module)
    print("=" * 80)
    print("TTIR (то, что дошло бы до backend-компилятора):")
    print("=" * 80)
    #print(ttir_text)

    with open("kernel.ttir.mlir", "w") as f:
        f.write(ttir_text)
    print("[ok] TTIR сохранён в kernel.ttir.mlir")
    ###with open("./kernel.ttir", "r") as f:
    ###    mlir_text = f.read()
    ###module = ir.parse_mlir_module(mlir_text, context)
    # -----------------------------------------------------------------------
    # 5. Точка вызова backend-компилятора отдельно ("ttadapter" стадия).
    #    Именно здесь ttir_to_linalg дёргает triton-adapter-opt / готовит
    #    вход для bishengir-compile. На чисто-CPU машине без установленного
    #    ascend-тулчейна (triton-adapter-opt, bishengir-compile и т.п.)
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("Пробуем вызвать backend-компилятор отдельно (ttadapter stage)...")
    print("=" * 80)

    # ttir_to_linalg читает много полей metadata как metadata["..."] (без .get),
    # поэтому нужно предварительно заполнить дефолты NPUOptions в metadata,
    # как это делает compile() перед прогоном стадий: metadata = {**options.__dict__, ...}
    metadata.update(options.__dict__)
    metadata.setdefault("hash", "manual-aot-run")

    try:
        linalg_ir = ttir_to_linalg(module, metadata, options, named_ops=True)
        print("[ok] Получен linalg IR, backend-toolchain доступен в этой среде.")
        with open("kernel.ttadapter.mlir", "w") as f:
            f.write(linalg_ir)
        print("[ok] Сохранён в kernel.ttadapter.mlir")
    except Exception as e:
        print(f"[expected on CPU-only машине без ascend toolchain] {type(e).__name__}: {e}")
        print("Это нормально — до сюда AOT дошёл без единого реального устройства,")
        print("а дальше нужен уже сам ascend-компилятор (bishengir-compile и т.п.).")


if __name__ == "__main__":
    main()

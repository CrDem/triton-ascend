"""
Test Case: main_loop hint - explicit choice of the loop the dynamic CV pipeline
is built around.

Covers `tl.range(..., main_loop=True/False)`:
  * the hint reaches the ssbuffer passes as `tt.main_loop` on the scf.for op;
  * MarkMainLoopPass tags the hinted (outer) loop instead of the innermost one;
  * `main_loop=False` takes a loop out of the candidates;
  * without a hint the innermost candidate is still the one that gets tagged.

MarkMainLoopPass runs inside add_dynamic_cv_pipeline and its `ssbuffer.main_loop`
attribute is stripped again by remove-ssbuf-attr, so the test drives the pass
pipeline up to mark-main-loop with triton-opt and inspects the IR there.
"""

import os
import subprocess
import tempfile

import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.backends.compiler import GPUTarget
from triton.backends.ascend.compiler import AscendBackend, NPUOptions, make_ttir
from triton.backends.ascend.utils import _get_triton_opt_path
from triton.compiler.compiler import ASTSource

# triton -> linalg, i.e. everything add_dynamic_cv_pipeline runs after.
_TO_LINALG = (
    "triton-control-flow-opt,"
    "triton-to-structured{enable-mask-fallback-conversion=false optimize-dynamic-offset=false},"
    "discrete-mask-access-conversion{compile-mode=simd_simt_template compile-on-910-95=true},"
    "triton-to-annotation,"
    "triton-to-unstructure{compile-mode=simd_simt_template compile-on-910-95=true force-scalarize-mode=false},"
    "triton-to-hivm,triton-to-hfusion,triton-to-llvm,"
    "bubble-up-operation{enable-aggressive-mode=true},"
    "triton-to-structured{enable-mask-fallback-conversion=false optimize-dynamic-offset=false},"
    "triton-to-linalg{compile-mode=simd_simt_template compile-on-910-95=true "
    "enable-nd2nz-on-vector=false enable-select-analysis=true global-kernel=false named-ops=true},"
    "merge-concat-load-buffer"
)
# ssbuffer sub-passes up to and including mark-main-loop.
_TO_MARK_MAIN_LOOP = (
    "pre-check-dynamic-cv-pipeline-available,ssbuf-standardize-op,plan-compute-block,"
    "compute-block-opt,ssbuf-check-unsupported-scenario,add-block-id-for-control-ops,"
    "data-dependency-analysis,inter-core-transfer-and-sync,mark-main-loop"
)

_SIG = {"A": "*fp16", "B": "*fp16", "C": "*fp32", "M": "i32", "K_ITERS": "i32",
        "BLOCK": "constexpr"}


def _make_ttir(kernel, signature, constants, arch="Ascend910_9589"):
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    options = NPUOptions(arch=arch, enable_dynamic_cv_pipeline=True)
    target = GPUTarget("npu", arch, 32)
    backend = AscendBackend(target)
    backend.load_dialects(context)
    src = ASTSource(kernel, signature, constants)
    module = src.make_ir(target, options, backend.get_codegen_implementation(options),
                         backend.get_module_map(), context)
    return str(make_ttir(module, {}, options))


def _run_pipeline(ttir_text, pipeline):
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "kernel.ttir.mlir")
        dst = os.path.join(tmp, "out.mlir")
        with open(src, "w") as f:
            f.write(ttir_text)
        proc = subprocess.run(
            [_get_triton_opt_path(), src, f"--pass-pipeline=builtin.module({pipeline})",
             "-o", dst],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr
        with open(dst) as f:
            return f.read()


def _main_loop_indents(mlir_text):
    """Indentation of every line carrying ssbuffer.main_loop (deeper = inner loop)."""
    return [len(line) - len(line.lstrip())
            for line in mlir_text.split("\n") if "ssbuffer.main_loop" in line]


# ---------------------------------------------------------------------------
# Kernels: outer loop over M tiles, inner loop with cube + vector traffic.
# ---------------------------------------------------------------------------
@triton.jit
def _nested_plain(A, B, C, M, K_ITERS, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    for m in range(0, M):
        acc = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
        for k in range(0, K_ITERS):
            a = tl.load(A + (m * BLOCK + offs[:, None]) * BLOCK + offs[None, :])
            b = tl.load(B + (k * BLOCK + offs[:, None]) * BLOCK + offs[None, :])
            acc += tl.dot(a, b)
        tl.store(C + (m * BLOCK + offs[:, None]) * BLOCK + offs[None, :], acc)


@triton.jit
def _nested_outer_hinted(A, B, C, M, K_ITERS, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    for m in tl.range(0, M, main_loop=True):
        acc = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
        for k in range(0, K_ITERS):
            a = tl.load(A + (m * BLOCK + offs[:, None]) * BLOCK + offs[None, :])
            b = tl.load(B + (k * BLOCK + offs[:, None]) * BLOCK + offs[None, :])
            acc += tl.dot(a, b)
        tl.store(C + (m * BLOCK + offs[:, None]) * BLOCK + offs[None, :], acc)


@triton.jit
def _single_opted_out(A, B, C, M, K_ITERS, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
    for k in tl.range(0, K_ITERS, main_loop=False):
        a = tl.load(A + offs[:, None] * BLOCK + offs[None, :])
        b = tl.load(B + (k * BLOCK + offs[:, None]) * BLOCK + offs[None, :])
        acc += tl.dot(a, b)
    tl.store(C + offs[:, None] * BLOCK + offs[None, :], acc)


def test_main_loop_hint_survives_lowering():
    """tl.range(..., main_loop=True) must reach the ssbuffer passes as tt.main_loop."""
    ttir = _make_ttir(_nested_outer_hinted, _SIG, {"BLOCK": 128})
    assert "tt.main_loop = 1" in ttir, "hint lost in TTIR"

    linalg = _run_pipeline(ttir, _TO_LINALG)
    assert "tt.main_loop = 1" in linalg, "hint lost on the way to linalg"


def test_main_loop_hint_moves_pipeline_outwards():
    """The hinted outer loop is marked instead of the innermost candidate."""
    pipeline = _TO_LINALG + "," + _TO_MARK_MAIN_LOOP
    plain = _run_pipeline(_make_ttir(_nested_plain, _SIG, {"BLOCK": 128}), pipeline)
    hinted = _run_pipeline(_make_ttir(_nested_outer_hinted, _SIG, {"BLOCK": 128}), pipeline)

    plain_indents = _main_loop_indents(plain)
    hinted_indents = _main_loop_indents(hinted)
    assert plain_indents, "default behaviour marked nothing"
    assert hinted_indents, "hinted kernel marked nothing"
    assert min(hinted_indents) < min(plain_indents), (
        f"hint did not move the main loop outwards: {hinted_indents} vs {plain_indents}"
    )


def test_main_loop_opt_out_removes_candidate():
    """main_loop=False on the only candidate loop leaves nothing marked."""
    ttir = _make_ttir(_single_opted_out, _SIG, {"BLOCK": 128})
    assert "tt.main_loop = 0" in ttir

    marked = _run_pipeline(ttir, _TO_LINALG + "," + _TO_MARK_MAIN_LOOP)
    assert "ssbuffer.main_loop" not in marked, "opted-out loop was still marked"

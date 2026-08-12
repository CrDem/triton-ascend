// Sanity check of the input: the CUBE -> VECTOR dependency inside the loop is
// what makes inter-core-transfer-and-sync put a transfer there and
// mark-main-loop mark that loop. That is the loop the unroll has to pick, and
// it locates it by running exactly these passes on a clone of the module.
// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s --check-prefix=PROBE

// RUN: triton-opt --main-loop-unroll="unroll-factor=2" %s | FileCheck %s
// RUN: triton-opt --main-loop-unroll %s | FileCheck %s --check-prefix=NOUNROLL

// PROBE-LABEL: func.func @main_loop_unroll
// PROBE: hivm.hir.fixpipe
// PROBE: ssbuffer.main_loop

// The main loop is unrolled by 2; the trip count (4) is a multiple of the
// factor, so there is no epilogue loop. Compute block ids of the second copy
// are shifted past the ids in use (max id 2, so the stride is 3), giving that
// copy compute blocks of its own. The probe tags must not survive the pass.
// CHECK-LABEL: func.func @main_loop_unroll
// CHECK: scf.for
// CHECK-COUNT-2: linalg.matmul
// CHECK-SAME: ssbuffer.block_id = 4
// CHECK-NOT: scf.for
// CHECK-NOT: ssbuffer.main_loop_probe

// Without a factor (default 1) the module is left untouched.
// NOUNROLL-LABEL: func.func @main_loop_unroll
// NOUNROLL-COUNT-1: linalg.matmul
// NOUNROLL-NOT: linalg.matmul

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @main_loop_unroll(%arg0: memref<32x32xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
    %ta = bufferization.to_tensor %arg0 restrict writable {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<32x32xf32> to tensor<32x32xf32>
    %empty = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %fill = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
    %init = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    %sum = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> (tensor<32x32xf32>) {
      %mat = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%ta, %ta : tensor<32x32xf32>, tensor<32x32xf32>) outs(%fill : tensor<32x32xf32>) -> tensor<32x32xf32>
      %vec = math.exp %mat {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
      %add = arith.addf %acc, %vec {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
      scf.yield %add : tensor<32x32xf32>
    }
    return
  }
}

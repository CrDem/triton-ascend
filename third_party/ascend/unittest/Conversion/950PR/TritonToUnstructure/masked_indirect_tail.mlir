// RUN: triton-opt --triton-to-structured '--triton-to-unstructure=compile-on-910-95=True compile-mode=simd_simt_template' %s | FileCheck %s

// A dynamic prefix mask can leave inactive lanes in an aligned tile. Keep the
// access on the bounded scalar-loop path rather than issuing a template-SIMT
// indirect load over the complete tile.
tt.func public @masked_indirect_tail(%base: !tt.ptr<i64>,
                                     %indices: !tt.ptr<i64>,
                                     %limit: i32) -> tensor<32xi64> {
  %c0 = arith.constant dense<0> : tensor<32xi64>
  %range = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %limit_splat = tt.splat %limit : i32 -> tensor<32xi32>
  %mask = arith.cmpi slt, %range, %limit_splat : tensor<32xi32>
  %indices_splat = tt.splat %indices : !tt.ptr<i64> -> tensor<32x!tt.ptr<i64>>
  %index_ptrs = tt.addptr %indices_splat, %range
    : tensor<32x!tt.ptr<i64>>, tensor<32xi32>
  %offsets = tt.load %index_ptrs : tensor<32x!tt.ptr<i64>>
  %base_splat = tt.splat %base : !tt.ptr<i64> -> tensor<32x!tt.ptr<i64>>
  %ptrs = tt.addptr %base_splat, %offsets
    : tensor<32x!tt.ptr<i64>>, tensor<32xi64>
  %result = tt.load %ptrs, %mask, %c0 : tensor<32x!tt.ptr<i64>>
  tt.return %result : tensor<32xi64>
}

// CHECK-LABEL: tt.func public @masked_indirect_tail
// CHECK-NOT: ascend.indirect_load
// CHECK: scf.for
// CHECK: tt.load

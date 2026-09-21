// RUN: triton-opt --debug-only=memory-effects-tracker --plan-compute-block %s 2>&1 | FileCheck %s

// Test that operations producing memref results (like memref.memory_space_cast)
// do not register as lastWriter on the memref slot.
// When memref.memory_space_cast appears BEFORE an scf.for loop containing an unknown memeffect op (like func.call),
// the loop (barrier) should not create spurious WAW/WAR dependency edges to the cast.

module {
  func.func private @unknown_side_effect_call(%arg0: memref<64xf32>)

  // CHECK-LABEL: func.func @memspace_cast_before_unknown_effect_loop
  // CHECK: %[[ALLOC:[A-Za-z0-9_]+]] = memref.alloc()
  // CHECK: %[[CAST:[A-Za-z0-9_]+]] = memref.memory_space_cast %[[ALLOC]]
  // CHECK: Analyzing op scf.for
  // CHECK: [memory-effects-tracker] Defs:
  // CHECK-NEXT: [memory-effects-tracker] Preds:
  // CHECK-NEXT: [memory-effects-tracker] %[[ALLOC]] = memref.alloc()
  // CHECK: Analyzing op func.call @unknown_side_effect_call
  // CHECK-NEXT: [memory-effects-tracker] Defs:
  // CHECK-NEXT: [memory-effects-tracker] Preds:
  // CHECK-NEXT: [memory-effects-tracker] %[[ALLOC]] = memref.alloc()
  func.func @memspace_cast_before_unknown_effect_loop(
      %arg0: memref<64xf32>,
      %rhs: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %zero = arith.constant 0.0 : f32

    %alloc = memref.alloc() : memref<64x64xf32>

    // memref.memory_space_cast produces a memref but does not write to it.
    // It should not be marked as lastWriter, and thus the subsequent loop with unknown memory effects
    // should not have spurious WAW/WAR dependency edges to the cast.
    %cast = memref.memory_space_cast %alloc : memref<64x64xf32> to memref<64x64xf32>

    // Loop containing unknown memory effect op (func.call)
    scf.for %iv = %c0 to %c4 step %c1 {
      func.call @unknown_side_effect_call(%arg0) : (memref<64xf32>) -> ()
    }

    %lhs = bufferization.to_tensor %cast restrict writable : memref<64x64xf32> to tensor<64x64xf32>
    %out = tensor.empty() : tensor<64x64xf32>
    %init = linalg.fill ins(%zero : f32) outs(%out : tensor<64x64xf32>) -> tensor<64x64xf32>
    %mm = linalg.matmul ins(%lhs, %rhs : tensor<64x64xf32>, tensor<64x64xf32>)
      outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %mm : tensor<64x64xf32>
  }
}

// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: log_op_test_bf16
func.func @log_op_test_bf16() -> tensor<2xbf16> {
  %0 = stablehlo.constant dense<[1.0, 0.125]> : tensor<2xbf16>
  %1 = stablehlo.log %0 : tensor<2xbf16>
  func.return %1 : tensor<2xbf16>
  // CHECK-NEXT: tensor<2xbf16>
  // CHECK-NEXT: 0.000000e+00 : bf16
  // CHECK-NEXT: -2.078130e+00 : bf16
}

// -----

// CHECK-LABEL: Evaluated results of function: log_op_test_c64
func.func @log_op_test_c64() -> tensor<2xcomplex<f32>> {
  %0 = stablehlo.constant dense<[(1.5, 2.5), (3.5, 4.5)]> : tensor<2xcomplex<f32>>
  %1 = stablehlo.log %0 : tensor<2xcomplex<f32>>
  func.return %1 : tensor<2xcomplex<f32>>
  // CHECK-NEXT: tensor<2xcomplex<f32>>
  // CHECK-NEXT: [1.07003307 : f32, 1.03037679 : f32]
  // CHECK-NEXT: [1.740620e+00 : f32, 0.909753143 : f32]
}

// -----

// CHECK-LABEL: Evaluated results of function: log_op_test_c128
func.func @log_op_test_c128() -> tensor<2xcomplex<f64>> {
  %0 = stablehlo.constant dense<[(1.5, 2.5), (3.5, 4.5)]> : tensor<2xcomplex<f64>>
  %1 = stablehlo.log %0 : tensor<2xcomplex<f64>>
  func.return %1 : tensor<2xcomplex<f64>>
  // CHECK-NEXT: tensor<2xcomplex<f64>>
  // CHECK-NEXT: [1.070033081748{{[0-9]+}} : f64, 1.030376826524{{[0-9]+}} : f64]
  // CHECK-NEXT: [1.740620044667{{[0-9]+}} : f64, 0.909753157944{{[0-9]+}} : f64]
}
// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=0.3.0' --verify-diagnostics --split-input-file %s

func.func @custom_call_v2_with_output_operand_aliases(%arg0 : tensor<f32>) -> tensor<f32> {
  // expected-error @+2 {{failed to downgrade vhlo.custom_call_v2, op has a non-empty output_operand_aliases attribute}}
  // expected-error @+1 {{failed to legalize operation 'vhlo.custom_call_v2' that was explicitly marked illegal}}
  %0 = "stablehlo.custom_call"(%arg0) {
    call_target_name = "foo",
    has_side_effect = false,
    backend_config = "",
    api_version = 1 : i32,
    called_computations = [@foo],
    operand_layouts = [dense<> : tensor<0xindex>],
    output_operand_aliases = [
      #stablehlo.output_operand_alias<output_tuple_indices = [],
                                 operand_index = 0,
                                 operand_tuple_indices = []>],
    result_layouts = [dense<> : tensor<0xindex>]
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// Unregistered attributes are permitted so long as their value is
// representable and valid in VHLO in the target version.
func.func @custom_call_v2_with_output_operand_aliases_unregistered(%arg0 : tensor<f32>) -> tensor<f32> {
  // expected-error @+1 {{failed to legalize operation 'vhlo.custom_call_v2' that was explicitly marked illegal}}
  %0 = "stablehlo.custom_call"(%arg0) {
    call_target_name = "foo",
    output_operand_aliases_unregistered =
      #stablehlo.output_operand_alias<output_tuple_indices = [],
                                 operand_index = 0,
                                 operand_tuple_indices = []>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

func.func @op_collective_permute(%arg0: tensor<16x8xf32>) -> tensor<16x8xf32> {
  // expected-error @+2 {{failed to downgrade vhlo.collective_permute_v2, op has a non-empty channel_handle attribute}}
  // expected-error @+1 {{failed to legalize operation 'vhlo.collective_permute_v2' that was explicitly marked illegal}}
  %0 = "stablehlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
  } : (tensor<16x8xf32>) -> tensor<16x8xf32>
  func.return %0 : tensor<16x8xf32>
}

// -----

func.func @op_all_gather(%arg0: tensor<16x8xf32>) -> tensor<16x16xf32> {
  // expected-error @+2 {{failed to downgrade vhlo.all_gather_v2, op has a non-empty use_global_device_ids attribute}}
  // expected-error @+1 {{failed to legalize operation 'vhlo.all_gather_v2' that was explicitly marked illegal}}
  %0 = "stablehlo.all_gather"(%arg0) {
    all_gather_dim = 1 : i64,
    replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
    use_global_device_ids
  } : (tensor<16x8xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}

// -----

func.func @op_partition_id() -> tensor<ui32> {
  // expected-error @+1 {{failed to legalize operation 'vhlo.partition_id' that was explicitly marked illegal}}
  %0 = "stablehlo.partition_id"() : () -> tensor<ui32>
  func.return %0 : tensor<ui32>
}

// -----

// This test emulates two things:
//   1. A file that is too old and no longer supported on consumer.
//   2. A file that is too new and not yet supported on consumer.
// More work should be done to improve this error message.
func.func @invalid_program_unknown_op(%arg0 : tensor<f32>) -> (tensor<f32>) {
  // expected-error @+1 {{unregistered operation 'vhlo.unknown_op' found in dialect ('vhlo') that does not allow unknown operations}}
  %0 = "vhlo.unknown_op"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return
}

// -----

func.func @all_to_all_to_v1(%arg0: tensor<4x16xf32>) -> tensor<16x4xf32> {
  // expected-error @+2 {{failed to downgrade vhlo.all_to_all_v2, op has a non-empty channel_handle attribute}}
  // expected-error @+1 {{failed to legalize operation 'vhlo.all_to_all_v2' that was explicitly marked illegal}}
  %0 = "stablehlo.all_to_all"(%arg0) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  func.return %0 : tensor<16x4xf32>
}


// -----

// expected-error @+1 {{failed to legalize operation 'vhlo.func' that was explicitly marked illegal}}
func.func @type_fp8_E5M2(%arg0: tensor<f8E5M2>) -> tensor<f8E5M2> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<f8E5M2>
  func.return %0 : tensor<f8E5M2>
}

// -----

// expected-error @+1 {{failed to legalize operation 'vhlo.func' that was explicitly marked illegal}}
func.func @type_fp8_E4M3FN(%arg0: tensor<f8E4M3FN>) -> tensor<f8E4M3FN> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<f8E4M3FN>
  func.return %0 : tensor<f8E4M3FN>
}

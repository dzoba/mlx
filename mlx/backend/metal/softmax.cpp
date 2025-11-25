// Copyright © 2023-2024 Apple Inc.
#include <algorithm>

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"

namespace mlx::core {

constexpr int SOFTMAX_LOOPED_LIMIT = 4096;

void Softmax::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  if (!issubdtype(out.dtype(), floating)) {
    throw std::runtime_error(
        "[softmax] Does not support non-floating point types.");
  }
  auto& s = stream();
  auto& d = metal::device(s.device);

  // Make sure that the last dimension is contiguous
  auto set_output = [&s, &out](const array& x) {
    if (x.flags().contiguous && x.strides()[x.ndim() - 1] == 1) {
      if (x.is_donatable()) {
        out.copy_shared_buffer(x);
      } else {
        out.set_data(
            allocator::malloc(x.data_size() * x.itemsize()),
            x.data_size(),
            x.strides(),
            x.flags());
      }
      return x;
    } else {
      array x_copy = contiguous_copy_gpu(x, s);
      out.copy_shared_buffer(x_copy);
      return x_copy;
    }
  };

  const array in = set_output(inputs[0]);

  int axis_size = in.shape().back();
  int n_rows = in.data_size() / axis_size;

  const int simd_size = 32;
  const int n_reads = SOFTMAX_N_READS;
  const int looped_limit = SOFTMAX_LOOPED_LIMIT;

  std::string kernel_name = (axis_size > looped_limit) ? "looped_" : "block_";
  kernel_name += "softmax_";
  if (in.dtype() != float32 && precise_) {
    kernel_name += "precise_";
  }
  kernel_name += type_to_name(out);

  auto kernel = get_softmax_kernel(d, kernel_name, precise_, out);
  auto& compute_encoder = d.get_command_encoder(s.index);
  {
    MTL::Size grid_dims, group_dims;
    if (axis_size <= looped_limit) {
      size_t threadgroup_needed = (axis_size + n_reads - 1) / n_reads;
      size_t simds_needed = (threadgroup_needed + simd_size - 1) / simd_size;
      size_t threadgroup_size = simd_size * simds_needed;
      assert(threadgroup_size <= kernel->maxTotalThreadsPerThreadgroup());
      size_t n_threads = n_rows * threadgroup_size;
      grid_dims = MTL::Size(n_threads, 1, 1);
      group_dims = MTL::Size(threadgroup_size, 1, 1);
    } else {
      size_t threadgroup_size = kernel->maxTotalThreadsPerThreadgroup();
      size_t n_threads = n_rows * threadgroup_size;
      grid_dims = MTL::Size(n_threads, 1, 1);
      group_dims = MTL::Size(threadgroup_size, 1, 1);
    }

    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(in, 0);
    compute_encoder.set_output_array(out, 1);
    compute_encoder.set_bytes(axis_size, 2);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  }
}

namespace fast {

void SoftmaxVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 2);
  assert(outputs.size() == 1);

  auto& s_arr = inputs[0]; // softmax output
  auto& g_arr = inputs[1]; // gradient
  auto& out = outputs[0];

  if (!issubdtype(out.dtype(), floating)) {
    throw std::runtime_error(
        "[softmax_vjp] Does not support non-floating point types.");
  }

  auto& str = stream();
  auto& d = metal::device(str.device);

  // Make sure that the last dimension is contiguous
  auto check_input = [&str](const array& x) -> std::pair<array, bool> {
    if (x.flags().contiguous && x.strides()[x.ndim() - 1] == 1) {
      return {x, false};
    }
    array x_copy = contiguous_copy_gpu(x, str);
    return {x_copy, true};
  };

  auto [s_in, s_copied] = check_input(s_arr);
  auto [g_in, g_copied] = check_input(g_arr);

  // Allocate output
  if (s_in.is_donatable()) {
    out.copy_shared_buffer(s_in);
  } else if (g_in.is_donatable()) {
    out.copy_shared_buffer(g_in);
  } else {
    out.set_data(
        allocator::malloc(out.data_size() * out.itemsize()),
        out.data_size(),
        s_in.strides(),
        s_in.flags());
  }

  // Clean up temporaries
  if (s_copied) {
    d.add_temporary(s_in, str.index);
  }
  if (g_copied) {
    d.add_temporary(g_in, str.index);
  }

  int axis_size = s_in.shape().back();
  int n_rows = s_in.data_size() / axis_size;

  const int simd_size = 32;
  const int n_reads = SOFTMAX_N_READS;
  const int looped_limit = SOFTMAX_LOOPED_LIMIT;

  // Use precise mode for non-float32 types (same as forward pass)
  bool precise = s_in.dtype() != float32;

  std::string kernel_name = (axis_size > looped_limit) ? "looped_" : "block_";
  kernel_name += "softmax_vjp_";
  if (precise) {
    kernel_name += "precise_";
  }
  kernel_name += type_to_name(out);

  auto kernel = get_softmax_kernel(d, kernel_name, precise, out);
  auto& compute_encoder = d.get_command_encoder(str.index);
  {
    MTL::Size grid_dims, group_dims;
    if (axis_size <= looped_limit) {
      size_t threadgroup_needed = (axis_size + n_reads - 1) / n_reads;
      size_t simds_needed = (threadgroup_needed + simd_size - 1) / simd_size;
      size_t threadgroup_size = simd_size * simds_needed;
      assert(threadgroup_size <= kernel->maxTotalThreadsPerThreadgroup());
      size_t n_threads = n_rows * threadgroup_size;
      grid_dims = MTL::Size(n_threads, 1, 1);
      group_dims = MTL::Size(threadgroup_size, 1, 1);
    } else {
      size_t threadgroup_size = kernel->maxTotalThreadsPerThreadgroup();
      size_t n_threads = n_rows * threadgroup_size;
      grid_dims = MTL::Size(n_threads, 1, 1);
      group_dims = MTL::Size(threadgroup_size, 1, 1);
    }

    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(s_in, 0);
    compute_encoder.set_input_array(g_in, 1);
    compute_encoder.set_output_array(out, 2);
    compute_encoder.set_bytes(axis_size, 3);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  }
}

} // namespace fast

} // namespace mlx::core

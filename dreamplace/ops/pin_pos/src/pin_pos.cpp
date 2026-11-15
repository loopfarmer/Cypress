/**
 * @file   pin_pos.cpp
 * @author Yibo Lin
 * @date   Aug 2019
 * @brief  Given cell locations, compute pin locations on CPU
 */
#include "utility/src/torch.h"
#include "utility/src/utils.h"
#include <cmath>

DREAMPLACE_BEGIN_NAMESPACE

// edaroute: rotation, ported from the CUDA op. A part turns by theta about its centre; pin offsets are from its
// lower-left corner in the unturned frame, w and h its unturned size:
//   pin_x = (x + w/2) + (ofx - w/2) cos(theta) - (ofy - h/2) sin(theta)
//   pin_y = (y + h/2) + (ofx - w/2) sin(theta) + (ofy - h/2) cos(theta)
// A part at exactly theta = 0 takes the old pin_x = x + ofx, which is the same value without the rounding of the
// centre form, so placements without rotation are unchanged bit for bit. theta covers the physical nodes; fillers have
// no pins.
// edaroute: (x, y) is the lower-left corner of the part's box as turned, as the macro legalizer and Bookshelf .pl files
// have it. So a part at an odd quarter turn has its centre at (x + h/2, y + w/2), not (x + w/2, y + h/2). Upstream's
// CUDA op used the unturned box, which put a turned part's pins (w - h)/2 away from where the legalizer and the output
// file put them. The centre is constant for a given quarter turn, so the theta gradient doesn't change.

template <typename T>
int computePinPosLauncher(const T* x, const T* y, const T* pin_offset_x, const T* pin_offset_y, const T* theta,
                          const int64_t* pin2node_map, int num_pins, int num_theta, const T* h, const T* w,
                          const int num_threads, T* pin_x, T* pin_y) {
#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < num_pins; ++i) {
    int node_id = pin2node_map[i];
    T t = node_id < num_theta ? theta[node_id] : T(0);
    if (t == 0) {
      pin_x[i] = pin_offset_x[i] + x[node_id];
      pin_y[i] = pin_offset_y[i] + y[node_id];
    } else {
      T c = std::cos(t), s = std::sin(t);
      T ox = pin_offset_x[i] - w[node_id] / 2, oy = pin_offset_y[i] - h[node_id] / 2;
      bool odd = ((long)std::llround(t / T(M_PI / 2)) & 1) != 0;
      pin_x[i] = (x[node_id] + (odd ? h[node_id] : w[node_id]) / 2) + ox * c - oy * s;
      pin_y[i] = (y[node_id] + (odd ? w[node_id] : h[node_id]) / 2) + ox * s + oy * c;
    }
  }
  return 0;
}

// d pin_x / d theta = -(ofx - w/2) sin(theta) - (ofy - h/2) cos(theta)
// d pin_y / d theta =  (ofx - w/2) cos(theta) - (ofy - h/2) sin(theta)
// (The CUDA kernel had + for the second term of d pin_y / d theta until edaroute fixed it.)
template <typename T>
int computePinPosGradLauncher(const T* grad_out_x, const T* grad_out_y, const T* pin_offset_x,
                              const T* pin_offset_y, const T* theta, const int* flat_node2pin_map,
                              const int* flat_node2pin_start_map, int num_nodes, int num_theta, const T* h,
                              const T* w, const int num_threads, T* grad_x, T* grad_y, T* grad_theta) {
#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < num_nodes; ++i) {
    T gx = 0, gy = 0, gt = 0;
    T t = i < num_theta ? theta[i] : T(0);
    T c = std::cos(t), s = std::sin(t);
    for (int j = flat_node2pin_start_map[i]; j < flat_node2pin_start_map[i + 1]; ++j) {
      int pin_id = flat_node2pin_map[j];
      gx += grad_out_x[pin_id];
      gy += grad_out_y[pin_id];
      T ox = pin_offset_x[pin_id] - w[i] / 2, oy = pin_offset_y[pin_id] - h[i] / 2;
      gt += grad_out_x[pin_id] * (-ox * s - oy * c) + grad_out_y[pin_id] * (ox * c - oy * s);
    }
    grad_x[i] = gx;
    grad_y[i] = gy;
    if (i < num_theta) grad_theta[i] = gt;
  }
  return 0;
}

/// @brief Given cell locations and orientations, compute pin locations
/// @param pos cell locations in x and then y direction
/// @param theta each physical node's rotation, in radians
/// @param h, w each node's unturned height and width
/// @return pin positions in x and then y direction
at::Tensor pin_pos_forward(at::Tensor pos, at::Tensor pin_offset_x, at::Tensor pin_offset_y, at::Tensor theta,
                           at::Tensor pin2node_map, at::Tensor flat_node2pin_map,
                           at::Tensor flat_node2pin_start_map, at::Tensor h, at::Tensor w) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_CONTIGUOUS(theta);
  CHECK_CONTIGUOUS(h);
  CHECK_CONTIGUOUS(w);

  auto out = at::zeros(pin_offset_x.numel() * 2, pos.options());
  int num_nodes = pos.numel() / 2;
  int num_pins = pin_offset_x.numel();

  DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "computePinPosLauncher", [&] {
    computePinPosLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_nodes,
        DREAMPLACE_TENSOR_DATA_PTR(pin_offset_x, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(pin_offset_y, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(theta, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(pin2node_map, int64_t), num_pins,
        theta.numel(), DREAMPLACE_TENSOR_DATA_PTR(h, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(w, scalar_t),
        at::get_num_threads(), DREAMPLACE_TENSOR_DATA_PTR(out, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(out, scalar_t) + num_pins);
  });

  return out;
}

/// @return gradients for x, then y, then theta, each num_nodes long as the CUDA op returns them
at::Tensor pin_pos_backward(at::Tensor grad_out, at::Tensor pos, at::Tensor pin_offset_x, at::Tensor pin_offset_y,
                            at::Tensor theta, at::Tensor pin2node_map, at::Tensor flat_node2pin_map,
                            at::Tensor flat_node2pin_start_map, at::Tensor h, at::Tensor w,
                            int num_physical_nodes) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CPU(grad_out);
  CHECK_EVEN(grad_out);
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(theta);

  int num_nodes = pos.numel() / 2;
  auto out = at::zeros(num_nodes * 3, pos.options());
  int num_pins = pin_offset_x.numel();

  DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "computePinPosGradLauncher", [&] {
    computePinPosGradLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t) + num_pins,
        DREAMPLACE_TENSOR_DATA_PTR(pin_offset_x, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(pin_offset_y, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(theta, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(flat_node2pin_map, int),
        DREAMPLACE_TENSOR_DATA_PTR(flat_node2pin_start_map, int), num_physical_nodes, theta.numel(),
        DREAMPLACE_TENSOR_DATA_PTR(h, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(w, scalar_t), at::get_num_threads(),
        DREAMPLACE_TENSOR_DATA_PTR(out, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(out, scalar_t) + num_nodes,
        DREAMPLACE_TENSOR_DATA_PTR(out, scalar_t) + 2 * num_nodes);
  });

  return out;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::pin_pos_forward, "PinPos forward");
  m.def("backward", &DREAMPLACE_NAMESPACE::pin_pos_backward, "PinPos backward");
}

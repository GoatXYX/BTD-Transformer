#include <torch/extension.h>
#include "ein.h"

void torch_launch_ein_forward(torch::Tensor core_value,
                              torch::Tensor rw_head_q,
                              torch::Tensor w_head_k,
                              torch::Tensor w_head_v,
                              torch::Tensor output,
                              int DIM_I, int DIM_J, int DIM_K, int DIM_B, int DIM_H) {

    launch_ein_forward(core_value.data_ptr<float>(),
                       rw_head_q.data_ptr<float>(),
                       w_head_k.data_ptr<float>(),
                       w_head_v.data_ptr<float>(),
                       output.data_ptr<float>(),
                       DIM_I, DIM_J, DIM_K, DIM_B, DIM_H);
}

void torch_launch_ein_backward(torch::Tensor grad_output,
                               torch::Tensor core_value,
                               torch::Tensor rw_head_q,
                               torch::Tensor w_head_k,
                               torch::Tensor w_head_v,
                               torch::Tensor grad_core_value,
                               torch::Tensor grad_rw_head_q,
                               torch::Tensor grad_w_head_k,
                               torch::Tensor grad_w_head_v,
                               int DIM_I, int DIM_J, int DIM_K, int DIM_B, int DIM_H) {

    launch_ein_backward(grad_output.data_ptr<float>(),
                        core_value.data_ptr<float>(),
                        rw_head_q.data_ptr<float>(),
                        w_head_k.data_ptr<float>(),
                        w_head_v.data_ptr<float>(),
                        grad_core_value.data_ptr<float>(),
                        grad_rw_head_q.data_ptr<float>(),
                        grad_w_head_k.data_ptr<float>(),
                        grad_w_head_v.data_ptr<float>(),
                        DIM_I, DIM_J, DIM_K, DIM_B, DIM_H);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_ein_forward", &torch_launch_ein_forward, "Launch ein forward CUDA kernel");
    m.def("torch_launch_ein_backward", &torch_launch_ein_backward, "Launch ein backward CUDA kernel");
}

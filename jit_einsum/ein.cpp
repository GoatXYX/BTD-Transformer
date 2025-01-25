// ein.cpp
#include <torch/extension.h>
#include "ein.h"

void torch_launch_ein(torch::Tensor core_value,
                      torch::Tensor rw_head_q,
                      torch::Tensor w_head_k,
                      torch::Tensor w_head_v,
                      torch::Tensor output,
                      int DIM_I, int DIM_J, int DIM_K, int DIM_B, int DIM_H) {
    launch_ein(core_value.data_ptr<float>(),
               rw_head_q.data_ptr<float>(),
               w_head_k.data_ptr<float>(),
               w_head_v.data_ptr<float>(),
               output.data_ptr<float>(),
               DIM_I, DIM_J, DIM_K, DIM_B, DIM_H);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_ein", &torch_launch_ein, "Launch ein CUDA kernel");
}

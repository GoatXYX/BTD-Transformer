#ifndef EIN_H
#define EIN_H

// 前向传播的CUDA内核启动函数
void launch_ein_forward(
    float* core_value, float* rw_head_q, float* w_head_k, float* w_head_v, float* output,
    int DIM_I, int DIM_J, int DIM_K, int DIM_B, int DIM_H);

// 反向传播的CUDA内核启动函数
void launch_ein_backward(
    float* grad_output,
    float* core_value, float* rw_head_q, float* w_head_k, float* w_head_v,
    float* grad_core_value, float* grad_rw_head_q, float* grad_w_head_k, float* grad_w_head_v,
    int DIM_I, int DIM_J, int DIM_K, int DIM_B, int DIM_H);

#endif // EIN_H

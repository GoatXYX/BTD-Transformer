#ifndef EIN_H
#define EIN_H

// 前向传播的CUDA内核启动函数
void launch_ein_forward(
    double* core_value, double* rw_head_q, double* w_head_k, double* w_head_v, double* output,
    int DIM_I, int DIM_J, int DIM_K, int DIM_B, int DIM_H);

// 反向传播的CUDA内核启动函数
void launch_ein_backward(
    double* grad_output,
    double* core_value, double* rw_head_q, double* w_head_k, double* w_head_v,
    double* grad_core_value, double* grad_rw_head_q, double* grad_w_head_k, double* grad_w_head_v,
    int DIM_I, int DIM_J, int DIM_K, int DIM_B, int DIM_H);

#endif // EIN_H

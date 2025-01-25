#include "ein.h"

#define TM 4
#define TN 4

__global__ void ein_forward(
    double* core_value, double* rw_head_q, double* w_head_k, double* w_head_v, double* output,
    int DIM_I, int DIM_J, int DIM_K, int DIM_B, int DIM_H) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int i = bx * blockDim.x + tx;
    int j = by * blockDim.y + ty;
    int k = bz * blockDim.z + tz;

    int b = k / DIM_K;
    k %= DIM_K;

    if (i < DIM_I && j < DIM_J && k < DIM_K && b < DIM_B) {
        double sum = 0.0;
        int base_q = i * DIM_B * DIM_H;
        int base_k = j * DIM_B * DIM_H;
        int base_v = k * DIM_B * DIM_H;

        for (int h = 0; h < DIM_H; h++) {
            int idx_q = base_q + b * DIM_H + h;
            int idx_k = base_k + b * DIM_H + h;
            int idx_v = base_v + b * DIM_H + h;

            double q = rw_head_q[idx_q];
            double k_val = w_head_k[idx_k];
            double v = w_head_v[idx_v];
            double core = core_value[h];

            sum += core * q * k_val * v;
        }
        int idx_output = i * (DIM_B * DIM_J * DIM_K) + b * (DIM_J * DIM_K) + j * DIM_K + k;
        output[idx_output] = sum;
    }
}

__global__ void ein_backward(
    const double* grad_output,
    const double* core_value,
    const double* rw_head_q,
    const double* w_head_k,
    const double* w_head_v,
    double* grad_core_value,
    double* grad_rw_head_q,
    double* grad_w_head_k,
    double* grad_w_head_v,
    int DIM_I, int DIM_J, int DIM_K, int DIM_B, int DIM_H) {

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int i = bx * blockDim.x + tx;
    int j = by * blockDim.y + ty;
    int k = bz * blockDim.z + tz;

    int b = k / DIM_K;
    k %= DIM_K;

    if (i < DIM_I && j < DIM_J && k < DIM_K && b < DIM_B) {
        double go = grad_output[i * (DIM_B * DIM_J * DIM_K) + b * (DIM_J * DIM_K) + j * DIM_K + k];

        int base_q = i * DIM_B * DIM_H;
        int base_k = j * DIM_B * DIM_H;
        int base_v = k * DIM_B * DIM_H;

        for (int h = 0; h < DIM_H; h++) {
            int idx_q = base_q + b * DIM_H + h;
            int idx_k = base_k + b * DIM_H + h;
            int idx_v = base_v + b * DIM_H + h;

            double q = rw_head_q[idx_q];
            double k_val = w_head_k[idx_k];
            double v = w_head_v[idx_v];
            double core = core_value[h];

            atomicAdd(&grad_core_value[h], go * q * k_val * v);
            atomicAdd(&grad_rw_head_q[idx_q], go * core * k_val * v);
            atomicAdd(&grad_w_head_k[idx_k], go * core * q * v);
            atomicAdd(&grad_w_head_v[idx_v], go * core * q * k_val);
        }
    }
}


void launch_ein_forward(double* core_value, double* rw_head_q, double* w_head_k, double* w_head_v, double* output,
                        int DIM_I, int DIM_J, int DIM_K, int DIM_B, int DIM_H) {
    dim3 grid((DIM_I + TM - 1) / TM, (DIM_J + TN - 1) / TN, (DIM_K + TN - 1) / TN * DIM_B);
    dim3 block(TM, TN, TN);
    ein_forward<<<grid, block>>>(core_value, rw_head_q, w_head_k, w_head_v, output, DIM_I, DIM_J, DIM_K, DIM_B, DIM_H);
}

void launch_ein_backward(
    double* grad_output,
    double* core_value,
    double* rw_head_q,
    double* w_head_k,
    double* w_head_v,
    double* grad_core_value,
    double* grad_rw_head_q,
    double* grad_w_head_k,
    double* grad_w_head_v,
    int DIM_I, int DIM_J, int DIM_K, int DIM_B, int DIM_H) {
    dim3 grid((DIM_I + TM - 1) / TM, (DIM_J + TN - 1) / TN, (DIM_K + TN - 1) / TN * DIM_B);
    dim3 block(TM, TN, TN);
    ein_backward<<<grid, block>>>(grad_output, core_value, rw_head_q, w_head_k, w_head_v, grad_core_value, grad_rw_head_q, grad_w_head_k, grad_w_head_v, DIM_I, DIM_J, DIM_K, DIM_B, DIM_H);
}

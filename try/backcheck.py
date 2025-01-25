import torch
from torch.autograd import gradcheck, Function
from torch.utils.cpp_extension import load

# 加载自定义 CUDA 模块
ein_module = load(
    name="ein_module",
    sources=["ein.cpp", "ein.cu"],
    verbose=True
)

class CustomEinsumFunction(Function):
    @staticmethod
    def forward(ctx, core_value, rw_head_q, w_head_k, w_head_v):
        DIM_I, DIM_J, DIM_K, DIM_B, DIM_H = 10, 10, 10, 10, 10  # 减小测试尺寸
        ctx.save_for_backward(core_value, rw_head_q, w_head_k, w_head_v)
        output = torch.empty(DIM_I, DIM_B, DIM_J, DIM_K, device='cuda', dtype=torch.float64)
        ein_module.torch_launch_ein_forward(core_value, rw_head_q, w_head_k, w_head_v, output, DIM_I, DIM_J, DIM_K, DIM_B, DIM_H)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        DIM_I, DIM_J, DIM_K, DIM_B, DIM_H = 10, 10, 10, 10, 10  # 减小测试尺寸
        core_value, rw_head_q, w_head_k, w_head_v = ctx.saved_tensors
        grad_core_value = torch.zeros_like(core_value)
        grad_rw_head_q = torch.zeros_like(rw_head_q)
        grad_w_head_k = torch.zeros_like(w_head_k)
        grad_w_head_v = torch.zeros_like(w_head_v)
        ein_module.torch_launch_ein_backward(grad_output, core_value, rw_head_q, w_head_k, w_head_v,
                                             grad_core_value, grad_rw_head_q, grad_w_head_k, grad_w_head_v,
                                             DIM_I, DIM_J, DIM_K, DIM_B, DIM_H)
        return grad_core_value, grad_rw_head_q, grad_w_head_k, grad_w_head_v

# 输入数据，使用更小的尺寸
core_value = torch.randn(10, device='cuda', dtype=torch.float64, requires_grad=True)
rw_head_q = torch.randn(10, 10, 10, device='cuda', dtype=torch.float64, requires_grad=True)
w_head_k = torch.randn(10, 10, 10, device='cuda', dtype=torch.float64, requires_grad=True)
w_head_v = torch.randn(10, 10, 10, device='cuda', dtype=torch.float64, requires_grad=True)

# 运行梯度检查
test = gradcheck(CustomEinsumFunction.apply, (core_value, rw_head_q, w_head_k, w_head_v), eps=1e-6, atol=1e-4, rtol=1e-3)
print("Gradcheck passed:", test)

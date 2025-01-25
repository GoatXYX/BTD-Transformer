import torch
from torch.autograd import gradcheck, Function
from torch.utils.cpp_extension import load

DIM_I, DIM_J, DIM_K, DIM_B, DIM_H = 32, 32, 32, 120, 40
# 加载自定义 CUDA 模块
ein_module = load(
    name="ein_module",
    sources=["ein.cpp", "ein.cu"],
    verbose=True
)

class CustomEinsumFunction(Function):
    @staticmethod
    def forward(ctx, core_value, rw_head_q, w_head_k, w_head_v):
        DIM_I, DIM_J, DIM_K, DIM_B, DIM_H = 32, 32, 32, 120, 40  # 更新为提供的尺寸
        ctx.save_for_backward(core_value, rw_head_q, w_head_k, w_head_v)
        output = torch.empty(DIM_I, DIM_B, DIM_J, DIM_K, device='cuda', dtype=torch.float32)
        ein_module.torch_launch_ein_forward(core_value, rw_head_q, w_head_k, w_head_v, output, DIM_I, DIM_J, DIM_K, DIM_B, DIM_H)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        core_value, rw_head_q, w_head_k, w_head_v = ctx.saved_tensors
        DIM_I, DIM_J, DIM_K, DIM_B, DIM_H = 32, 32, 32, 120, 40  # 保持与前向传播相同的尺寸
        grad_core_value = torch.zeros_like(core_value)
        grad_rw_head_q = torch.zeros_like(rw_head_q)
        grad_w_head_k = torch.zeros_like(w_head_k)
        grad_w_head_v = torch.zeros_like(w_head_v)
        ein_module.torch_launch_ein_backward(grad_output, core_value, rw_head_q, w_head_k, w_head_v,
                                             grad_core_value, grad_rw_head_q, grad_w_head_k, grad_w_head_v,
                                             DIM_I, DIM_J, DIM_K, DIM_B, DIM_H)
        return grad_core_value, grad_rw_head_q, grad_w_head_k, grad_w_head_v

# 初始化输入数据，使用更小的尺寸进行测试
core_value = torch.randn(DIM_H, device='cuda', dtype=torch.float32, requires_grad=True)
rw_head_q = torch.randn(DIM_I, DIM_B, DIM_H, device='cuda', dtype=torch.float32, requires_grad=True)
w_head_k = torch.randn(DIM_J, DIM_B, DIM_H, device='cuda', dtype=torch.float32, requires_grad=True)
w_head_v = torch.randn(DIM_K, DIM_B, DIM_H, device='cuda', dtype=torch.float32, requires_grad=True)

# 运行梯度检查
test = gradcheck(CustomEinsumFunction.apply, (core_value, rw_head_q, w_head_k, w_head_v), eps=1e-4, atol=1e-3, rtol=1e-2)
print("Gradcheck passed:", test)

# 运行前向和反向传播，查看core_value的梯度
output = CustomEinsumFunction.apply(core_value, rw_head_q, w_head_k, w_head_v)
output.sum().backward()
print("Gradient of core_value:", core_value.grad)

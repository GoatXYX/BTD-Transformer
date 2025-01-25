import torch
from torch.autograd import Function
from torch import nn
from torch.utils.cpp_extension import load

# 尝试加载自定义的 CUDA 扩展模块
try:
    ein_module = load(
        name="ein_module",
        sources=["ein.cpp", "ein.cu"],
        verbose=True
    )
except Exception as e:
    print(f"Failed to load CUDA extension: {e}")
    exit()

class EinFunction(Function):
    @staticmethod
    def forward(ctx, core_value, rw_head_q, w_head_k, w_head_v):
        DIM_I, DIM_J, DIM_K, DIM_B, DIM_H = core_value.size(0), rw_head_q.size(1), w_head_k.size(1), w_head_v.size(1), core_value.size(0)
        outputs = torch.empty(DIM_I, DIM_B, DIM_J, DIM_K, device='cuda', dtype=core_value.dtype)
        ein_module.torch_launch_ein_forward(core_value, rw_head_q, w_head_k, w_head_v, outputs, DIM_I, DIM_J, DIM_K, DIM_B, DIM_H)
        ctx.save_for_backward(core_value, rw_head_q, w_head_k, w_head_v)
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        core_value, rw_head_q, w_head_k, w_head_v = ctx.saved_tensors
        grad_core_value = torch.empty_like(core_value)
        grad_rw_head_q = torch.empty_like(rw_head_q)
        grad_w_head_k = torch.empty_like(w_head_k)
        grad_w_head_v = torch.empty_like(w_head_v)
        DIM_I, DIM_J, DIM_K, DIM_B, DIM_H = core_value.size(0), rw_head_q.size(1), w_head_k.size(1), w_head_v.size(1), core_value.size(0)
        ein_module.torch_launch_ein_backward(grad_output, core_value, rw_head_q, w_head_k, w_head_v, grad_core_value, grad_rw_head_q, grad_w_head_k, grad_w_head_v, DIM_I, DIM_J, DIM_K, DIM_B, DIM_H)
        return grad_core_value, grad_rw_head_q, grad_w_head_k, grad_w_head_v

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        # 使用dtype=torch.double确保所有参数都是Double类型
        self.core_value = nn.Parameter(torch.randn(10, device='cuda', dtype=torch.double))
        self.rw_head_q = nn.Parameter(torch.randn(32, 120, 10, device='cuda', dtype=torch.double))
        self.w_head_k = nn.Parameter(torch.randn(32, 120, 10, device='cuda', dtype=torch.double))
        self.w_head_v = nn.Parameter(torch.randn(32, 120, 10, device='cuda', dtype=torch.double))


    def forward(self):
        return EinFunction.apply(self.core_value, self.rw_head_q, self.w_head_k, self.w_head_v)

def main():
    model = TestModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    input_tensor = torch.randn(1, 32, device='cuda')  # Dummy input for dimension match
    output = model()
    output.sum().backward()  # Backward on sum for simplicity

    # Updating parameters (optional step to see parameter changes)
    optimizer.step()

    print("Backward pass successfully completed. Parameters updated (if optimizer.step() is called).")

if __name__ == "__main__":
    main()

import torch
from torch.utils.cpp_extension import load
import time
import matplotlib.pyplot as plt

# 加载 JIT 编译的模块
ein_module = load(
    name="ein_module",
    sources=["ein.cpp", "ein.cu"],
    verbose=True
)

# 初始化输入数据
DIM_I, DIM_J, DIM_K, DIM_B, DIM_H = 32, 32, 32, 120, 40
core_value = torch.randn(DIM_H, device='cuda')
rw_head_q = torch.randn(DIM_I, DIM_B, DIM_H, device='cuda')
w_head_k = torch.randn(DIM_J, DIM_B, DIM_H, device='cuda')
w_head_v = torch.randn(DIM_K, DIM_B, DIM_H, device='cuda')
output = torch.empty(DIM_I, DIM_B, DIM_J, DIM_K, device='cuda')
qlen = 32
bsz = 120

# 准备记录时间的列表，单位改为毫秒
cuda_times = []
einsum_times = []

# 进行10次测试
for _ in range(102):
    start_time = time.time()
    ein_module.torch_launch_ein(core_value, rw_head_q, w_head_k, w_head_v, output, DIM_I, DIM_J, DIM_K, DIM_B, DIM_H)
    output_cuda = output.contiguous().view(qlen, bsz, -1)
    cuda_times.append((time.time() - start_time) * 1000000)  # 将秒转换为毫秒


for _ in range(102):
    start_time = time.time()
    output_einsum = torch.einsum('h,ibh,jbh,kbh->ibjk', [core_value, rw_head_q, w_head_k, w_head_v]).contiguous().view(qlen, bsz, -1)
    einsum_times.append((time.time() - start_time) * 1000000)  # 将秒转换为毫秒

print('CUDA Extension Time (ms):', cuda_times)
print('PyTorch Einsum Time (ms):', einsum_times)

# 绘图
plt.figure(figsize=(10, 5))
plt.plot(cuda_times[2:], label='CUDA Extension')
plt.plot(einsum_times[2:], label='PyTorch Einsum')
plt.xlabel('Run')
plt.ylabel('Time (microsecond)')
plt.title('Comparison of Execution Time in Microsecond')
plt.legend()

# 保存图像到当前目录
plt.savefig('execution_time_comparison_ms.png')

# 显示图表
plt.show()

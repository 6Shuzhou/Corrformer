import torch
import torch.nn as nn

class Adapter(nn.Module):
    """
    Adapter模块，一个瓶颈结构的轻量级神经网络。
    它接收一个张量，通过一个下采样全连接层、一个非线性激活函数、
    一个上采样全连接层，最后将输出与原始输入通过残差连接相加。
    """
    def __init__(self, d_model, bottleneck_dim):
        super(Adapter, self).__init__()
        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        self.non_linear = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck_dim, d_model)

        # 初始化上采样层的权重为0，使得在训练初期，Adapter模块是一个恒等变换
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.non_linear(x)
        x = self.up_proj(x)
        return x + residual
import torch
import torch.nn as nn

class FusionLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 初始化时不创建固定维度的注意力层

    def forward(self, x_base, x_ext):
        """
        x_base: [B, T, D]
        x_ext:  [B, D]
        """

        x_ext = x_ext.unsqueeze(1).repeat(1, x_base.size(1), 1)

        # 确保维度匹配
        if x_base.shape[-1] != x_ext.shape[-1]:
            # 添加投影层
            proj = nn.Linear(x_ext.shape[-1], x_base.shape[-1]).to(x_ext.device)
            x_ext = proj(x_ext)

        # 动态创建注意力层，确保维度匹配
        dim = x_ext.shape[-1]
        attn = nn.Linear(dim, dim).to(x_ext.device)
        alpha = torch.softmax(attn(x_ext), dim=-1)

        x_fused = x_base + alpha * x_ext

        return x_fused, alpha
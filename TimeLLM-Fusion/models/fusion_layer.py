import torch
import torch.nn as nn

class FusionLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(2, dim)  # 两个权重 → 映射到特征空间

    def forward(self, x_base, llm_weights):
        # llm_weights: [B, 2]

        weight_feature = self.proj(llm_weights)  # [B, D]

        weight_feature = weight_feature.unsqueeze(1).repeat(1, x_base.size(1), 1)

        x_fused = x_base + weight_feature

        return x_fused, llm_weights
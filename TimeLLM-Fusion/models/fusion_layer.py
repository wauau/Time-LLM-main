import torch
import torch.nn as nn

class FusionLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, dim)

    def forward(self, x_base, x_ext):
        # x_base: [B, T, D]
        # x_ext:  [B, D]

        x_ext = x_ext.unsqueeze(1).repeat(1, x_base.size(1), 1)

        alpha = torch.softmax(self.attn(x_ext), dim=-1)

        x_fused = x_base + alpha * x_ext

        return x_fused, alpha
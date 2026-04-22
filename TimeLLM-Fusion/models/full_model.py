import torch.nn as nn
from models.time_llm_modified import TimeLLM
from models.external_llm_encoder import ExternalLLMEncoder
from models.fusion_layer import FusionLayer

class FullModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.time_model = TimeLLM(config)
        self.external_model = ExternalLLMEncoder(out_dim=config.d_model)
        self.fusion = FusionLayer(config.d_model)

    def forward(self, x, external):
        # 时间序列
        x_base = self.time_model.encode(x)

        # 外因
        x_ext = self.external_model(external)

        # 融合
        x_fused, alpha = self.fusion(x_base, x_ext)

        # 预测
        out = self.time_model.forward_from_embedding(x_fused)

        return out, alpha
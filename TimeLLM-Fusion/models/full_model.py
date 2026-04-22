import torch.nn as nn
from models.time_llm_modified import TimeLLM
from models.external_llm_encoder import ExternalLLMReasoner
from models.fusion_layer import FusionLayer

class FullModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.time_model = TimeLLM(config)
        self.external_reasoner = ExternalLLMReasoner()
        self.fusion = FusionLayer(config.d_model)

    def forward(self, x, external):
        # 时间序列编码
        x_base = self.time_model.encode(x)

        # LLM推理（关键）
        llm_weights = self.external_reasoner(external)

        # 用LLM权重融合
        x_fused, weights = self.fusion(x_base, llm_weights)

        # 预测
        out = self.time_model.forward_from_embedding(x_fused)

        return out, weights
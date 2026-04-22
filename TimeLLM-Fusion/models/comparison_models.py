import torch
import torch.nn as nn
from models.time_llm_modified import TimeLLM
from models.external_llm_encoder import ExternalLLMReasoner


class BaselineModel(nn.Module):
    """无外因模型"""
    def __init__(self, config):
        super().__init__()
        self.time_model = TimeLLM(config)

    def forward(self, x, external):
        # 不使用外因
        x_base = self.time_model.encode(x)
        out = self.time_model.forward_from_embedding(x_base)
        return out, None


class ConcatModel(nn.Module):
    """简单拼接模型"""
    def __init__(self, config):
        super().__init__()
        self.time_model = TimeLLM(config)
        self.external_reasoner = ExternalLLMReasoner()
        # 拼接后的维度
        self.concat_proj = nn.Linear(config.d_model + 2, config.d_model)

    def forward(self, x, external):
        x_base = self.time_model.encode(x)
        # 获取外因权重
        llm_weights = self.external_reasoner(external)
        # 扩展权重维度
        llm_weights = llm_weights.unsqueeze(1).repeat(1, x_base.size(1), 1)
        # 拼接
        x_concat = torch.cat([x_base, llm_weights], dim=-1)
        x_concat = self.concat_proj(x_concat)
        out = self.time_model.forward_from_embedding(x_concat)
        return out, llm_weights


class LLMOnlyModel(nn.Module):
    """只有LLM外因模型"""
    def __init__(self, config):
        super().__init__()
        self.time_model = TimeLLM(config)
        self.external_reasoner = ExternalLLMReasoner()
        # 映射外因权重到特征空间
        self.llm_proj = nn.Linear(2, config.d_model)

    def forward(self, x, external):
        x_base = self.time_model.encode(x)
        # 获取外因权重
        llm_weights = self.external_reasoner(external)
        # 映射到特征空间
        weight_feature = self.llm_proj(llm_weights)
        weight_feature = weight_feature.unsqueeze(1).repeat(1, x_base.size(1), 1)
        # 直接使用权重特征
        out = self.time_model.forward_from_embedding(weight_feature)
        return out, llm_weights
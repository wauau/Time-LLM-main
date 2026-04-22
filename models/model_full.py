import torch
import torch.nn as nn

from models.TimeLLM import Model as TimeLLM
from models.external_llm_encoder import ExternalLLMEncoder
from models.fusion_layer import FusionLayer


class FullModel(nn.Module):
    def __init__(self, config, llm_model, tokenizer):
        super().__init__()

        # 时间序列模型
        self.time_llm = TimeLLM(config)

        # 外因LLM
        self.external_encoder = ExternalLLMEncoder(
            llm_model, tokenizer, output_dim=config.d_model
        )

        # 融合层
        self.fusion = FusionLayer(config.d_model)

    def forward(self, x, external_data):

        # Step1：时间序列编码（走原TimeLLM前半部分）
        x_base, n_vars = self.time_llm.embed(x)   # ⚠️ 你需要在TimeLLM里拆出embed函数

        # Step2：外因 → LLM
        x_ext = self.external_encoder(external_data)

        # Step3：融合
        x_fused, alpha = self.fusion(x_base, x_ext)

        # Step4：送回TimeLLM后半部分
        # 生成prompt embeddings
        B, T, N = x.size()
        x_enc_reshaped = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        
        min_values = torch.min(x_enc_reshaped, dim=1)[0]
        max_values = torch.max(x_enc_reshaped, dim=1)[0]
        medians = torch.median(x_enc_reshaped, dim=1).values
        lags = self.time_llm.calcute_lags(x_enc_reshaped)
        trends = x_enc_reshaped.diff(dim=1).sum(dim=1)
        
        prompt = []
        for b in range(x_enc_reshaped.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.time_llm.description}"
                f"Task description: forecast the next {str(self.time_llm.pred_len)} steps given the previous {str(self.time_llm.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )
            prompt.append(prompt_)
        
        prompt = self.time_llm.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.time_llm.llm_model.get_input_embeddings()(prompt.to(x.device))
        
        output = self.time_llm.forward_from_embedding(x_fused, n_vars, prompt_embeddings)

        return output, alpha
import torch
import torch.nn as nn

class ExternalLLMEncoder(nn.Module):
    def __init__(self, llm_model, tokenizer, output_dim=16):
        super().__init__()
        self.llm = llm_model
        self.tokenizer = tokenizer
        self.proj = nn.Linear(768, output_dim)  # 根据LLM hidden size改

    def build_prompt(self, date, weather, holiday):
        return f"""
        日期: {date}
        天气: {weather}
        节假日: {holiday}
        请判断对人流影响强度（0-1）
        """

    def forward(self, batch_external):
        prompts = []
        for item in batch_external:
            p = self.build_prompt(
                item["date"],
                item["weather"],
                item["holiday"]
            )
            prompts.append(p)

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        outputs = self.llm(**inputs)

        # 取CLS或最后token
        hidden = outputs.last_hidden_state[:, -1, :]

        external_feature = self.proj(hidden)
        return external_feature
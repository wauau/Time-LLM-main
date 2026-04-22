import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizerFast

class ExternalLLMEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        # 使用本地配置，不依赖下载
        self.config = BertConfig(
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=2,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            output_attentions=True,
            output_hidden_states=True
        )
        
        # 创建BERT模型实例
        self.llm = BertModel(self.config)
        
        # 创建本地tokenizer
        self.tokenizer = BertTokenizerFast(
            vocab_file=None,
            do_lower_case=True,
            strip_accents=True,
            tokenize_chinese_chars=False,
            wordpiece_prefix="##"
        )
        # 添加特殊 tokens
        special_tokens = {'pad_token': '[PAD]', 'cls_token': '[CLS]', 'sep_token': '[SEP]'}
        self.tokenizer.add_special_tokens(special_tokens)
        
        self.proj = nn.Linear(768, out_dim)

    def build_prompt(self, item):
        return f"""
        Date: {item['date']}
        Weather: {item['weather']}
        Holiday: {item['holiday']}
        Predict impact on crowd flow (0-1)
        """

    def forward(self, batch_external):
        prompts = [self.build_prompt(x) for x in batch_external]

        inputs = self.tokenizer(prompts, return_tensors="pt",
                                 padding=True, truncation=True)

        outputs = self.llm(**inputs)
        hidden = outputs.last_hidden_state[:, -1, :]

        return self.proj(hidden)
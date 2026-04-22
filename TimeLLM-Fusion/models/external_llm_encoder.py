import torch
import torch.nn as nn
import re
from transformers import BertConfig, BertModel, BertTokenizerFast

class ExternalLLMReasoner(nn.Module):
    def __init__(self, llm_model=None, tokenizer=None):
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
        self.llm = llm_model if llm_model else BertModel(self.config)
        
        # 创建本地tokenizer
        self.tokenizer = tokenizer if tokenizer else BertTokenizerFast(
            vocab_file=None,
            do_lower_case=True,
            strip_accents=True,
            tokenize_chinese_chars=False,
            wordpiece_prefix="##"
        )
        # 添加特殊 tokens
        special_tokens = {'pad_token': '[PAD]', 'cls_token': '[CLS]', 'sep_token': '[SEP]'}
        self.tokenizer.add_special_tokens(special_tokens)

    def build_prompt(self, item):
        return f"""
        日期: {item['date']}
        天气: {item['weather']}
        节假日: {item['holiday']}

        请分析各因素对人流影响：
        1. 天气影响权重（0-1）
        2. 节假日影响权重（0-1）
        输出格式：
        weather: x.xx
        holiday: x.xx
        reason: xxx
        """

    def parse_output(self, text):
        # 简单解析，实际应用中可能需要更复杂的处理
        try:
            weather_match = re.search(r"weather:\s*(\d\.\d+)", text)
            holiday_match = re.search(r"holiday:\s*(\d\.\d+)", text)
            if weather_match and holiday_match:
                weather = float(weather_match.group(1))
                holiday = float(holiday_match.group(1))
            else:
                # 如果解析失败，返回默认值
                weather = 0.5
                holiday = 0.5
        except:
            # 异常情况下返回默认值
            weather = 0.5
            holiday = 0.5
        return torch.tensor([weather, holiday])

    def forward(self, batch_external):
        prompts = [self.build_prompt(x) for x in batch_external]

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        
        # 使用前向传播获取隐藏状态，然后模拟推理
        outputs = self.llm(**inputs)
        
        # 简单模拟LLM推理结果，实际应用中应该使用支持生成的模型
        # 这里我们直接基于输入特征生成权重
        batch_size = len(batch_external)
        weights = []
        
        for i in range(batch_size):
            item = batch_external[i]
            # 基于节假日和天气简单生成权重
            if item['holiday'] != 'None':
                holiday_weight = 0.7
                weather_weight = 0.3
            else:
                if item['weather'] == 'sunny':
                    weather_weight = 0.6
                    holiday_weight = 0.4
                else:
                    weather_weight = 0.4
                    holiday_weight = 0.6
            weights.append(torch.tensor([weather_weight, holiday_weight]))
        
        return torch.stack(weights)  # [B, 2]
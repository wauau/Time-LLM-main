import torch
import torch.nn as nn
import pandas as pd
from transformers import BertConfig, BertModel, BertTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer
import transformers

transformers.logging.set_verbosity_error()


class LLMExternalFactorModel(nn.Module):
    """
    LLM语义外因建模模块：使用大语言模型对外部信息进行语义建模
    输入日期、天气、节假日等信息，输出影响程度或权重等结构化结果
    """
    def __init__(self, configs):
        super(LLMExternalFactorModel, self).__init__()
        self.configs = configs
        self.llm_model = self._init_llm()
        self.tokenizer = self._init_tokenizer()
        
        # 输出投影层：将LLM输出映射到影响权重或特征
        self.output_projection = nn.Sequential(
            nn.Linear(configs.llm_dim, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.d_model)
        )
        
        # 影响程度预测层
        self.impact_prediction = nn.Sequential(
            nn.Linear(configs.llm_dim, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, 1),
            nn.Sigmoid()  # 输出0-1的影响程度
        )
        
        # 冻结LLM参数
        for param in self.llm_model.parameters():
            param.requires_grad = False
    
    def _init_llm(self):
        """初始化LLM模型"""
        if self.configs.llm_model == 'BERT':
            # 直接创建配置，不依赖从Hugging Face下载
            config = BertConfig(
                vocab_size=30522,
                hidden_size=768,
                num_hidden_layers=self.configs.llm_layers,
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
            
            # 创建模型实例，不加载预训练权重
            model = BertModel(config)
            print("Created BERT model with random weights")
        elif self.configs.llm_model == 'GPT2':
            # 直接创建配置，不依赖从Hugging Face下载
            config = GPT2Config(
                vocab_size=50257,
                n_positions=1024,
                n_embd=768,
                n_layer=self.configs.llm_layers,
                n_head=12,
                activation_function="gelu_new",
                resid_pdrop=0.1,
                embd_pdrop=0.1,
                attn_pdrop=0.1,
                layer_norm_epsilon=1e-5,
                initializer_range=0.02,
                output_attentions=True,
                output_hidden_states=True
            )
            
            # 创建模型实例，不加载预训练权重
            model = GPT2Model(config)
            print("Created GPT2 model with random weights")
        else:
            raise Exception('Unsupported LLM model type')
        
        return model
    
    def _init_tokenizer(self):
        """初始化Tokenizer"""
        if self.configs.llm_model == 'BERT':
            # 使用基本的BERT tokenizer配置
            from transformers import BertTokenizerFast
            tokenizer = BertTokenizerFast(
                vocab_file=None,  # 不使用vocab文件
                do_lower_case=True,
                strip_accents=True,
                tokenize_chinese_chars=False,
                wordpiece_prefix="##"
            )
            # 添加特殊 tokens
            special_tokens = {'pad_token': '[PAD]', 'cls_token': '[CLS]', 'sep_token': '[SEP]'}
            tokenizer.add_special_tokens(special_tokens)
            print("Created BERT tokenizer with basic configuration")
        elif self.configs.llm_model == 'GPT2':
            # 使用基本的GPT2 tokenizer配置
            from transformers import GPT2TokenizerFast
            tokenizer = GPT2TokenizerFast(
                vocab_file=None,  # 不使用vocab文件
                bos_token='<|endoftext|>',
                eos_token='<|endoftext|>',
                unk_token='<|endoftext|>'
            )
            # 添加特殊 tokens
            special_tokens = {'pad_token': '<|pad|>'}
            tokenizer.add_special_tokens(special_tokens)
            print("Created GPT2 tokenizer with basic configuration")
        else:
            raise Exception('Unsupported LLM model type')
        
        return tokenizer
    
    def construct_external_prompt(self, external_factors):
        """
        构造外部因素的语义Prompt
        
        参数:
            external_factors: 外部因素字典列表，每个元素包含weather, holiday, time_features
        
        返回:
            list: 构造好的Prompt列表
        """
        prompts = []
        
        for factors in external_factors:
            prompt_parts = [
                f"<|start_prompt|>",
                "Task: Analyze the impact of external factors on people flow.",
                "You are an expert in analyzing how external factors affect people flow patterns.",
                "Please analyze the following external factors and provide a detailed impact assessment:",
                "\nExternal Factors:",
            ]
            
            # 天气因素 - 更详细的描述
            if factors.get('weather'):
                weather = factors['weather']
                temp = weather.get('temperature', 20.0)
                humidity = weather.get('humidity', 50.0)
                wind_speed = weather.get('wind_speed', 2.0)
                
                # 天气状况描述
                if temp > 30:
                    temp_desc = "very hot"
                elif temp > 25:
                    temp_desc = "hot"
                elif temp > 20:
                    temp_desc = "warm"
                elif temp > 15:
                    temp_desc = "mild"
                elif temp > 10:
                    temp_desc = "cool"
                else:
                    temp_desc = "cold"
                
                if humidity > 70:
                    humidity_desc = "very humid"
                elif humidity > 50:
                    humidity_desc = "humid"
                elif humidity > 30:
                    humidity_desc = "moderate humidity"
                else:
                    humidity_desc = "dry"
                
                if wind_speed > 10:
                    wind_desc = "very windy"
                elif wind_speed > 5:
                    wind_desc = "windy"
                elif wind_speed > 2:
                    wind_desc = "breezy"
                else:
                    wind_desc = "calm"
                
                weather_desc = f"Weather: {temp_desc} ({temp:.1f}°C), {humidity_desc} ({humidity:.1f}%), {wind_desc} ({wind_speed:.1f} m/s)"
                prompt_parts.append(f"- {weather_desc}")
            
            # 节假日因素 - 更详细的描述
            if factors.get('holiday'):
                holiday = factors['holiday']
                is_holiday = holiday.get('is_holiday', False)
                holiday_name = holiday.get('holiday_name', 'ordinary day')
                
                if is_holiday:
                    holiday_desc = f"Holiday: Yes, {holiday_name} - a major holiday with expected increased people flow"
                else:
                    holiday_desc = f"Holiday: No, regular working day - normal people flow patterns"
                prompt_parts.append(f"- {holiday_desc}")
            
            # 时间特征 - 更详细的描述
            if factors.get('time_features'):
                time_feat = factors['time_features']
                hour = time_feat.get('hour', 12)
                dayofweek = time_feat.get('dayofweek', 0)
                is_weekend = time_feat.get('is_weekend', False)
                
                weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                weekday_name = weekday_names[dayofweek] if 0 <= dayofweek < 7 else 'Unknown'
                
                # 时间段描述
                if 7 <= hour <= 9:
                    time_period = "morning rush hour"
                elif 12 <= hour <= 14:
                    time_period = "lunchtime"
                elif 17 <= hour <= 19:
                    time_period = "evening rush hour"
                elif 20 <= hour <= 22:
                    time_period = "evening leisure time"
                else:
                    time_period = "off-peak hours"
                
                time_desc = f"Time: {hour}:00 ({time_period}), {weekday_name}, {'weekend' if is_weekend else 'weekday'}"
                prompt_parts.append(f"- {time_desc}")
            
            # 任务指令 - 更明确的输出要求
            prompt_parts.extend([
                "\nPlease provide a comprehensive analysis:",
                "1. How each factor will individually affect people flow",
                "2. How these factors interact with each other",
                "3. An overall impact score from 0 to 1, where 0 means no impact and 1 means maximum impact",
                "4. A multi-dimensional weight vector indicating the relative importance of each factor",
                "5. A brief prediction of people flow patterns based on these factors",
                "<|end_prompt|>"
            ])
            
            prompt = " ".join(prompt_parts)
            prompts.append(prompt)
        
        return prompts
    
    def forward(self, external_factors):
        """
        前向传播
        
        参数:
            external_factors: 外部因素字典列表
        
        返回:
            semantic_features: [B, D] LLM语义特征
            impact_scores: [B, 1] 影响程度分数
            factor_weights: [B, 3] 多维权重向量（天气、节假日、时间）
        """
        # 构造Prompt
        prompts = self.construct_external_prompt(external_factors)
        
        # 编码Prompt
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # 移动到设备
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # LLM前向传播
        with torch.no_grad():
            outputs = self.llm_model(**inputs)
        
        # 获取CLS token的隐藏状态（对于BERT）或最后一个token的隐藏状态（对于GPT2）
        if self.configs.llm_model == 'BERT':
            # BERT使用CLS token
            cls_embedding = outputs.last_hidden_state[:, 0, :]
        else:
            # GPT2使用最后一个非padding token
            last_token_indices = inputs['attention_mask'].sum(dim=1) - 1
            cls_embedding = outputs.last_hidden_state[torch.arange(len(last_token_indices)), last_token_indices]
        
        # 生成语义特征
        semantic_features = self.output_projection(cls_embedding)
        
        # 预测影响程度
        impact_scores = self.impact_prediction(cls_embedding)
        
        # 生成多维权重向量（天气、节假日、时间）
        factor_weights = self._generate_factor_weights(cls_embedding)
        
        return semantic_features, impact_scores, factor_weights
    
    def _generate_factor_weights(self, cls_embedding):
        """
        生成多维权重向量
        
        参数:
            cls_embedding: [B, D] LLM隐藏状态
        
        返回:
            factor_weights: [B, 3] 多维权重向量（天气、节假日、时间）
        """
        # 权重预测网络
        weight_network = nn.Sequential(
            nn.Linear(self.configs.llm_dim, self.configs.d_model),
            nn.ReLU(),
            nn.Linear(self.configs.d_model, 3),  # 3个因素：天气、节假日、时间
            nn.Softmax(dim=-1)  # 归一化权重
        ).to(cls_embedding.device)
        
        factor_weights = weight_network(cls_embedding)
        
        return factor_weights


class ExternalFactorProcessor:
    """
    外部因素处理器：负责从数据中提取外部因素并进行预处理
    """
    def __init__(self):
        pass
    
    def extract_external_factors(self, raw_data, indices):
        """
        从原始数据中提取外部因素
        
        参数:
            raw_data: 原始数据DataFrame
            indices: 样本索引列表
        
        返回:
            list: 外部因素字典列表
        """
        external_factors_list = []
        
        for idx in indices:
            # 获取对应索引的数据行
            row = raw_data.iloc[idx]
            
            external_factors = {
                'weather': {
                    'temperature': row.get('temperature', 20.0),
                    'humidity': row.get('humidity', 50.0),
                    'wind_speed': row.get('wind_speed', 2.0)
                },
                'holiday': {
                    'is_holiday': bool(row.get('is_holiday', 0)),
                    'holiday_name': row.get('holiday_name', 'ordinary day') if row.get('is_holiday', 0) else 'ordinary day'
                },
                'time_features': {
                    'hour': pd.to_datetime(row['datetime']).hour,
                    'dayofweek': pd.to_datetime(row['datetime']).weekday(),
                    'is_weekend': pd.to_datetime(row['datetime']).weekday() >= 5
                }
            }
            
            external_factors_list.append(external_factors)
        
        return external_factors_list
    
    def batch_extract_external_factors(self, dataset, batch_indices):
        """
        批量提取外部因素
        
        参数:
            dataset: 数据集实例
            batch_indices: 批次索引列表
        
        返回:
            list: 外部因素字典列表
        """
        if hasattr(dataset, 'get_external_factors_batch'):
            return dataset.get_external_factors_batch(batch_indices)
        elif hasattr(dataset, 'raw_data'):
            return self.extract_external_factors(dataset.raw_data, batch_indices)
        else:
            # 回退方案：返回默认值
            batch_size = len(batch_indices)
            default_factors = {
                'weather': {'temperature': 20.0, 'humidity': 50.0, 'wind_speed': 2.0},
                'holiday': {'is_holiday': False, 'holiday_name': 'ordinary day'},
                'time_features': {'hour': 12, 'dayofweek': 0, 'is_weekend': False}
            }
            return [default_factors for _ in range(batch_size)]


class LLMSemanticFeatureGenerator:
    """
    LLM语义特征生成器：整合LLM外因建模和特征生成
    """
    def __init__(self, configs):
        self.configs = configs
        self.llm_external_model = LLMExternalFactorModel(configs)
        self.external_factor_processor = ExternalFactorProcessor()
    
    def generate_semantic_features(self, dataset, batch_indices):
        """
        生成语义特征
        
        参数:
            dataset: 数据集实例
            batch_indices: 批次索引列表
        
        返回:
            semantic_features: [B, D] 语义特征
            impact_scores: [B, 1] 影响程度分数
        """
        # 提取外部因素
        external_factors = self.external_factor_processor.batch_extract_external_factors(dataset, batch_indices)
        
        # 使用LLM生成语义特征和影响分数
        semantic_features, impact_scores = self.llm_external_model(external_factors)
        
        return semantic_features, impact_scores
    
    def generate_semantic_features_from_batch(self, batch_data):
        """
        从批次数据生成语义特征
        
        参数:
            batch_data: 批次数据，包含外部因素信息
        
        返回:
            semantic_features: [B, D] 语义特征
            impact_scores: [B, 1] 影响程度分数
        """
        # 这里需要根据batch_data的具体结构实现
        # 暂时使用默认实现
        batch_size = len(batch_data)
        default_factors = {
            'weather': {'temperature': 20.0, 'humidity': 50.0, 'wind_speed': 2.0},
            'holiday': {'is_holiday': False, 'holiday_name': 'ordinary day'},
            'time_features': {'hour': 12, 'dayofweek': 0, 'is_weekend': False}
        }
        external_factors = [default_factors for _ in range(batch_size)]
        
        semantic_features, impact_scores = self.llm_external_model(external_factors)
        
        return semantic_features, impact_scores
    
    def extract_and_embed_external_factors(self, x_mark_enc, seq_len):
        """
        提取并嵌入外部因素（用于简单拼接模式）
        
        参数:
            x_mark_enc: [B, T, D] 时间标记特征
            seq_len: int 序列长度
        
        返回:
            semantic_features: [B, seq_len, D] 嵌入后的外部因素特征
        """
        B, T, D = x_mark_enc.shape
        
        # 提取外部因素（假设前4个特征是天气和节假日）
        if D >= 4:
            external_features = x_mark_enc[:, :, :4]
            # 简单嵌入
            embedding = nn.Linear(4, 768).to(x_mark_enc.device)
            semantic_features = embedding(external_features)
            # 调整长度以匹配时序特征
            if semantic_features.shape[1] != seq_len:
                # 重复或截断以匹配长度
                if semantic_features.shape[1] > seq_len:
                    semantic_features = semantic_features[:, :seq_len, :]
                else:
                    semantic_features = semantic_features.repeat(1, seq_len // semantic_features.shape[1] + 1, 1)[:, :seq_len, :]
        else:
            # 如果没有足够的特征，返回零向量
            semantic_features = torch.zeros(B, seq_len, 768).to(x_mark_enc.device)
        
        return semantic_features
    
    def llm_external_model(self, external_factors):
        """
        调用LLM外部因素模型
        
        参数:
            external_factors: 外部因素字典列表
        
        返回:
            semantic_features: [B, D] 语义特征
            impact_scores: [B, 1] 影响程度分数
            factor_weights: [B, 3] 多维权重向量
        """
        return self.llm_external_model.forward(external_factors)


# 测试代码
if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    
    # 创建测试配置
    class TestConfig:
        llm_model = 'BERT'
        llm_layers = 6
        llm_dim = 768
        d_model = 512
    
    config = TestConfig()
    
    # 创建LLM外因模型
    llm_external_model = LLMExternalFactorModel(config)
    
    # 创建测试数据
    test_factors = [
        {
            'weather': {'temperature': 25.5, 'humidity': 65.0, 'wind_speed': 3.2},
            'holiday': {'is_holiday': True, 'holiday_name': 'National Day'},
            'time_features': {'hour': 10, 'dayofweek': 5, 'is_weekend': True}
        },
        {
            'weather': {'temperature': 18.3, 'humidity': 45.0, 'wind_speed': 2.1},
            'holiday': {'is_holiday': False, 'holiday_name': 'ordinary day'},
            'time_features': {'hour': 14, 'dayofweek': 2, 'is_weekend': False}
        }
    ]
    
    # 测试前向传播
    semantic_features, impact_scores = llm_external_model(test_factors)
    
    print(f"Semantic features shape: {semantic_features.shape}")
    print(f"Impact scores shape: {impact_scores.shape}")
    print(f"Impact scores: {impact_scores.detach().numpy().flatten()}")
    
    print("\nLLM External Factor Model test completed successfully!")

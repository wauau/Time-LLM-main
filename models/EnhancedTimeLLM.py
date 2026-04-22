import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel, BertTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer
from layers.Embed import PatchEmbedding
from layers.StandardNorm import Normalize
from math import sqrt
import transformers

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class SemanticPromptConstructor:
    """
    语义Prompt构造器：构造包含任务背景、历史统计特征和外部因素的完整Prompt
    """
    def __init__(self, task_description, pred_len, seq_len):
        self.task_description = task_description
        self.pred_len = pred_len
        self.seq_len = seq_len
    
    def construct_prompt(self, batch_stats, external_factors):
        """
        构造完整的语义Prompt
        
        参数:
            batch_stats: 批次统计特征字典，包含min, max, median, trend, lags等
            external_factors: 外部因素字典，包含weather, holiday等
        
        返回:
            list: 批次prompt列表
        """
        prompts = []
        batch_size = len(batch_stats['min_values'])
        
        for b in range(batch_size):
            # 1. 任务背景
            prompt_parts = [
                f"<|start_prompt|>",
                f"Task: {self.task_description}",
                f"Forecast the next {self.pred_len} steps given the previous {self.seq_len} steps information."
            ]
            
            # 2. 历史统计特征
            prompt_parts.append("Historical Statistics:")
            prompt_parts.append(f"- Min value: {batch_stats['min_values'][b]:.2f}")
            prompt_parts.append(f"- Max value: {batch_stats['max_values'][b]:.2f}")
            prompt_parts.append(f"- Median value: {batch_stats['median_values'][b]:.2f}")
            prompt_parts.append(f"- Trend: {'upward' if batch_stats['trends'][b] > 0 else 'downward'}")
            prompt_parts.append(f"- Top 5 lags: {batch_stats['lags'][b]}")
            
            # 3. 外部因素语义描述
            prompt_parts.append("External Factors:")
            
            # 天气因素
            if external_factors.get('weather') is not None:
                weather = external_factors['weather'][b]
                temp = weather.get('temperature', 0)
                humidity = weather.get('humidity', 0)
                wind_speed = weather.get('wind_speed', 0)
                weather_desc = self._describe_weather(temp, humidity, wind_speed)
                prompt_parts.append(f"- Weather: {weather_desc}")
            
            # 节假日因素
            if external_factors.get('holiday') is not None:
                holiday = external_factors['holiday'][b]
                holiday_desc = self._describe_holiday(holiday)
                prompt_parts.append(f"- Holiday: {holiday_desc}")
            
            # 时间特征
            if external_factors.get('time_features') is not None:
                time_feat = external_factors['time_features'][b]
                time_desc = self._describe_time_features(time_feat)
                prompt_parts.append(f"- Time: {time_desc}")
            
            # 4. 预测目标
            prompt_parts.append("Prediction Target: People flow in the next time steps.")
            prompt_parts.append("<|end_prompt|>")
            
            prompt = " ".join(prompt_parts)
            prompts.append(prompt)
        
        return prompts
    
    def _describe_weather(self, temp, humidity, wind_speed):
        """生成天气的语义描述"""
        temp_desc = "hot" if temp > 30 else "warm" if temp > 20 else "cool" if temp > 10 else "cold"
        humidity_desc = "humid" if humidity > 70 else "moderate" if humidity > 40 else "dry"
        wind_desc = "windy" if wind_speed > 5 else "breezy" if wind_speed > 2 else "calm"
        return f"{temp_desc} ({temp:.1f}°C), {humidity_desc} ({humidity:.1f}%), {wind_desc} ({wind_speed:.1f} m/s)"
    
    def _describe_holiday(self, holiday_info):
        """生成节假日的语义描述"""
        is_holiday = holiday_info.get('is_holiday', False)
        holiday_name = holiday_info.get('holiday_name', 'ordinary day')
        if is_holiday:
            return f"holiday ({holiday_name}), expected increased flow"
        else:
            return "working day, normal flow pattern"
    
    def _describe_time_features(self, time_feat):
        """生成时间特征的语义描述"""
        hour = time_feat.get('hour', 0)
        dayofweek = time_feat.get('dayofweek', 0)
        is_weekend = time_feat.get('is_weekend', False)
        
        hour_desc = "morning peak" if 7 <= hour <= 9 else "evening peak" if 17 <= hour <= 19 else "off-peak"
        day_desc = "weekend" if is_weekend else "weekday"
        
        return f"{hour_desc} (hour {hour}), {day_desc}"


class CrossAttentionFusion(nn.Module):
    """
    交叉注意力融合模块：动态学习外部因素对时序特征的影响权重
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Query来自时序特征，Key和Value来自外部因素语义特征
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        
        # 输出投影
        self.out_projection = nn.Linear(d_model, d_model)
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, temporal_features, semantic_features):
        """
        前向传播
        
        参数:
            temporal_features: [B, L, D] 时序特征
            semantic_features: [B, S, D] 外部因素语义特征
        
        返回:
            fused_features: [B, L, D] 融合后的特征
            attention_weights: [B, n_heads, L, S] 注意力权重
        """
        B, L, D = temporal_features.shape
        S = semantic_features.shape[1]
        
        # 计算Q, K, V
        Q = self.query_projection(temporal_features).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.key_projection(semantic_features).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        V = self.value_projection(semantic_features).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 加权求和
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(B, L, D)
        
        # 输出投影
        attended = self.out_projection(attended)
        
        # 门控融合
        gate = self.gate(torch.cat([temporal_features, attended], dim=-1))
        fused_features = gate * temporal_features + (1 - gate) * attended
        
        # 残差连接和层归一化
        fused_features = self.layer_norm(fused_features + temporal_features)
        
        return fused_features, attention_weights


class SimilarityAlignmentFusion(nn.Module):
    """
    相似度对齐融合模块：基于相似度计算自适应权重
    """
    def __init__(self, d_model, dropout=0.1):
        super(SimilarityAlignmentFusion, self).__init__()
        self.d_model = d_model
        
        # 相似度计算
        self.similarity_projection = nn.Linear(d_model, d_model)
        
        # 自适应权重生成
        self.weight_network = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # 特征融合
        self.fusion_layer = nn.Linear(d_model * 2, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, temporal_features, semantic_features):
        """
        前向传播
        
        参数:
            temporal_features: [B, L, D] 时序特征
            semantic_features: [B, L, D] 外部因素语义特征（已对齐到相同长度）
        
        返回:
            fused_features: [B, L, D] 融合后的特征
            similarity_weights: [B, L] 相似度权重
        """
        # 计算相似度
        temporal_proj = self.similarity_projection(temporal_features)
        semantic_proj = self.similarity_projection(semantic_features)
        
        # 余弦相似度
        similarity = F.cosine_similarity(temporal_proj, semantic_proj, dim=-1)
        similarity_weights = similarity.unsqueeze(-1)
        
        # 自适应权重生成
        combined_features = torch.cat([temporal_features, semantic_features], dim=-1)
        adaptive_weights = self.weight_network(combined_features)
        
        # 组合权重
        final_weights = similarity_weights * adaptive_weights
        
        # 加权融合
        fused_features = final_weights * temporal_features + (1 - final_weights) * semantic_features
        
        # 进一步融合
        fused_features = self.fusion_layer(torch.cat([temporal_features, fused_features], dim=-1))
        
        # 残差连接和层归一化
        fused_features = self.layer_norm(fused_features + temporal_features)
        
        return fused_features, final_weights.squeeze(-1)


class EnhancedTimeLLM(nn.Module):
    """
    增强版TimeLLM：LLM语义驱动 + 外因影响自适应建模
    """
    def __init__(self, configs, patch_len=16, stride=8):
        super(EnhancedTimeLLM, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        
        # 初始化LLM
        self._init_llm(configs)
        
        # 语义Prompt构造器
        task_desc = configs.content if hasattr(configs, 'prompt_domain') and configs.prompt_domain else "People flow forecasting"
        self.prompt_constructor = SemanticPromptConstructor(task_desc, self.pred_len, self.seq_len)
        
        # Patch嵌入
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)
        
        # 词嵌入映射
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        
        # 重编程层
        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)
        
        # 自适应融合模块
        self.use_cross_attention = getattr(configs, 'use_cross_attention', True)
        if self.use_cross_attention:
            self.fusion_module = CrossAttentionFusion(self.d_ff, configs.n_heads, self.d_ff, configs.dropout)
        else:
            self.fusion_module = SimilarityAlignmentFusion(self.d_ff, configs.dropout)
        
        # 外部因素嵌入
        self.external_embedding = nn.Linear(4, self.d_ff)  # weather: 3 features + holiday: 1 feature
        
        # 输出投影
        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError
        
        # 归一化层
        self.normalize_layers = Normalize(configs.enc_in, affine=False)
        
        # Dropout
        self.dropout = nn.Dropout(configs.dropout)
    
    def _init_llm(self, configs):
        """初始化LLM模型"""
        if configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')
            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )
            
            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Attempting to download...")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('Only BERT is supported in EnhancedTimeLLM')
        
        # 设置pad token
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token
        
        # 冻结LLM参数
        for param in self.llm_model.parameters():
            param.requires_grad = False
    
    def extract_external_factors(self, x_mark_enc):
        """
        从时间标记中提取外部因素
        
        参数:
            x_mark_enc: [B, T, D] 时间标记特征
        
        返回:
            external_factors: dict 外部因素字典
        """
        B, T, D = x_mark_enc.shape
        external_factors = {
            'weather': [],
            'holiday': [],
            'time_features': []
        }
        
        for b in range(B):
            # 提取天气特征（假设在x_mark_enc中）
            # 这里需要根据实际数据结构调整
            weather_features = {
                'temperature': 20.0,  # 默认值，需要从实际数据中提取
                'humidity': 50.0,
                'wind_speed': 2.0
            }
            external_factors['weather'].append(weather_features)
            
            # 提取节假日特征
            holiday_features = {
                'is_holiday': False,
                'holiday_name': 'ordinary day'
            }
            external_factors['holiday'].append(holiday_features)
            
            # 提取时间特征
            time_features = {
                'hour': int(x_mark_enc[b, 0, 3].item()) if D > 3 else 12,
                'dayofweek': int(x_mark_enc[b, 0, 2].item()) if D > 2 else 0,
                'is_weekend': False
            }
            external_factors['time_features'].append(time_features)
        
        return external_factors
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 归一化
        x_enc = self.normalize_layers(x_enc, 'norm')
        
        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        
        # 计算统计特征
        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)
        
        # 构造统计特征字典
        batch_stats = {
            'min_values': min_values.squeeze(-1).cpu().numpy(),
            'max_values': max_values.squeeze(-1).cpu().numpy(),
            'median_values': medians.squeeze(-1).cpu().numpy(),
            'trends': trends.squeeze(-1).cpu().numpy(),
            'lags': lags.cpu().numpy()
        }
        
        # 提取外部因素
        external_factors = self.extract_external_factors(x_mark_enc)
        
        # 构造语义Prompt
        prompts = self.prompt_constructor.construct_prompt(batch_stats, external_factors)
        
        # 恢复x_enc形状
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()
        
        # 生成Prompt embedding
        prompt = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))
        
        # 时序特征提取
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc)
        
        # 重编程
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        
        # 外部因素嵌入
        if x_mark_enc.shape[-1] >= 4:
            external_emb = self.external_embedding(x_mark_enc[:, :, :4])
            external_emb = external_emb.mean(dim=1, keepdim=True).repeat(1, enc_out.shape[1], 1)
        else:
            external_emb = torch.zeros_like(enc_out)
        
        # 自适应融合
        if self.use_cross_attention:
            fused_features, attention_weights = self.fusion_module(enc_out, external_emb)
        else:
            # 对齐特征长度
            min_len = min(enc_out.shape[1], external_emb.shape[1])
            enc_out_aligned = enc_out[:, :min_len, :]
            external_emb_aligned = external_emb[:, :min_len, :]
            fused_features, similarity_weights = self.fusion_module(enc_out_aligned, external_emb_aligned)
        
        # 拼接prompt embedding和融合特征
        llama_enc_out = torch.cat([prompt_embeddings, fused_features], dim=1)
        
        # LLM处理
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]
        
        # 重塑和投影
        dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        
        # 反归一化
        dec_out = self.normalize_layers(dec_out, 'denorm')
        
        return dec_out
    
    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding


def create_enhanced_model(configs):
    """
    创建增强版TimeLLM模型的工厂函数
    
    参数:
        configs: 配置对象
    
    返回:
        EnhancedTimeLLM: 增强版模型实例
    """
    return EnhancedTimeLLM(configs)

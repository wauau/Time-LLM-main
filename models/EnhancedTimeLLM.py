import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel, BertTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer
from layers.Embed import PatchEmbedding
from layers.StandardNorm import Normalize
from math import sqrt
import transformers
from models.llm_external_model import LLMSemanticFeatureGenerator
from models.fusion_modules import DynamicFusionModule, GateFusion, AttentionFusion

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
    明确分离四个模块：时间序列分支、LLM外因分支、融合层、预测层
    """
    def __init__(self, configs, patch_len=16, stride=8):
        super(EnhancedTimeLLM, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        
        # 实验配置
        self.use_external_factors = getattr(configs, 'use_external_factors', True)
        self.use_llm_semantic = getattr(configs, 'use_llm_semantic', True)
        self.use_fusion = getattr(configs, 'use_fusion', True)
        self.fusion_mode = getattr(configs, 'fusion_mode', 'attention')  # attention, gate, concat
        
        # 对比实验模式
        self.experiment_mode = getattr(configs, 'experiment_mode', 'full')  # full, no_external, simple_concat, llm_only
        
        # 根据实验模式设置参数
        if self.experiment_mode == 'no_external':
            # 无外因模式
            self.use_external_factors = False
        elif self.experiment_mode == 'simple_concat':
            # 简单拼接模式
            self.use_external_factors = True
            self.use_llm_semantic = False
            self.use_fusion = True
            self.fusion_mode = 'concat'
        elif self.experiment_mode == 'llm_only':
            # LLM外因模式（无融合）
            self.use_external_factors = True
            self.use_llm_semantic = True
            self.use_fusion = False
        
        # 1. 时间序列分支
        self.temporal_branch = TemporalBranch(configs, self.patch_len, self.stride)
        
        # 2. LLM外因分支
        self.llm_external_branch = LLMSemanticFeatureGenerator(configs)
        
        # 3. 融合层
        self.fusion_layer = FusionLayer(configs, self.d_llm)
        
        # 4. 预测层
        self.prediction_layer = PredictionLayer(configs, self.patch_len, self.stride)
        
        # 归一化层
        self.normalize_layers = Normalize(configs.enc_in, affine=False)
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, external_factors=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, external_factors)
            return dec_out[:, -self.pred_len:, :]
        return None
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, external_factors=None):
        # 归一化
        x_enc = self.normalize_layers(x_enc, 'norm')
        
        B, T, N = x_enc.size()
        
        # 1. 时间序列分支：提取时序特征
        temporal_features, batch_stats, n_vars = self.temporal_branch(x_enc)
        
        # 2. LLM外因分支：处理外部因素
        semantic_features = None
        impact_scores = None
        factor_weights = None
        
        if self.use_external_factors:
            if self.use_llm_semantic and external_factors is not None:
                # 使用LLM语义建模外部因素
                semantic_features, impact_scores, factor_weights = self.llm_external_branch.llm_external_model(external_factors)
            else:
                # 传统外部因素嵌入（简单拼接模式）
                semantic_features = self.llm_external_branch.extract_and_embed_external_factors(x_mark_enc, temporal_features.shape[1])
        
        # 3. 融合层：生成融合特征X_fused
        # 初始化为时序特征，当没有外部因素或不使用融合时直接使用时序特征
        X_fused = temporal_features
        fusion_weights = None
        
        if self.use_fusion and semantic_features is not None:
            # 动态融合：X_fused = X_base + α·X_external
            # 这里是X_fused的核心生成位置，通过可学习的融合机制将外部因素融入时序特征
            X_fused, fusion_weights = self.fusion_layer(temporal_features, semantic_features, impact_scores, factor_weights, self.fusion_mode)
        
        # X_fused现在包含了融合后的特征，将用于后续的预测
        
        # 4. 预测层：使用融合特征进行预测
        dec_out = self.prediction_layer(X_fused, batch_stats, n_vars, self.configs.d_model)
        
        # 反归一化
        dec_out = self.normalize_layers(dec_out, 'denorm')
        
        return dec_out


class TemporalBranch(nn.Module):
    """
    时间序列分支：负责提取时间序列特征
    """
    def __init__(self, configs, patch_len, stride):
        super(TemporalBranch, self).__init__()
        self.configs = configs
        self.patch_len = patch_len
        self.stride = stride
        self.top_k = 5
        
        # Patch嵌入
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, configs.dropout)
        
        # 词嵌入映射
        self.word_embeddings = None
        self.mapping_layer = None
        
        # 重编程层
        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, configs.d_ff, configs.llm_dim)
    
    def forward(self, x_enc):
        """
        前向传播
        
        参数:
            x_enc: [B, T, N] 输入时间序列
        
        返回:
            temporal_features: [B, L, D] 时序特征
            batch_stats: dict 批次统计特征
            n_vars: int 变量数量
        """
        B, T, N = x_enc.size()
        x_enc_reshaped = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        
        # 计算统计特征
        min_values = torch.min(x_enc_reshaped, dim=1)[0]
        max_values = torch.max(x_enc_reshaped, dim=1)[0]
        medians = torch.median(x_enc_reshaped, dim=1).values
        lags = self.calcute_lags(x_enc_reshaped)
        trends = x_enc_reshaped.diff(dim=1).sum(dim=1)
        
        # 恢复x_enc形状
        x_enc = x_enc_reshaped.reshape(B, N, T).permute(0, 2, 1).contiguous()
        
        # 时序特征提取
        if self.word_embeddings is None:
            # 延迟初始化词嵌入，因为需要访问LLM模型
            from transformers import BertModel, BertConfig
            temp_config = BertConfig(
                vocab_size=30522,
                hidden_size=768,
                num_hidden_layers=2,
                num_attention_heads=12,
                intermediate_size=3072
            )
            temp_model = BertModel(temp_config)
            self.word_embeddings = temp_model.get_input_embeddings().weight
            self.vocab_size = self.word_embeddings.shape[0]
            self.num_tokens = 1000
            self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens).to(x_enc.device)
        
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc)
        
        # 重编程
        temporal_features = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        
        # 构建批次统计特征
        batch_stats = {
            'min_values': min_values.squeeze(-1).cpu().numpy(),
            'max_values': max_values.squeeze(-1).cpu().numpy(),
            'median_values': medians.squeeze(-1).cpu().numpy(),
            'trends': trends.squeeze(-1).cpu().numpy(),
            'lags': lags.cpu().numpy()
        }
        
        return temporal_features, batch_stats, n_vars
    
    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class FusionLayer(nn.Module):
    """
    融合层：实现可学习的融合机制
    """
    def __init__(self, configs, d_llm):
        super(FusionLayer, self).__init__()
        self.configs = configs
        self.d_llm = d_llm
        
        # 维度投影层：将语义特征维度映射到LLM隐藏状态维度
        self.dimension_projection = nn.Linear(configs.d_model, d_llm)
        
        # 传统外部因素嵌入（仅用于简单拼接模式）
        self.external_embedding = nn.Linear(4, d_llm)  # weather: 3 features + holiday: 1 feature
        
        # 动态融合模块
        self.dynamic_fusion = DynamicFusionModule(d_llm, configs.n_heads, configs.dropout)
        
        # 门控融合模块
        self.gate_fusion = GateFusion(d_llm, configs.dropout)
        
        # 注意力融合模块
        self.attention_fusion = AttentionFusion(d_llm, configs.n_heads, configs.dropout)
        
        # 可学习的权重网络：用于生成融合权重α
        self.weight_network = nn.Sequential(
            nn.Linear(d_llm * 2, d_llm),
            nn.ReLU(),
            nn.Linear(d_llm, 1),
            nn.Sigmoid()  # 输出0-1的权重
        )
        
        # 因素权重融合网络
        self.factor_weight_network = nn.Sequential(
            nn.Linear(3, d_llm),
            nn.ReLU(),
            nn.Linear(d_llm, 1),
            nn.Sigmoid()
        )
    
    def forward(self, temporal_features, semantic_features, impact_scores=None, factor_weights=None, fusion_mode='attention'):
        """
        前向传播
        
        参数:
            temporal_features: [B, L, D] 时序特征
            semantic_features: [B, D] 或 [B, L, D] 语义特征
            impact_scores: [B, 1] 影响程度分数
            factor_weights: [B, 3] 多维权重向量（天气、节假日、时间）
            fusion_mode: str 融合模式 ('attention', 'gate', 'concat')
        
        返回:
            X_fused: [B, L, D] 融合后的特征
            fusion_weights: 融合权重
        """
        B, L, D = temporal_features.shape
        
        # 处理语义特征形状
        if semantic_features.dim() == 2:
            # [B, D] -> [B, 1, D] -> [B, L, D]
            semantic_features = semantic_features.unsqueeze(1).repeat(1, L, 1)
        
        # 投影到相同维度
        if semantic_features.shape[-1] != D:
            semantic_features = self.dimension_projection(semantic_features)
        
        # 生成可学习的融合权重α
        # 方法1：使用时间特征和语义特征的拼接来预测权重
        combined_features = torch.cat([temporal_features, semantic_features], dim=-1)
        alpha = self.weight_network(combined_features)  # [B, L, 1]
        
        # 方法2：如果提供了factor_weights，结合因素权重调整融合权重
        if factor_weights is not None:
            # 处理因素权重
            factor_alpha = self.factor_weight_network(factor_weights)  # [B, 1]
            factor_alpha = factor_alpha.unsqueeze(1).repeat(1, L, 1)  # [B, L, 1]
            # 结合两种权重
            alpha = alpha * factor_alpha
        
        # 显式融合公式：X_fused = X_base + α·X_external
        X_fused = temporal_features + alpha * semantic_features
        
        # 可选：使用其他融合模式进一步优化
        if fusion_mode == 'attention':
            # 使用注意力机制进一步融合
            attention_fused, fusion_weights = self.attention_fusion(temporal_features, semantic_features)
            # 结合注意力融合和显式融合
            X_fused = 0.5 * X_fused + 0.5 * attention_fused
        elif fusion_mode == 'gate':
            # 使用门控机制进一步融合
            gate_fused = self.gate_fusion(temporal_features, semantic_features, impact_scores)
            # 结合门控融合和显式融合
            X_fused = 0.5 * X_fused + 0.5 * gate_fused
        elif fusion_mode == 'concat':
            # 简单拼接（用于对比实验）
            concat_fused = torch.cat([temporal_features, semantic_features], dim=-1)
            concat_fused = nn.Linear(concat_fused.shape[-1], D).to(concat_fused.device)(concat_fused)
            X_fused = concat_fused
            fusion_weights = None
        else:
            # 默认只使用显式融合
            fusion_weights = alpha
        
        return X_fused, fusion_weights


class PredictionLayer(nn.Module):
    """
    预测层：使用融合特征进行最终预测
    """
    def __init__(self, configs, patch_len, stride):
        super(PredictionLayer, self).__init__()
        self.configs = configs
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.patch_len = patch_len
        self.stride = stride
        
        # 计算patch数量
        self.patch_nums = int((configs.seq_len - patch_len) / stride + 2)
        # 使用d_model作为特征维度
        self.head_nf = configs.d_model * self.patch_nums
        
        # 输出投影
        if configs.task_name == 'long_term_forecast' or configs.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError
        
        # 初始化LLM用于预测
        self.llm_model = self._init_llm(configs)
        self.tokenizer = self._init_tokenizer(configs)
    
    def _init_llm(self, configs):
        """初始化LLM模型"""
        if configs.llm_model == 'BERT':
            # 直接创建BERT配置
            from transformers import BertConfig, BertModel
            bert_config = BertConfig(
                vocab_size=30522,
                hidden_size=768,
                num_hidden_layers=configs.llm_layers,
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
            llm_model = BertModel(bert_config)
            print("Created BERT model with random weights")
        else:
            raise Exception('Only BERT is supported in EnhancedTimeLLM')
        
        # 冻结LLM参数
        for param in llm_model.parameters():
            param.requires_grad = False
        
        return llm_model
    
    def _init_tokenizer(self, configs):
        """初始化Tokenizer"""
        if configs.llm_model == 'BERT':
            from transformers import BertTokenizerFast
            tokenizer = BertTokenizerFast(
                vocab_file=None,
                do_lower_case=True,
                strip_accents=True,
                tokenize_chinese_chars=False,
                wordpiece_prefix="##"
            )
            # 添加特殊 tokens
            special_tokens = {'pad_token': '[PAD]', 'cls_token': '[CLS]', 'sep_token': '[SEP]'}
            tokenizer.add_special_tokens(special_tokens)
            print("Created BERT tokenizer with basic configuration")
        else:
            raise Exception('Only BERT is supported in EnhancedTimeLLM')
        
        return tokenizer
    
    def forward(self, fused_features, batch_stats, n_vars, d_model):
        """
        前向传播
        
        参数:
            fused_features: [B, L, D] 融合后的特征
            batch_stats: dict 批次统计特征
            n_vars: int 变量数量
            d_model: int 模型维度
        
        返回:
            dec_out: [B, T, N] 预测结果
        """
        B = fused_features.shape[0]
        
        # 构造基础Prompt
        prompts = []
        for b in range(B):
            prompt_parts = [
                f"<|start_prompt|>",
                "Task: People flow forecasting",
                f"Forecast the next {self.pred_len} steps given the previous {self.seq_len} steps information.",
                "Historical Statistics:",
                f"- Min value: {batch_stats['min_values'][b]:.2f}",
                f"- Max value: {batch_stats['max_values'][b]:.2f}",
                f"- Median value: {batch_stats['median_values'][b]:.2f}",
                f"- Trend: {'upward' if batch_stats['trends'][b] > 0 else 'downward'}",
                f"- Top 5 lags: {batch_stats['lags'][b]}",
                "Prediction Target: People flow in the next time steps.",
                "<|end_prompt|>"
            ]
            prompts.append(" ".join(prompt_parts))
        
        # 生成Prompt embedding
        prompt = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(fused_features.device))
        
        # 拼接prompt embedding和融合特征
        llama_enc_out = torch.cat([prompt_embeddings, fused_features], dim=1)
        
        # LLM处理
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        # 使用d_model作为特征维度
        dec_out = dec_out[:, :, :d_model]
        
        # 重塑和投影
        dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        
        return dec_out


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

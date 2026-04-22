import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class DynamicFusionModule(nn.Module):
    """
    动态融合模块：使用attention和gate机制，由模型自动计算外部因素对人流的动态影响权重
    """
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super(DynamicFusionModule, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 注意力机制
        self.attention = AttentionFusion(d_model, n_heads, dropout)
        
        # 门控机制
        self.gate = GateFusion(d_model, dropout)
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, temporal_features, semantic_features, impact_scores=None):
        """
        前向传播
        
        参数:
            temporal_features: [B, L, D] 时序特征
            semantic_features: [B, D] 或 [B, L, D] 语义特征
            impact_scores: [B, 1] 影响程度分数（可选）
        
        返回:
            fused_features: [B, L, D] 融合后的特征
            attention_weights: [B, n_heads, L, S] 注意力权重
        """
        # 处理语义特征形状
        if semantic_features.dim() == 2:
            # [B, D] -> [B, 1, D] -> [B, L, D]
            B, D = semantic_features.shape
            L = temporal_features.shape[1]
            semantic_features = semantic_features.unsqueeze(1).repeat(1, L, 1)
        
        # 使用注意力机制融合
        attended_features, attention_weights = self.attention(temporal_features, semantic_features)
        
        # 使用门控机制融合
        if impact_scores is not None:
            # 将影响分数融入门控机制
            gate_output = self.gate(temporal_features, attended_features, impact_scores)
        else:
            gate_output = self.gate(temporal_features, attended_features)
        
        # 进一步融合特征
        combined_features = torch.cat([temporal_features, gate_output], dim=-1)
        fused_features = self.feature_fusion(combined_features)
        
        # 残差连接和层归一化
        fused_features = self.layer_norm(fused_features + temporal_features)
        
        return fused_features, attention_weights


class AttentionFusion(nn.Module):
    """
    注意力融合模块：计算时序特征和语义特征之间的注意力权重
    """
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super(AttentionFusion, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 投影层
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value=None):
        """
        前向传播
        
        参数:
            query: [B, L, D] 查询（时序特征）
            key: [B, S, D] 键（语义特征）
            value: [B, S, D] 值（语义特征，默认与key相同）
        
        返回:
            attended: [B, L, D] 注意力加权后的特征
            attention_weights: [B, n_heads, L, S] 注意力权重
        """
        if value is None:
            value = key
        
        B, L, D = query.shape
        S = key.shape[1]
        
        # 投影到多头注意力空间
        Q = self.query_proj(query).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.key_proj(key).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        V = self.value_proj(value).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 加权求和
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(B, L, D)
        
        # 输出投影
        attended = self.output_proj(attended)
        
        return attended, attention_weights


class GateFusion(nn.Module):
    """
    门控融合模块：使用门控机制控制语义特征的融入程度
    """
    def __init__(self, d_model, dropout=0.1):
        super(GateFusion, self).__init__()
        self.d_model = d_model
        
        # 门控网络
        self.gate_network = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        # 影响分数融合
        self.impact_fusion = nn.Linear(d_model + 1, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, temporal_features, semantic_features, impact_scores=None):
        """
        前向传播
        
        参数:
            temporal_features: [B, L, D] 时序特征
            semantic_features: [B, L, D] 语义特征
            impact_scores: [B, 1] 影响程度分数（可选）
        
        返回:
            fused_features: [B, L, D] 融合后的特征
        """
        # 计算门控信号
        combined = torch.cat([temporal_features, semantic_features], dim=-1)
        gate = self.gate_network(combined)
        
        # 应用门控
        gated_semantic = gate * semantic_features
        gated_temporal = (1 - gate) * temporal_features
        fused_features = gated_temporal + gated_semantic
        
        # 如果提供了影响分数，进一步调整融合
        if impact_scores is not None:
            # 扩展影响分数到时序维度
            B, L, D = fused_features.shape
            impact = impact_scores.unsqueeze(1).repeat(1, L, 1)
            # 融合影响分数
            combined_with_impact = torch.cat([fused_features, impact], dim=-1)
            fused_features = self.impact_fusion(combined_with_impact)
        
        fused_features = self.dropout(fused_features)
        
        return fused_features


class AdaptiveWeightFusion(nn.Module):
    """
    自适应权重融合模块：根据外部因素的影响动态调整权重
    """
    def __init__(self, d_model, dropout=0.1):
        super(AdaptiveWeightFusion, self).__init__()
        self.d_model = d_model
        
        # 权重预测网络
        self.weight_network = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # 特征融合
        self.fusion_layer = nn.Linear(d_model * 2, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, temporal_features, semantic_features, impact_scores=None):
        """
        前向传播
        
        参数:
            temporal_features: [B, L, D] 时序特征
            semantic_features: [B, L, D] 语义特征
            impact_scores: [B, 1] 影响程度分数（可选）
        
        返回:
            fused_features: [B, L, D] 融合后的特征
            weights: [B, L, 1] 自适应权重
        """
        # 计算自适应权重
        combined = torch.cat([temporal_features, semantic_features], dim=-1)
        weights = self.weight_network(combined)
        
        # 如果提供了影响分数，调整权重
        if impact_scores is not None:
            B, L, _ = weights.shape
            impact = impact_scores.unsqueeze(1).repeat(1, L, 1)
            weights = weights * impact
        
        # 加权融合
        weighted_temporal = weights * temporal_features
        weighted_semantic = (1 - weights) * semantic_features
        fused_features = weighted_temporal + weighted_semantic
        
        # 进一步融合
        combined_features = torch.cat([temporal_features, fused_features], dim=-1)
        fused_features = self.fusion_layer(combined_features)
        
        fused_features = self.dropout(fused_features)
        
        return fused_features, weights


class MultiScaleFusion(nn.Module):
    """
    多尺度融合模块：在不同尺度上融合特征
    """
    def __init__(self, d_model, scales=[1, 2, 4], dropout=0.1):
        super(MultiScaleFusion, self).__init__()
        self.d_model = d_model
        self.scales = scales
        
        # 多尺度卷积
        self.scale_convs = nn.ModuleList()
        for scale in scales:
            self.scale_convs.append(
                nn.Conv1d(d_model, d_model, kernel_size=scale, padding=scale//2)
            )
        
        # 融合网络
        self.fusion_network = nn.Sequential(
            nn.Linear(d_model * (len(scales) + 2), d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, temporal_features, semantic_features):
        """
        前向传播
        
        参数:
            temporal_features: [B, L, D] 时序特征
            semantic_features: [B, L, D] 语义特征
        
        返回:
            fused_features: [B, L, D] 融合后的特征
        """
        B, L, D = temporal_features.shape
        
        # 多尺度特征提取
        scale_features = []
        for conv in self.scale_convs:
            # [B, L, D] -> [B, D, L] -> [B, D, L] -> [B, L, D]
            feat = conv(temporal_features.transpose(1, 2)).transpose(1, 2)
            scale_features.append(feat)
        
        # 组合所有特征
        all_features = [temporal_features, semantic_features] + scale_features
        combined = torch.cat(all_features, dim=-1)
        
        # 融合
        fused_features = self.fusion_network(combined)
        
        # 残差连接
        fused_features = fused_features + temporal_features
        
        fused_features = self.dropout(fused_features)
        
        return fused_features


# 测试代码
if __name__ == '__main__':
    # 创建测试数据
    B = 4
    L = 24
    D = 512
    
    temporal_features = torch.randn(B, L, D)
    semantic_features = torch.randn(B, D)
    impact_scores = torch.randn(B, 1)
    
    # 测试动态融合模块
    dynamic_fusion = DynamicFusionModule(D, n_heads=8)
    fused, attention = dynamic_fusion(temporal_features, semantic_features, impact_scores)
    print(f"Dynamic Fusion Output shape: {fused.shape}")
    print(f"Attention weights shape: {attention.shape}")
    
    # 测试注意力融合模块
    attention_fusion = AttentionFusion(D, n_heads=8)
    attended, attn_weights = attention_fusion(temporal_features, semantic_features.unsqueeze(1))
    print(f"Attention Fusion Output shape: {attended.shape}")
    
    # 测试门控融合模块
    gate_fusion = GateFusion(D)
    gated = gate_fusion(temporal_features, semantic_features.unsqueeze(1), impact_scores)
    print(f"Gate Fusion Output shape: {gated.shape}")
    
    # 测试自适应权重融合模块
    adaptive_fusion = AdaptiveWeightFusion(D)
    adaptive, weights = adaptive_fusion(temporal_features, semantic_features.unsqueeze(1), impact_scores)
    print(f"Adaptive Fusion Output shape: {adaptive.shape}")
    print(f"Adaptive Weights shape: {weights.shape}")
    
    # 测试多尺度融合模块
    multi_scale_fusion = MultiScaleFusion(D)
    multi_scale = multi_scale_fusion(temporal_features, semantic_features.unsqueeze(1))
    print(f"Multi Scale Fusion Output shape: {multi_scale.shape}")
    
    print("\nAll fusion modules test completed successfully!")

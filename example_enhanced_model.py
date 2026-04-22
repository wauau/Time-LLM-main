"""
增强版TimeLLM模型使用示例

本脚本展示如何使用增强版TimeLLM模型进行人流预测，
该模型实现了"LLM语义驱动 + 外因影响自适应建模"的统一框架。
"""

import torch
import numpy as np
from models.EnhancedTimeLLM import EnhancedTimeLLM, SemanticPromptConstructor
from data_provider.data_loader_enhanced import create_enhanced_dataset


class EnhancedConfig:
    """增强版模型配置类"""
    def __init__(self):
        # 任务配置
        self.task_name = 'long_term_forecast'
        self.pred_len = 96
        self.seq_len = 96
        self.d_ff = 512
        self.top_k = 5
        self.d_llm = 768
        self.patch_len = 16
        self.stride = 8
        
        # 模型配置
        self.llm_model = 'BERT'
        self.llm_layers = 6
        self.prompt_domain = True
        self.content = 'People flow forecasting with external factors including weather and holidays'
        
        # 网络配置
        self.d_model = 512
        self.n_heads = 8
        self.e_layers = 2
        self.d_layers = 1
        self.dropout = 0.1
        
        # 融合配置
        self.use_cross_attention = True  # 使用交叉注意力融合
        # self.use_cross_attention = False  # 使用相似度对齐融合
        
        # 数据配置
        self.enc_in = 5  # 5个特征：人流、温度、湿度、风速、节假日
        self.dec_in = 5
        self.c_out = 5


def example_semantic_prompt_construction():
    """
    示例1：语义Prompt构造
    
    展示如何构造包含任务背景、历史统计特征和外部因素的完整Prompt
    """
    print("=" * 80)
    print("示例1：语义Prompt构造")
    print("=" * 80)
    
    # 创建Prompt构造器
    task_desc = "People flow forecasting with external factors including weather and holidays"
    prompt_constructor = SemanticPromptConstructor(task_desc, pred_len=96, seq_len=96)
    
    # 模拟批次统计特征
    batch_stats = {
        'min_values': np.array([10.5, 15.2]),
        'max_values': np.array([150.8, 180.5]),
        'median_values': np.array([65.3, 78.9]),
        'trends': np.array([1.2, -0.5]),
        'lags': np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    }
    
    # 模拟外部因素
    external_factors = {
        'weather': [
            {'temperature': 25.5, 'humidity': 65.0, 'wind_speed': 3.2},
            {'temperature': 18.3, 'humidity': 45.0, 'wind_speed': 2.1}
        ],
        'holiday': [
            {'is_holiday': False, 'holiday_name': 'ordinary day'},
            {'is_holiday': True, 'holiday_name': 'National Day'}
        ],
        'time_features': [
            {'hour': 14, 'dayofweek': 2, 'is_weekend': False},
            {'hour': 10, 'dayofweek': 5, 'is_weekend': True}
        ]
    }
    
    # 构造Prompt
    prompts = prompt_constructor.construct_prompt(batch_stats, external_factors)
    
    # 打印第一个样本的Prompt
    print("\n生成的语义Prompt（样本1）：")
    print("-" * 80)
    print(prompts[0])
    print("-" * 80)
    
    return prompts


def example_cross_attention_fusion():
    """
    示例2：交叉注意力融合
    
    展示如何使用交叉注意力机制动态学习外部因素对时序特征的影响权重
    """
    print("\n" + "=" * 80)
    print("示例2：交叉注意力融合")
    print("=" * 80)
    
    from models.EnhancedTimeLLM import CrossAttentionFusion
    
    # 创建交叉注意力融合模块
    d_model = 512
    n_heads = 8
    fusion_module = CrossAttentionFusion(d_model, n_heads, d_ff=512, dropout=0.1)
    
    # 模拟时序特征和语义特征
    batch_size = 4
    temporal_len = 24
    semantic_len = 10
    
    temporal_features = torch.randn(batch_size, temporal_len, d_model)
    semantic_features = torch.randn(batch_size, semantic_len, d_model)
    
    # 执行融合
    fused_features, attention_weights = fusion_module(temporal_features, semantic_features)
    
    print(f"\n输入时序特征形状: {temporal_features.shape}")
    print(f"输入语义特征形状: {semantic_features.shape}")
    print(f"融合后特征形状: {fused_features.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    
    # 分析注意力权重
    print(f"\n注意力权重统计:")
    print(f"  - 平均值: {attention_weights.mean().item():.4f}")
    print(f"  - 最大值: {attention_weights.max().item():.4f}")
    print(f"  - 最小值: {attention_weights.min().item():.4f}")
    
    return fused_features, attention_weights


def example_similarity_alignment_fusion():
    """
    示例3：相似度对齐融合
    
    展示如何基于相似度计算自适应权重进行特征融合
    """
    print("\n" + "=" * 80)
    print("示例3：相似度对齐融合")
    print("=" * 80)
    
    from models.EnhancedTimeLLM import SimilarityAlignmentFusion
    
    # 创建相似度对齐融合模块
    d_model = 512
    fusion_module = SimilarityAlignmentFusion(d_model, dropout=0.1)
    
    # 模拟时序特征和语义特征（相同长度）
    batch_size = 4
    seq_len = 24
    
    temporal_features = torch.randn(batch_size, seq_len, d_model)
    semantic_features = torch.randn(batch_size, seq_len, d_model)
    
    # 执行融合
    fused_features, similarity_weights = fusion_module(temporal_features, semantic_features)
    
    print(f"\n输入时序特征形状: {temporal_features.shape}")
    print(f"输入语义特征形状: {semantic_features.shape}")
    print(f"融合后特征形状: {fused_features.shape}")
    print(f"相似度权重形状: {similarity_weights.shape}")
    
    # 分析相似度权重
    print(f"\n相似度权重统计:")
    print(f"  - 平均值: {similarity_weights.mean().item():.4f}")
    print(f"  - 最大值: {similarity_weights.max().item():.4f}")
    print(f"  - 最小值: {similarity_weights.min().item():.4f}")
    print(f"  - 标准差: {similarity_weights.std().item():.4f}")
    
    return fused_features, similarity_weights


def example_full_model_forward():
    """
    示例4：完整模型前向传播
    
    展示如何使用完整的增强版TimeLLM模型进行预测
    """
    print("\n" + "=" * 80)
    print("示例4：完整模型前向传播")
    print("=" * 80)
    
    # 创建配置
    config = EnhancedConfig()
    
    # 创建模型
    model = EnhancedTimeLLM(config)
    model.eval()
    
    # 模拟输入数据
    batch_size = 4
    seq_len = 96
    pred_len = 96
    enc_in = 5
    
    x_enc = torch.randn(batch_size, seq_len, enc_in)  # 编码器输入
    x_mark_enc = torch.randn(batch_size, seq_len, 5)  # 编码器时间标记
    x_dec = torch.randn(batch_size, pred_len, enc_in)  # 解码器输入
    x_mark_dec = torch.randn(batch_size, pred_len, 5)  # 解码器时间标记
    
    print(f"\n输入数据形状:")
    print(f"  - x_enc: {x_enc.shape}")
    print(f"  - x_mark_enc: {x_mark_enc.shape}")
    print(f"  - x_dec: {x_dec.shape}")
    print(f"  - x_mark_dec: {x_mark_dec.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    
    print(f"\n输出数据形状: {output.shape}")
    print(f"  - 预测长度: {output.shape[1]}")
    print(f"  - 特征维度: {output.shape[2]}")
    
    return output


def example_data_loading():
    """
    示例5：数据加载
    
    展示如何使用增强版数据集加载器
    """
    print("\n" + "=" * 80)
    print("示例5：数据加载")
    print("=" * 80)
    
    # 创建数据集
    root_path = './dataset/数据/'
    data_path = 'processed_data.csv'
    
    try:
        # 尝试创建训练集
        train_dataset = create_enhanced_dataset(
            root_path=root_path,
            data_path=data_path,
            flag='train',
            size=[96, 48, 96],
            features='M',
            target='people',
            scale=True,
            timeenc=0,
            freq='10min',
            percent=100,
            dataset_type='people_flow'
        )
        
        print(f"\n训练集大小: {len(train_dataset)}")
        
        # 获取一个样本
        seq_x, seq_y, seq_x_mark, seq_y_mark = train_dataset[0]
        
        print(f"\n样本数据形状:")
        print(f"  - seq_x: {seq_x.shape}")
        print(f"  - seq_y: {seq_y.shape}")
        print(f"  - seq_x_mark: {seq_x_mark.shape}")
        print(f"  - seq_y_mark: {seq_y_mark.shape}")
        
        # 获取外部因素
        external_factors = train_dataset.get_external_factors_batch([0])
        print(f"\n外部因素（样本0）:")
        print(f"  - 天气: 温度={external_factors[0]['weather']['temperature']:.1f}°C, "
              f"湿度={external_factors[0]['weather']['humidity']:.1f}%")
        print(f"  - 节假日: {external_factors[0]['holiday']['holiday_name']}")
        print(f"  - 时间: {external_factors[0]['time_features']['hour']}点, "
              f"{'周末' if external_factors[0]['time_features']['is_weekend'] else '工作日'}")
        
        return train_dataset
        
    except FileNotFoundError:
        print(f"\n数据文件未找到: {os.path.join(root_path, data_path)}")
        print("请确保数据文件存在后再运行此示例")
        return None


def example_training_loop():
    """
    示例6：训练循环
    
    展示如何使用增强版模型进行训练
    """
    print("\n" + "=" * 80)
    print("示例6：训练循环（示例代码）")
    print("=" * 80)
    
    print("""
# 训练循环示例代码

import torch
import torch.nn as nn
from models.EnhancedTimeLLM import EnhancedTimeLLM
from data_provider.data_loader_enhanced import create_enhanced_dataset
from torch.utils.data import DataLoader

# 1. 创建配置
config = EnhancedConfig()

# 2. 创建模型
model = EnhancedTimeLLM(config)
model = model.cuda() if torch.cuda.is_available() else model

# 3. 创建数据集和数据加载器
train_dataset = create_enhanced_dataset(
    root_path='./dataset/数据/',
    data_path='processed_data.csv',
    flag='train',
    size=[96, 48, 96],
    features='M',
    target='people',
    scale=True,
    timeenc=0,
    freq='10min',
    percent=100,
    dataset_type='people_flow'
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    drop_last=True
)

# 4. 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 5. 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch_idx, (seq_x, seq_y, seq_x_mark, seq_y_mark) in enumerate(train_loader):
        # 移动到GPU
        if torch.cuda.is_available():
            seq_x = seq_x.cuda()
            seq_y = seq_y.cuda()
            seq_x_mark = seq_x_mark.cuda()
            seq_y_mark = seq_y_mark.cuda()
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(seq_x, seq_x_mark, seq_y, seq_y_mark)
        
        # 计算损失
        loss = criterion(outputs, seq_y[:, -config.pred_len:, :])
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    """)
    
    print("\n注意：这只是示例代码，实际使用时需要根据具体需求调整")


def main():
    """运行所有示例"""
    print("\n" + "=" * 80)
    print("增强版TimeLLM模型使用示例")
    print("=" * 80)
    print("\n本示例展示如何使用增强版TimeLLM模型，该模型实现了：")
    print("1. 语义增强：构造包含任务背景、历史统计特征和外部因素的完整Prompt")
    print("2. 自适应融合：使用交叉注意力或相似度对齐机制动态学习外部因素影响权重")
    print("3. LLM语义驱动：利用预训练语言模型的语义理解能力")
    print("=" * 80)
    
    # 运行各个示例
    try:
        example_semantic_prompt_construction()
        example_cross_attention_fusion()
        example_similarity_alignment_fusion()
        example_full_model_forward()
        example_data_loading()
        example_training_loop()
        
        print("\n" + "=" * 80)
        print("所有示例运行完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n运行示例时出错: {str(e)}")
        print("某些示例可能需要特定的依赖或数据文件")


if __name__ == '__main__':
    main()

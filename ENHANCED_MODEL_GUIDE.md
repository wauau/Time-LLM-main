# 增强版TimeLLM模型使用指南

## 概述

增强版TimeLLM模型实现了"**LLM语义驱动 + 外因影响自适应建模**"的统一框架，通过以下两个关键改进强化了模型的"参与预测"能力：

### 1. 语义增强（Semantic Enhancement）
- **构造完整Prompt**：包含任务背景、历史统计特征（人流均值、趋势、周期）以及预测目标
- **LLM语义理解**：使用预训练语言模型（BERT/GPT）获得更具语义表达能力的embedding
- **外部因素语义化**：将天气、节假日等外部因素转化为语义描述，而非简单数值

### 2. 自适应融合机制（Adaptive Fusion Mechanism）
- **交叉注意力融合**：动态学习"外因语义特征 → 人流变化"的影响强度
- **相似度对齐融合**：基于相似度计算自适应权重，实现特征对齐
- **加权融合表示**：将自适应权重作用到人流时序特征上，形成fused feature

## 核心组件

### 1. SemanticPromptConstructor
语义Prompt构造器，负责构造包含任务背景、历史统计特征和外部因素的完整Prompt。

**功能**：
- 任务背景描述
- 历史统计特征（min, max, median, trend, lags）
- 外部因素语义描述（天气、节假日、时间特征）
- 预测目标说明

**使用示例**：
```python
from models.EnhancedTimeLLM import SemanticPromptConstructor

# 创建Prompt构造器
prompt_constructor = SemanticPromptConstructor(
    task_description="People flow forecasting",
    pred_len=96,
    seq_len=96
)

# 构造Prompt
prompts = prompt_constructor.construct_prompt(batch_stats, external_factors)
```

### 2. CrossAttentionFusion
交叉注意力融合模块，动态学习外部因素对时序特征的影响权重。

**功能**：
- Query来自时序特征
- Key和Value来自外部因素语义特征
- 计算注意力权重
- 门控融合机制

**使用示例**：
```python
from models.EnhancedTimeLLM import CrossAttentionFusion

# 创建融合模块
fusion_module = CrossAttentionFusion(d_model=512, n_heads=8, d_ff=512)

# 执行融合
fused_features, attention_weights = fusion_module(temporal_features, semantic_features)
```

### 3. SimilarityAlignmentFusion
相似度对齐融合模块，基于相似度计算自适应权重。

**功能**：
- 计算时序特征与语义特征的相似度
- 自适应权重生成
- 加权特征融合

**使用示例**：
```python
from models.EnhancedTimeLLM import SimilarityAlignmentFusion

# 创建融合模块
fusion_module = SimilarityAlignmentFusion(d_model=512)

# 执行融合
fused_features, similarity_weights = fusion_module(temporal_features, semantic_features)
```

### 4. EnhancedTimeLLM
增强版TimeLLM模型，整合了语义增强和自适应融合机制。

**核心特性**：
- 语义Prompt构造
- LLM语义理解
- 自适应融合（交叉注意力或相似度对齐）
- 时序重编程

**使用示例**：
```python
from models.EnhancedTimeLLM import EnhancedTimeLLM

# 创建配置
config = EnhancedConfig()

# 创建模型
model = EnhancedTimeLLM(config)

# 前向传播
output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
```

## 数据准备

### 数据格式要求

增强版模型需要包含以下特征的数据：

1. **人流数据**（必需）：`people` 列
2. **天气数据**（推荐）：`temperature`, `humidity`, `wind_speed` 列
3. **节假日数据**（推荐）：`is_holiday` 列
4. **时间戳**（必需）：`datetime` 列

### 数据预处理

使用提供的预处理脚本处理数据：

```bash
python data_preprocess.py
```

这将生成 `processed_data.csv` 文件，包含所有必要的特征。

## 训练模型

### 方法1：使用训练脚本

```bash
python run_enhanced.py --task_name long_term_forecast \\
                       --is_training 1 \\
                       --root_path ./dataset/数据/ \\
                       --data_path processed_data.csv \\
                       --model_id people_flow_96_96 \\
                       --model EnhancedTimeLLM \\
                       --data custom \\
                       --features M \\
                       --seq_len 96 \\
                       --label_len 48 \\
                       --pred_len 96 \\
                       --llm_model BERT \\
                       --llm_layers 6 \\
                       --use_cross_attention \\
                       --train_epochs 10 \\
                       --batch_size 32 \\
                       --learning_rate 0.0001
```

### 方法2：使用Python代码

```python
import torch
from models.EnhancedTimeLLM import EnhancedTimeLLM
from data_provider.data_loader_enhanced import create_enhanced_dataset
from torch.utils.data import DataLoader

# 创建配置
config = EnhancedConfig()

# 创建模型
model = EnhancedTimeLLM(config)
model = model.cuda() if torch.cuda.is_available() else model

# 创建数据集
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

# 训练循环
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(10):
    model.train()
    for seq_x, seq_y, seq_x_mark, seq_y_mark in train_loader:
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
    
    print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')
```

## 配置参数

### 模型参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `task_name` | 任务名称 | `long_term_forecast` |
| `seq_len` | 输入序列长度 | `96` |
| `label_len` | 标签序列长度 | `48` |
| `pred_len` | 预测序列长度 | `96` |
| `d_model` | 模型维度 | `512` |
| `n_heads` | 注意力头数 | `8` |
| `d_ff` | 前馈网络维度 | `512` |
| `dropout` | Dropout率 | `0.1` |

### LLM参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `llm_model` | LLM模型类型 | `BERT` |
| `llm_dim` | LLM嵌入维度 | `768` |
| `llm_layers` | LLM层数 | `6` |
| `prompt_domain` | 是否使用领域特定Prompt | `True` |

### 融合参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `use_cross_attention` | 使用交叉注意力融合 | `True` |
| `use_similarity_alignment` | 使用相似度对齐融合 | `False` |

## 融合机制选择

### 交叉注意力融合（Cross-Attention Fusion）

**优点**：
- 能够捕捉时序特征与语义特征之间的复杂关系
- 注意力权重可视化，便于解释
- 适合语义特征与时间步对应的情况

**适用场景**：
- 外部因素与时间步有明确对应关系
- 需要理解外部因素对不同时间步的影响差异

**启用方式**：
```bash
python run_enhanced.py --use_cross_attention
```

### 相似度对齐融合（Similarity Alignment Fusion）

**优点**：
- 计算效率高
- 基于相似度的自适应权重
- 适合特征长度相同的情况

**适用场景**：
- 时序特征和语义特征长度相同
- 需要快速融合
- 关注特征相似性

**启用方式**：
```bash
python run_enhanced.py --use_similarity_alignment
```

## 示例代码

运行示例代码了解模型使用：

```bash
python example_enhanced_model.py
```

示例包括：
1. 语义Prompt构造
2. 交叉注意力融合
3. 相似度对齐融合
4. 完整模型前向传播
5. 数据加载
6. 训练循环

## 性能优化建议

### 1. 数据预处理
- 确保数据质量，处理缺失值和异常值
- 合理归一化，避免数值范围差异过大
- 提取丰富的时间特征

### 2. 模型配置
- 根据数据规模调整模型大小
- 使用合适的序列长度（seq_len, pred_len）
- 调整LLM层数以平衡性能和效率

### 3. 训练策略
- 使用学习率调度器
- 早停策略防止过拟合
- 梯度裁剪防止梯度爆炸

### 4. 融合机制选择
- 根据数据特点选择合适的融合机制
- 可以尝试两种融合机制，选择效果更好的

## 常见问题

### Q1: 如何添加新的外部因素？

A: 在数据中添加新列，然后在`EnhancedTimeLLM.extract_external_factors`方法中提取该因素。

### Q2: 如何使用其他LLM模型？

A: 在`EnhancedTimeLLM._init_llm`方法中添加对其他LLM模型的支持。

### Q3: 如何调整融合机制的权重？

A: 修改`CrossAttentionFusion`或`SimilarityAlignmentFusion`中的网络结构。

### Q4: 如何可视化注意力权重？

A: 在训练过程中保存attention_weights，然后使用matplotlib等工具可视化。

## 文件结构

```
Time-LLM-main/
├── models/
│   ├── EnhancedTimeLLM.py          # 增强版模型实现
│   └── TimeLLM.py                  # 原始模型
├── data_provider/
│   ├── data_loader_enhanced.py     # 增强版数据加载器
│   └── data_loader.py              # 原始数据加载器
├── run_enhanced.py                 # 增强版训练脚本
├── example_enhanced_model.py       # 使用示例
├── data_preprocess.py              # 数据预处理
└── ENHANCED_MODEL_GUIDE.md         # 本文档
```

## 引用

如果您使用了增强版TimeLLM模型，请引用：

```bibtex
@article{timellm2023,
  title={Time-LLM: Time Series Forecasting with Large Language Models},
  author={...},
  journal={...},
  year={2023}
}
```

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送邮件

## 许可证

本项目遵循原Time-LLM项目的许可证。

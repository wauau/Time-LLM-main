# Time-LLM 人流量预测系统

## 项目简介

Time-LLM 是一个基于大语言模型的时间序列重编程预测系统，专门用于人流量预测。本项目实现了数据处理、模型训练、预测和可视化等功能，为商圈人流量预测提供了完整的解决方案。

## 项目结构

```
Time-LLM-main/
├── data_provider/           # 数据处理模块
│   ├── data_processor.py    # 数据处理核心功能
│   └── __init__.py          # 模块导出
├── dataset/                 # 数据集
│   ├── 数据/                # 原始数据和处理后数据
│   └── prompt_bank/         # 提示词库
├── models/                  # 模型实现
│   ├── Autoformer.py        # Autoformer模型
│   ├── DLinear.py           # DLinear模型
│   └── TimeLLM.py           # Time-LLM模型
├── results/                 # 模型评估结果
├── app.py                   # 可视化与交互平台
├── data_preprocess.py       # 数据预处理脚本
├── model_comparison.py      # 模型对比分析脚本
├── arima.py                 # ARIMA模型实现
├── lstm.py                  # LSTM模型实现
└── README_zh.md             # 中文使用说明
```

## 环境要求

- Python 3.11+
- 主要依赖：
  - torch==2.2.2
  - pandas==1.5.3
  - numpy==1.23.5
  - matplotlib==3.7.0
  - streamlit==1.32.0
  - scikit-learn==1.2.2
  - statsmodels==0.14.0
  - plotly==5.18.0
  - tabulate==0.10.0

## 安装指南

1. 克隆项目到本地：
   ```bash
   git clone <项目地址>
   cd Time-LLM-main
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 使用说明

### 1. 数据预处理

运行数据预处理脚本，对原始数据进行处理：

```bash
python data_preprocess.py
```

参数说明：
- `--input_people`：人流量数据文件路径（默认：./dataset/数据/people_10_12_new.csv）
- `--input_weather`：天气数据文件路径（默认：./dataset/数据/weather_10_12_10min_utf8.csv）
- `--input_events`：节假日数据文件路径（默认：./dataset/数据/events_10_12_utf8.csv）
- `--output`：处理后的数据保存路径（默认：./dataset/数据/processed_data.csv）
- `--freq`：数据对齐的时间频率（默认：10min）
- `--denoise_method`：去噪方法（默认：moving_average）

### 2. 模型训练与评估

运行模型对比分析脚本，训练并评估不同模型：

```bash
python model_comparison.py
```

该脚本会：
- 训练ARIMA、LSTM和Time-LLM三种模型
- 计算评估指标（MAE、RMSE、MAPE、R²）
- 生成对比报告和可视化结果

### 3. 可视化与交互平台

启动可视化平台：

```bash
streamlit run app.py
```

在浏览器中访问：http://localhost:8501

平台功能：
- **数据可视化**：展示人流量趋势、热力图等
- **模型预测**：使用不同模型进行预测
- **模型评估**：查看模型性能对比
- **系统配置**：设置数据路径等参数

## 核心功能

### 1. 数据处理

- **时间特征提取**：提取年、月、日、小时、分钟、星期几、是否周末、季节、是否高峰期等特征
- **节假日特征提取**：从节假日数据中提取节假日信息
- **数据对齐**：将人流量和天气数据对齐到统一的时间频率
- **数据去噪**：使用移动平均法对人流量数据进行去噪

### 2. 模型实现

- **ARIMA**：传统时间序列模型，适合线性数据
- **LSTM**：深度学习模型，适合捕捉时间依赖关系
- **Time-LLM**：基于大语言模型的方法，能够利用更多上下文信息

### 3. 模型评估

- 计算多种评估指标：MAE、RMSE、MAPE、R²
- 生成详细的对比报告和可视化结果
- 支持模型参数调优

### 4. 可视化平台

- 响应式界面设计
- 多种数据可视化方式
- 实时模型预测
- 模型性能对比

## 示例代码

### 数据预处理

```python
from data_provider import complete_data_processing

# 执行完整的数据处理流程
processed_data = complete_data_processing()
print(f"处理后的数据形状: {processed_data.shape}")
```

### 模型预测

```python
from model_comparison import run_arima_prediction, run_lstm_prediction, run_time_llm_prediction
import numpy as np

# 加载数据
data = np.loadtxt('./dataset/数据/processed_data.csv', delimiter=',', skiprows=1)
values = data[:, 1]  # 假设第二列是人流量数据

# 预测未来24小时
prediction_steps = 24

# 使用ARIMA模型预测
arima_pred = run_arima_prediction(values, prediction_steps, (5, 0, 2))

# 使用LSTM模型预测
lstm_pred = run_lstm_prediction(values, prediction_steps, hidden_size=64, epochs=20)

# 使用Time-LLM模型预测
time_llm_pred = run_time_llm_prediction(values, prediction_steps)
```

## 常见问题

### 1. 数据预处理失败

- 检查数据文件路径是否正确
- 确保数据文件格式正确
- 检查数据文件的列名是否与代码中一致

### 2. 模型训练时间过长

- 减少训练轮数
- 减小数据集大小
- 调整模型参数

### 3. 可视化平台无法启动

- 确保安装了streamlit
- 检查端口是否被占用
- 尝试使用不同的浏览器

## 未来改进

1. **模型优化**：
   - 实现完整的Time-LLM模型
   - 添加更多模型（如Prophet、XGBoost等）
   - 实现模型融合

2. **功能扩展**：
   - 添加多数据源支持
   - 实现自动超参数搜索
   - 添加预测结果导出功能

3. **性能优化**：
   - 优化模型训练速度
   - 改进数据处理效率
   - 优化可视化平台响应速度

## 许可证

本项目采用MIT许可证。

## 联系方式

如有问题或建议，请联系项目维护者。

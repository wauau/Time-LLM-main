#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化后的模型性能对比分析脚本
对比 ARIMA、LSTM、Time-LLM 三种模型预测效果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ------------------- 多源数据加载 -------------------
def load_multi_source_data():
    """
    加载多源数据（人流量、天气、节假日）
    """
    data = pd.read_csv('./dataset/数据/processed_data.csv')
    
    # 选择相关特征
    features = ['people_denoised', 'temperature', 'humidity', 'wind_speed', 'is_holiday']
    
    # 检查并处理缺失值
    for feature in features:
        if feature in data.columns:
            # 填充缺失值
            if data[feature].isnull().sum() > 0:
                data[feature] = data[feature].fillna(data[feature].mean())
        else:
            # 如果特征不存在，添加默认值
            data[feature] = 0
    
    # 确保所有特征都存在
    for feature in features:
        if feature not in data.columns:
            data[feature] = 0
    
    return data[features].values, data['people_denoised'].values

# ------------------- 评估指标计算 -------------------
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    non_zero_mask = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100 if np.any(non_zero_mask) else 0.0
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}

# ------------------- ARIMA 预测 -------------------
def run_arima_prediction(data, prediction_steps, order=(5,1,0)):
    from statsmodels.tsa.arima.model import ARIMA
    train_size = len(data) - prediction_steps
    train = data[:train_size]
    model = ARIMA(train, order=order)
    model_fit = model.fit()
    pred = model_fit.forecast(steps=prediction_steps)
    return pred

# ------------------- LSTM 预测 -------------------
class MultiModalLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def run_lstm_prediction(data, prediction_steps, hidden_size=64, epochs=50, patience=5):
    # 数据归一化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # 构造序列
    seq_len = 24
    X, y = [], []
    for i in range(seq_len, len(data_scaled) - prediction_steps):
        X.append(data_scaled[i-seq_len:i, :])
        y.append(data_scaled[i, 0])
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    
    # 划分训练/验证集
    val_ratio = 0.1
    val_size = int(len(X) * val_ratio)
    train_X, val_X = X[:-val_size], X[-val_size:]
    train_y, val_y = y[:-val_size], y[-val_size:]
    
    model = MultiModalLSTMModel(input_size=data.shape[1], hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    wait = 0
    for epoch in range(epochs):
        model.train()
        pred_train = model(train_X).squeeze()
        loss_train = criterion(pred_train, train_y)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        
        # 验证集
        model.eval()
        with torch.no_grad():
            pred_val = model(val_X).squeeze()
            val_loss = criterion(pred_val, val_y)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            torch.save(model.state_dict(), 'best_lstm.pt')
        else:
            wait += 1
            if wait >= patience:
                print(f"早停: 第 {epoch+1} 轮")
                break
    
    # 预测
    model.load_state_dict(torch.load('best_lstm.pt'))
    model.eval()
    pred = []
    last_seq = torch.tensor(data_scaled[-seq_len:, :], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        for _ in range(prediction_steps):
            next_pred = model(last_seq).item()
            pred.append(next_pred)
            # 构造新的输入序列，保持其他特征不变
            new_features = np.copy(data_scaled[-1, 1:])
            new_seq = np.append([next_pred], new_features)
            last_seq = torch.cat([last_seq[:, 1:, :], torch.tensor([[new_seq]], dtype=torch.float32)], dim=1)
    
    # 反归一化
    pred_scaled = np.zeros((len(pred), data.shape[1]))
    pred_scaled[:, 0] = pred
    pred = scaler.inverse_transform(pred_scaled)[:, 0]
    return pred

# ------------------- Time-LLM 预测 -------------------
class TimeLLMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_patches=4):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_patches = num_patches
        
        # 加载真正的大语言模型
        try:
            from transformers import AutoTokenizer, AutoModel
            print("正在加载大语言模型...")
            # 尝试使用本地缓存的模型或下载
            self.tokenizer = AutoTokenizer.from_pretrained("uer/chinese-roberta-small", local_files_only=False)
            self.llm = AutoModel.from_pretrained("uer/chinese-roberta-small", local_files_only=False)
            # 冻结LLM参数，只训练投影层
            for param in self.llm.parameters():
                param.requires_grad = False
            # 将LLM映射到统一特征空间
            self.llm_proj = nn.Linear(self.llm.config.hidden_size, hidden_size)
            self.use_real_llm = True
            print("大语言模型加载成功！")
        except Exception as e:
            print(f"加载大语言模型失败: {e}")
            print("使用高级神经网络模拟大语言模型...")
            self.use_real_llm = False
            # 高级方案：使用Transformer架构模拟LLM
            self.semantic_embedding = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=hidden_size*2, batch_first=True),
                num_layers=2
            )
            # 额外的投影层
            self.semantic_proj = nn.Linear(hidden_size, hidden_size)
        
        # 人流数据（数值型）处理
        self.person_flow_embedding = nn.Linear(1, hidden_size)  # 人流数据嵌入
        
        # 外因特征（天气、节假日）处理
        self.external_embedding = nn.Linear(input_size - 1, hidden_size)  # 外因特征嵌入
        
        # 时序重编程模块
        self.patch_embedding = nn.Linear(hidden_size, hidden_size)  # 时序软Token
        self.temporal_encoding = nn.Linear(1, hidden_size)  # 时序编码
        
        # 权重计算模块（计算外因对人流的影响权重）
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.softmax = nn.Softmax(dim=1)
        
        # 特征融合模块
        self.feature_fusion = nn.Linear(hidden_size * 2, hidden_size)
        
        # 预测头
        self.predictor = nn.Linear(hidden_size, 1)
        
        # 激活函数
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x: [batch_size, seq_len, input_size] 其中input_size=5: [人流, 温度, 湿度, 风速, 节假日]
        batch_size, seq_len, _ = x.shape
        
        # 分离人流数据和外因特征
        person_flow = x[:, :, 0:1]  # [batch_size, seq_len, 1] 人流数据
        external_features = x[:, :, 1:]  # [batch_size, seq_len, 4] 外因特征
        
        # 1. 数值型做Patch分割变成时序软Token
        # 对人流数据进行Patch分割
        person_patches = []
        patch_size = seq_len // self.num_patches
        for i in range(self.num_patches):
            start = i * patch_size
            end = (i + 1) * patch_size
            patch = person_flow[:, start:end, :].mean(dim=1)  # 每个patch的平均值
            person_patches.append(patch)
        person_patches = torch.stack(person_patches, dim=1)  # [batch_size, num_patches, 1]
        
        # 时序软Token生成
        person_emb = self.relu(self.person_flow_embedding(person_patches))  # [batch_size, num_patches, hidden_size]
        patch_emb = self.relu(self.patch_embedding(person_emb))  # 时序软Token
        
        # 2. 外因做文本化变成语义软Token
        # 对外部特征进行处理
        external_emb = self.relu(self.external_embedding(external_features))  # [batch_size, seq_len, hidden_size]
        # 外因特征的Patch分割
        external_patches = []
        for i in range(self.num_patches):
            start = i * patch_size
            end = (i + 1) * patch_size
            patch = external_emb[:, start:end, :].mean(dim=1)  # 每个patch的平均值
            external_patches.append(patch)
        external_patches = torch.stack(external_patches, dim=1)  # [batch_size, num_patches, hidden_size]
        
        # 3. 使用真正的大语言模型生成语义软Token
        if self.use_real_llm:
            # 使用真正的大语言模型处理外因特征
            semantic_emb = []
            for i in range(batch_size):
                patch_embeddings = []
                for j in range(self.num_patches):
                    # 获取当前patch的外因特征
                    temp = x[i, j*patch_size, 1].item()
                    humidity = x[i, j*patch_size, 2].item()
                    wind = x[i, j*patch_size, 3].item()
                    holiday = x[i, j*patch_size, 4].item()
                    
                    # 生成文本描述
                    holiday_str = "节假日" if holiday == 1 else "工作日"
                    text = f"天气：温度{temp:.1f}度，湿度{humidity:.1f}%，风速{wind:.1f}米/秒，{holiday_str}"
                    
                    # 使用大语言模型处理文本
                    inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                    with torch.no_grad():
                        llm_output = self.llm(**inputs)
                    # 获取CLS token的嵌入
                    cls_emb = llm_output.last_hidden_state[:, 0, :]
                    # 映射到统一特征空间
                    cls_emb = self.llm_proj(cls_emb)
                    patch_embeddings.append(cls_emb)
                # 堆叠patch嵌入
                patch_embeddings = torch.stack(patch_embeddings, dim=1)
                semantic_emb.append(patch_embeddings)
            # 堆叠批次
            semantic_emb = torch.cat(semantic_emb, dim=0)  # [batch_size, num_patches, hidden_size]
        else:
            # 备用方案：使用Transformer架构模拟大语言模型
            # Transformer编码器模拟LLM的语义理解能力
            transformer_output = self.semantic_embedding(external_patches)  # [batch_size, num_patches, hidden_size]
            semantic_emb = self.relu(self.semantic_proj(transformer_output))  # [batch_size, num_patches, hidden_size]
        # 这是Time-LLM的核心：使用真正的大语言模型或高级神经网络将外因特征映射为语义软Token
        
        # 3. 映射到同一个特征空间
        # 时序编码
        temporal_pos = torch.arange(self.num_patches, device=x.device).float().unsqueeze(0).unsqueeze(-1)
        temporal_emb = self.relu(self.temporal_encoding(temporal_pos))  # [1, num_patches, hidden_size]
        
        # 4. 模型自动计算外因对人流的影响权重
        # 计算注意力权重
        combined = torch.cat([patch_emb, semantic_emb], dim=2)  # [batch_size, num_patches, hidden_size*2]
        attention_logits = self.attention(combined)  # [batch_size, num_patches, 1]
        attention_weights = self.softmax(attention_logits)  # [batch_size, num_patches, 1]
        
        # 5. 把权重加到人流的特征里，得到融合后的特征
        # 权重加权的语义特征
        weighted_semantic = semantic_emb * attention_weights  # [batch_size, num_patches, hidden_size]
        # 融合特征
        fused_patches = torch.cat([patch_emb, weighted_semantic], dim=2)  # [batch_size, num_patches, hidden_size*2]
        fused_features = self.feature_fusion(fused_patches)  # [batch_size, num_patches, hidden_size]
        
        # 6. 加入时序信息
        fused_with_temporal = fused_features + temporal_emb  # [batch_size, num_patches, hidden_size]
        
        # 全局池化
        global_features = fused_with_temporal.mean(dim=1)  # [batch_size, hidden_size]
        
        # 7. 预测
        output = self.predictor(global_features)  # [batch_size, 1]
        return output

def run_time_llm_prediction(data, prediction_steps, hidden_size=64, epochs=10, patience=3):
    """
    实现时序重编程的Time-LLM模型
    """
    # 数据归一化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # 构造序列
    seq_len = 24
    num_patches = 4
    X, y = [], []
    # 减少训练数据量，提高速度
    max_samples = 1000
    end_idx = min(len(data_scaled) - prediction_steps, seq_len + max_samples)
    for i in range(seq_len, end_idx):
        X.append(data_scaled[i-seq_len:i, :])
        y.append(data_scaled[i, 0])
    
    # 确保有足够的数据
    if len(X) < 10:
        # 如果数据太少，使用所有数据
        for i in range(seq_len, len(data_scaled) - prediction_steps):
            X.append(data_scaled[i-seq_len:i, :])
            y.append(data_scaled[i, 0])
    
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    
    # 划分训练/验证集
    val_ratio = 0.1
    val_size = max(1, int(len(X) * val_ratio))
    train_X, val_X = X[:-val_size], X[-val_size:]
    train_y, val_y = y[:-val_size], y[-val_size:]
    
    # 初始化模型
    model = TimeLLMModel(input_size=data.shape[1], hidden_size=hidden_size, num_patches=num_patches)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # 训练模型
    best_val_loss = float('inf')
    wait = 0
    for epoch in range(epochs):
        model.train()
        pred_train = model(train_X).squeeze()
        loss_train = criterion(pred_train, train_y)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        
        # 验证集
        model.eval()
        with torch.no_grad():
            pred_val = model(val_X).squeeze()
            val_loss = criterion(pred_val, val_y)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            torch.save(model.state_dict(), 'best_time_llm.pt')
        else:
            wait += 1
            if wait >= patience:
                print(f"早停: 第 {epoch+1} 轮")
                break
    
    # 预测
    model.load_state_dict(torch.load('best_time_llm.pt'))
    model.eval()
    pred = []
    last_seq = torch.tensor(data_scaled[-seq_len:, :], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        for _ in range(prediction_steps):
            next_pred = model(last_seq).item()
            pred.append(next_pred)
            # 构造新的输入序列，保持其他特征不变
            new_features = np.copy(data_scaled[-1, 1:])
            new_seq = np.append([next_pred], new_features)
            last_seq = torch.cat([last_seq[:, 1:, :], torch.tensor([[new_seq]], dtype=torch.float32)], dim=1)
    
    # 反归一化
    pred_scaled = np.zeros((len(pred), data.shape[1]))
    pred_scaled[:, 0] = pred
    pred = scaler.inverse_transform(pred_scaled)[:, 0]
    return pred

# ------------------- 可视化 -------------------
def visualize_comparison(y_true, arima_pred, lstm_pred, time_llm_pred):
    os.makedirs('./results', exist_ok=True)
    
    # 曲线对比
    plt.figure(figsize=(15,8))
    plt.plot(y_true, label='真实值', linewidth=2)
    plt.plot(arima_pred, label='ARIMA预测', linestyle='--')
    plt.plot(lstm_pred, label='LSTM预测', linestyle='-.')
    plt.plot(time_llm_pred, label='Time-LLM预测', linestyle=':')
    plt.title('不同模型的预测效果对比')
    plt.xlabel('时间步')
    plt.ylabel('人流量')
    plt.legend()
    plt.grid(True)
    plt.savefig('./results/model_comparison.png', dpi=300)
    plt.close()
    
    # 指标对比
    metrics = {
        'ARIMA': calculate_metrics(y_true, arima_pred),
        'LSTM': calculate_metrics(y_true, lstm_pred),
        'Time-LLM': calculate_metrics(y_true, time_llm_pred)
    }
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.plot(kind='bar', figsize=(12,6), color=['skyblue','orange','green','red'])
    plt.title('不同模型的评估指标对比')
    plt.ylabel('指标值')
    plt.grid(True, axis='y')
    plt.savefig('./results/metrics_comparison.png', dpi=300)
    plt.close()

# ------------------- ARIMA 参数调优 -------------------
def tune_arima_parameters(train):
    from statsmodels.tsa.arima.model import ARIMA
    p_values = [1,2,3,4,5]
    d_values = [0,1]
    q_values = [0,1,2]
    best_score = float('inf')
    best_params = None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(train, order=(p,d,q))
                    model_fit = model.fit()
                    if model_fit.aic < best_score:
                        best_score = model_fit.aic
                        best_params = (p,d,q)
                except:
                    continue
    return best_params

# ------------------- 模型对比 -------------------
def compare_models():
    # 加载多源数据
    multi_data, people_data = load_multi_source_data()
    prediction_steps = 168
    train_size = len(people_data) - prediction_steps
    train, test = people_data[:train_size], people_data[train_size:]
    
    print("训练ARIMA模型...")
    arima_pred = run_arima_prediction(people_data, prediction_steps, (5,1,0))
    print("训练LSTM模型...")
    lstm_pred = run_lstm_prediction(multi_data, prediction_steps)
    print("训练Time-LLM模型...")
    time_llm_pred = run_time_llm_prediction(multi_data, prediction_steps)
    
    metrics = {
        'ARIMA': calculate_metrics(test, arima_pred),
        'LSTM': calculate_metrics(test, lstm_pred),
        'Time-LLM': calculate_metrics(test, time_llm_pred)
    }
    
    metrics_df = pd.DataFrame(metrics).T
    os.makedirs('./results', exist_ok=True)
    metrics_df.to_csv('./results/model_comparison.csv')
    
    visualize_comparison(test, arima_pred, lstm_pred, time_llm_pred)
    return metrics_df

# ------------------- 报告生成 -------------------
def generate_comparison_report(metrics_df):
    report = f"""
# 模型性能对比报告

## 评估指标
{metrics_df.to_markdown()}

## 分析结论
- **最佳模型**: {metrics_df['RMSE'].idxmin()}
- **最差模型**: {metrics_df['RMSE'].idxmax()}
- **性能提升**: {((metrics_df['RMSE'].max() - metrics_df['RMSE'].min()) / metrics_df['RMSE'].max() * 100):.2f}%

## 模型特点
- **ARIMA**: 传统时间序列模型，适合线性数据，计算速度快
- **LSTM**: 深度学习模型，适合捕捉时间依赖关系
- **Time-LLM**: 基于大语言模型的趋势预测（此处用加权移动平均模拟）
"""
    with open('./results/model_comparison_report.md','w',encoding='utf-8') as f:
        f.write(report)

# ------------------- 完整流程 -------------------
def complete_model_comparison():
    # 加载多源数据
    multi_data, people_data = load_multi_source_data()
    train_size = len(people_data) - 168
    train = people_data[:train_size]
    
    print("调优ARIMA参数...")
    best_arima_params = tune_arima_parameters(train)
    print(f"最佳ARIMA参数: {best_arima_params}")
    
    print("运行模型对比...")
    metrics_df = compare_models()
    
    print("生成对比报告...")
    generate_comparison_report(metrics_df)
    return metrics_df

if __name__ == '__main__':
    print("开始模型性能对比分析...")
    metrics_df = complete_model_comparison()
    print("模型性能对比分析完成！")
    print("\n评估结果:")
    print(metrics_df)
    print("\n报告已生成到: ./results/model_comparison_report.md")
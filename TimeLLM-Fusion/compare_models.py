import torch
import numpy as np
from config import Config
from models.full_model import FullModel
from models.comparison_models import BaselineModel, ConcatModel, LLMOnlyModel
from data.dataset import generate_data
from evaluation import evaluate
from visualization import plot_comparison, plot_prediction


config = Config()

# 创建所有模型
model_base = BaselineModel(config)
model_concat = ConcatModel(config)
model_llm = LLMOnlyModel(config)
model_full = FullModel(config)

models = {
    "baseline": model_base,
    "concat": model_concat,
    "llm_only": model_llm,
    "llm_fusion": model_full
}

# 生成测试数据
def generate_test_data(batch_size=100, seq_len=24):
    x, ext, y = generate_data(batch_size, seq_len)
    return x, ext, y

# 评估所有模型
results = {}

for name, model in models.items():
    print(f"Evaluating {name}...")
    
    # 生成测试数据
    x, ext, y = generate_test_data()
    
    # 前向传播
    pred, _ = model(x, ext)
    
    # 转换为numpy
    y_true = y.detach().cpu().numpy()
    y_pred = pred.detach().cpu().numpy()
    
    # 评估
    mae, rmse, r2 = evaluate(y_true, y_pred)
    results[name] = (mae, rmse, r2)
    
    print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")
    
    # 保存预测对比图
    if name == "llm_fusion":
        plot_prediction(y_true[0], y_pred[0], f"{name}_prediction.png")

# 可视化比较结果
plot_comparison(results)

print("\nComparison completed!")
print("Results:")
for name, (mae, rmse, r2) in results.items():
    print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")
import torch
import torch.nn as nn
from config import Config
from models.full_model import FullModel
from data.dataset import generate_data
from visualization import plot_attention_heatmap, plot_weight_curve

config = Config()
model = FullModel(config)

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
loss_fn = nn.MSELoss()

for epoch in range(config.epochs):
    x, ext, y = generate_data(config.batch_size, config.seq_len)

    pred, alpha = model(x, ext)

    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")
    print("Sample weights:", alpha[0])
    
    # 每2个epoch保存一次可视化
    if epoch % 2 == 0:
        # 生成权重随时间变化图
        # 为了可视化，我们重复权重以匹配时间步长
        weights_time = alpha.unsqueeze(1).repeat(1, config.seq_len, 1)[0]
        plot_weight_curve(weights_time, f"weight_curve_epoch_{epoch}.png")
        
        # 生成注意力热力图
        # 为了可视化，我们创建一个模拟的注意力矩阵
        # 实际应用中，这应该是模型中真实的注意力权重
        attention_sim = torch.randn(config.seq_len, config.d_model)
        plot_attention_heatmap(attention_sim, f"attention_heatmap_epoch_{epoch}.png")
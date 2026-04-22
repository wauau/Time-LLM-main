import torch
import torch.nn as nn
from config import Config
from models.full_model import FullModel
from data.dataset import generate_data

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
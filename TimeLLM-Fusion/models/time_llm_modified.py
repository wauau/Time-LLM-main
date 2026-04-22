import torch
import torch.nn as nn

class TimeLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Linear(1, config.d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config.d_model, nhead=4),
            num_layers=2
        )
        self.head = nn.Linear(config.d_model, 1)

    def encode(self, x):
        x = self.embed(x)
        return self.encoder(x)

    def forward_from_embedding(self, x):
        return self.head(x)
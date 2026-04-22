import torch

def generate_data(batch_size=8, seq_len=24):
    x = torch.randn(batch_size, seq_len, 1)

    external = []
    for i in range(batch_size):
        external.append({
            "date": "2024-05-01",
            "weather": "rainy",
            "holiday": "Labor Day"
        })

    y = torch.randn(batch_size, seq_len, 1)
    return x, external, y
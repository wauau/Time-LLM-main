import matplotlib.pyplot as plt
import numpy as np


def plot_attention_heatmap(alpha, save_path="attention_heatmap.png"):
    """
    alpha: [B, T, D] 或 [T, D]
    """

    if len(alpha.shape) == 3:
        alpha = alpha[0]  # 取一个batch

    alpha = alpha.detach().cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.imshow(alpha, aspect='auto')
    plt.colorbar()

    plt.xlabel("Feature Dimension")
    plt.ylabel("Time Step")
    plt.title("External Factor Attention Heatmap")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_weight_curve(weights, save_path="weight_curve.png"):
    """
    weights: [T, 2]  (weather, holiday)
    """

    weights = weights.detach().cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.plot(weights[:, 0], label="Weather Impact")
    plt.plot(weights[:, 1], label="Holiday Impact")

    plt.legend()
    plt.xlabel("Time Step")
    plt.ylabel("Weight")
    plt.title("LLM Inferred External Impact Over Time")

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_comparison(results):
    names = list(results.keys())
    mae = [results[n][0] for n in names]

    plt.figure(figsize=(8, 5))
    plt.bar(names, mae)
    plt.title("Model Comparison (MAE)")
    plt.ylabel("MAE")

    for i, v in enumerate(mae):
        plt.text(i, v, f"{v:.3f}", ha='center')

    plt.tight_layout()
    plt.savefig("comparison.png")
    plt.close()


def plot_prediction(y_true, y_pred, save_path="prediction.png"):
    plt.figure(figsize=(10, 5))

    plt.plot(y_true[:100], label="True")
    plt.plot(y_pred[:100], label="Pred")

    plt.legend()
    plt.title("Prediction Comparison")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
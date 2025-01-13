import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def set_random_seed(seed_value=42, use_cuda=True):

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def compute_metrics(y_true, y_pred):

    # for 2D (batch, n_outputs) output
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    mse_val = mean_squared_error(y_true_flat, y_pred_flat)
    rmse_val = np.sqrt(mse_val)
    mae_val = mean_absolute_error(y_true_flat, y_pred_flat)
    r2_val = r2_score(y_true_flat, y_pred_flat)

    return {
        "MSE": mse_val,
        "RMSE": rmse_val,
        "MAE": mae_val,
        "R2": r2_val
    }


def plot_loss_curve(train_losses, test_losses=None, title="Loss Curve"):

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    if test_losses is not None:
        plt.plot(test_losses, label="Test Loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_predictions(time_data, y_true, y_pred, title="Prediction vs. Actual"):

    plt.figure(figsize=(10, 5))
    plt.plot(time_data, y_true, label="Actual")
    plt.plot(time_data, y_pred, label="Predicted")
    plt.ylim(3000, 6000)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.show()


def save_model(model, save_dir="saved_models", filename="model.pt"):

    try:

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        torch.save(
            model.state_dict(),
            save_path,
            _use_new_zipfile_serialization=True,
            weights_only=True
        )

        print(f"Model successfully saved to: {save_path}")
        return save_path

    except Exception as e:
        print(f"Error saving model to {save_path}: {e}")
        raise


def load_model(model, ckpt_path):

    if not os.path.exists(ckpt_path):
        print(f"[Warning] Checkpoint not found: {ckpt_path}")
        return model
    model.load_state_dict(torch.load(ckpt_path))
    return model


def plot_feature_target_relationships(df, time_col, feature_cols, target_col, n_cols=3):

    import matplotlib.pyplot as plt
    import math

    # Calculate number of rows needed
    n_features = len(feature_cols)
    n_rows = math.ceil(n_features / n_cols)

    # Create figure and subplots
    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))





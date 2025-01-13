import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from data.data_loader import load_data_sliding
from model import TimeTransformerLSTM
from utilities import set_random_seed, compute_metrics, plot_loss_curve, plot_predictions, save_model
from explanation import compute_vanilla_gradients
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR

def main():


    # 1.Basic configuration
    SEED = 42
    set_random_seed(SEED)  # set random seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)


    # 2.Data configuration
    file_path = "data/enhanced raw data.xlsx"
    time_col = "Date"
    feature_cols = [
        'EFFR', 'CPI', 'GEU', 'HY Index', 'HPI',
        'PPI', '3m T-Yield', '10y T-Yield', 'UR', 'DTI',
        'Imports', 'GDP', 'Crude Oil', 'BTC', 'Party',
        'year', 'year_offset', 'month'
    ]
    target_cols = ["S&P 500 Price"]

    window_size = 8
    horizon = 1
    train_ratio = 0.8
    val_ratio = 0.1
    normalization = "minmax"


    # 3.Load data
    (X_train, y_train,
     X_val, y_val,
     X_test, y_test,
     scaler_X, scaler_y,
     val_time, test_time) = load_data_sliding(
        file_path=file_path,
        time_col=time_col,
        feature_cols=feature_cols,
        target_cols=target_cols,
        window_size=window_size,
        horizon=horizon,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        normalization=normalization
    )
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # Convert to torch.Tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # DataLoader
    batch_size = 8
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # 4. Set model
    feature_dim = X_train.shape[-1]
    d_model = 32
    nhead = 2
    num_encoder_layers = 1
    dim_feedforward = 128
    dropout = 0.2
    lstm_hidden_dim = 512
    lstm_num_layers = 2

    model = TimeTransformerLSTM(
        feature_dim=feature_dim,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        lstm_hidden_dim=lstm_hidden_dim,
        lstm_num_layers=lstm_num_layers,
        horizon=horizon
    ).to(device)

    print(model)


    # 5. Set training hyperparameters
    num_epochs = 100
    base_lr = 3.2e-4
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    scheduler = StepLR(optimizer, step_size=8, gamma=0.002)

    # Early stop
    patience = 20
    best_val_loss = float('inf')
    patience_counter = 0


    # 6. Training, validation and testing
    train_losses = []
    val_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        # training
        model.train()
        running_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * batch_x.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        # validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                out = model(batch_x)
                loss_val = criterion(out, batch_y)
                running_val_loss += loss_val.item() * batch_x.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)

        # testing
        running_test_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                out = model(batch_x)
                loss_test = criterion(out, batch_y)
                running_test_loss += loss_test.item() * batch_x.size(0)

        epoch_test_loss = running_test_loss / len(test_loader.dataset)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        test_losses.append(epoch_test_loss)

        # checking for early stop
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            try:
                save_model(model, filename="best_model.pt")
                print(f"Saved best model with validation loss: {epoch_val_loss:.6f}")
            except Exception as e:
                print(f"Error saving model: {e}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

        # updating learning rate
        scheduler.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"Train Loss: {epoch_train_loss:.6f}, "
              f"Val Loss: {epoch_val_loss:.6f}, "
              f"Test Loss: {epoch_test_loss:.6f}, ")

    # 7. Loss curve visualization
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training, Validation and Test Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


    # 8. Load best model and evaluate on Test Set
    try:
        # Check if best model file exists
        if os.path.exists("best_model.pt"):
            # Add weights_only=True to address the FutureWarning
            model.load_state_dict(torch.load("best_model.pt", weights_only=True))
            print("Successfully loaded best model.")
        else:
            print("No saved best model found. Using the last model state.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Continuing with current model state.")

    model.eval()

    preds_list = []
    trues_list = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            pred = model(batch_x)
            preds_list.append(pred.cpu().numpy())
            trues_list.append(batch_y.numpy())

    y_preds = np.concatenate(preds_list, axis=0)
    y_trues = np.concatenate(trues_list, axis=0)

    if scaler_y is not None:
        y_preds = scaler_y.inverse_transform(y_preds)
        y_trues = scaler_y.inverse_transform(y_trues)

    y_pred_1d = y_preds[:, 0]
    y_true_1d = y_trues[:, 0]

    plot_predictions(test_time, y_true_1d, y_pred_1d, title="Prediction vs Actual (Best Model)")


    # 9. Evaluation metrics output
    scores = compute_metrics(y_true_1d, y_pred_1d)
    print("Test metrics:", scores)


    # 10. Interpretability analysis
    model.train()
    x_sample = X_test_t[0:1].clone().to(device)
    y_sample = y_test_t[0:1].clone().to(device)

    grad_arr = compute_vanilla_gradients(
        model=model,
        x_input=x_sample,
        y_true=y_sample,
        criterion=criterion,
        device=device
    )

    print("Vanilla Grad shape:", grad_arr.shape)

    grad_2d = grad_arr[0]

    plt.figure(figsize=(8, 5))
    im = plt.imshow(grad_2d, aspect='auto', cmap='RdBu')
    cb = plt.colorbar(im)
    cb.set_label("Gradient Value", fontsize=9)
    plt.xticks(ticks=range(len(feature_cols)), labels=feature_cols, rotation=90, fontsize=8)
    plt.yticks(ticks=range(window_size), labels=range(window_size), fontsize=8)
    plt.xlabel("Features", fontsize=9)
    plt.ylabel("Time Steps", fontsize=9)
    plt.title("Vanilla Gradient Heatmap (time vs. feature)", fontsize=10)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
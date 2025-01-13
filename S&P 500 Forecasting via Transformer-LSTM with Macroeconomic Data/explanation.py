import torch
import numpy as np

def compute_vanilla_gradients(
    model,
    x_input,    # (batch, seq_len, feature_dim)
    y_true=None,
    criterion=None,
    device='cpu'
):  # Ensure not within no_grad() context

    # Input
    x_input = x_input.to(device)
    x_input.requires_grad = True

    # Forward
    output = model(x_input)  # => (batch, horizon)

    # Calculate loss
    if (y_true is not None) and (criterion is not None):
        y_true = y_true.to(device)
        loss = criterion(output, y_true)
    else:
        loss = output.sum()

    # Backward propagation
    model.zero_grad()
    loss.backward()

    # Get the gradient of x_input
    grad = x_input.grad  # shape same as x_input
    grad_arr = grad.detach().cpu().numpy()

    # Release requires_grad
    x_input.requires_grad = False

    return grad_arr
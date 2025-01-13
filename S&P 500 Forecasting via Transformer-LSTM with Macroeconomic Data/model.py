import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):


    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create a long enough PEs matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 2i even dimension -> sin, 2i+1 odd dimension -> cos
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd

        # Register as buffer so it won't be trained
        self.register_buffer('pe', pe.unsqueeze(0))  # shape = (1, max_len, d_model)

    def forward(self, x):

        seq_len = x.size(1)
        # self.pe[:, :seq_len, :] -> shape (1, seq_len, d_model)
        return x + self.pe[:, :seq_len, :]


class TimeTransformerLSTM(nn.Module):


    def __init__(
            self,
            feature_dim=22,
            d_model=64,
            nhead=4,
            num_encoder_layers=2,
            dim_feedforward=256,
            dropout=0.1,
            lstm_hidden_dim=128,
            lstm_num_layers=1,
            horizon=1,
            max_len=5000
    ):
        super(TimeTransformerLSTM, self).__init__()

        # Feature enhancement
        self.input_fc = nn.Linear(feature_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        # TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # (batch, seq_len, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # 4) LSTM
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0
        )

        # Linear output layer
        self.fc_out = nn.Linear(lstm_hidden_dim, horizon)


    def forward(self, x):

        # shape => (batch_size, seq_len, d_model)
        x_embed = self.input_fc(x)

        # Positional encoding
        # shape => (batch_size, seq_len, d_model)
        x_pos = self.pos_encoder(x_embed)

        # TransformerEncoder
        # shape => (batch_size, seq_len, d_model)
        x_trans = self.transformer_encoder(x_pos)

        # LSTM
        # shape => (batch_size, seq_len, lstm_hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(x_trans)

        # Hidden state of the last time step
        # shape => (batch_size, lstm_hidden_dim)
        last_hidden = lstm_out[:, -1, :]

        # Output
        # shape => (batch_size, horizon)
        out = self.fc_out(last_hidden)
        return out

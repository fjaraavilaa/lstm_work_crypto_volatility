import torch
import torch.nn as nn


class LSTMGeneralized_Output_SameSize(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        output_size: int,
    ):
        super(LSTMGeneralized_Output_SameSize, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


class LSTMGeneralized_Output_DifferentSize(nn.Module):
    def __init__(
        self,
        step_size,
        input_dimensions,
        n_layers: int,
        hidden_sizes: list,
        output_dimensions: int,
        device: str,
    ):
        super(LSTMGeneralized_Output_DifferentSize, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.sizes = hidden_sizes.copy()
        self.sizes.insert(0, input_dimensions)
        self.step_size = step_size
        self.device = device
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions
        self.hidden_layers = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=self.sizes[i],
                    hidden_size=self.sizes[i + 1],
                    num_layers=1,
                    batch_first=True,
                )
                for i in range(n_layers)
            ]
        )
        self.fc_layer = nn.Linear(
            self.step_size * self.hidden_sizes[len(self.hidden_sizes) - 1],
            self.output_dimensions,
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        i = 0
        for layer in self.hidden_layers:
            h_i = torch.zeros(1, batch_size, self.hidden_sizes[i]).to(self.device)
            c_i = torch.zeros(1, batch_size, self.hidden_sizes[i]).to(self.device)
            hidden_i = (h_i, c_i)
            x, hidden_i = layer(x, hidden_i)
            i += 1
        x = x.contiguous().view(batch_size, -1)
        return self.fc_layer(x).unsqueeze(1)
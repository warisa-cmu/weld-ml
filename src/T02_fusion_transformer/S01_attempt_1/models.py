from torch import nn
import torch
import torch.nn.functional as F


class MultiModalRegressionModel(nn.Module):
    def __init__(
        self,
        num_tabular_features,
        cnn_input_channels,
        lstm_input_size,
        lstm_hidden_size,
        lstm_num_layers,
        output_size,
    ):
        super().__init__()

        # MLP for tabular data
        self.mlp = nn.Sequential(
            nn.Linear(num_tabular_features, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU()
        )

        # CNN for image data
        self.cnn = nn.Sequential(
            nn.Conv2d(cnn_input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        # Calculate the flattened CNN output size given your image dimensions
        self.cnn_output_size = 32 * 4 * 4  # From nn.AdaptiveAvgPool2d((4, 4))

        # LSTM for time series data
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
        )
        self.lstm_fc = nn.Linear(lstm_hidden_size, 32)

        # Final regression head
        fusion_dim = 32 + self.cnn_output_size + 32
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 64), nn.ReLU(), nn.Linear(64, output_size)
        )

    def forward(self, x_tab, x_img, x_seq):
        # x_tab: (batch, num_tabular_features)
        # x_img: (batch, channels, height, width)
        # x_seq: (batch, seq_len, lstm_input_size)

        tab_out = self.mlp(x_tab)
        img_out = self.cnn(x_img)
        lstm_out, (lstm_hidden, c_n) = self.lstm(x_seq)
        lstm_last = lstm_hidden[-1]  # Return last hidden state of last layer
        lstm_feat = F.relu(self.lstm_fc(lstm_last))

        # Concatenate all features
        combined = torch.cat([tab_out, img_out, lstm_feat], dim=1)
        output = self.regressor(combined)
        return output.squeeze(-1)


class GatedResidualNetwork(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, context_dim=None, dropout=0.1
    ):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.context_linear = (
            nn.Linear(context_dim, hidden_dim) if context_dim is not None else None
        )
        self.elu = nn.ELU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(output_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.skip = (
            nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
        )
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x, context=None):
        out = self.linear1(x)
        if context is not None:
            out += self.context_linear(context)
        out = self.elu(out)
        out = self.linear2(out)
        out = self.dropout(out)
        gated = self.sigmoid(self.gate(out))
        out = out * gated
        skip = self.skip(x) if self.skip is not None else x
        out = self.layer_norm(out + skip)
        return out


class VariableSelectionNetwork(nn.Module):
    def __init__(
        self, num_inputs, input_dim, hidden_dim, context_dim=None, dropout=0.1
    ):
        super().__init__()
        self.num_inputs = num_inputs
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.variable_grns = nn.ModuleList(
            [
                GatedResidualNetwork(
                    input_dim, hidden_dim, hidden_dim, context_dim, dropout
                )
                for _ in range(num_inputs)
            ]
        )
        self.softmax = nn.Softmax(dim=-1)
        self.weight_grn = GatedResidualNetwork(
            num_inputs * hidden_dim, hidden_dim, num_inputs, context_dim, dropout
        )

    def forward(self, x, context=None):
        # x: (batch, num_inputs, input_dim)
        var_outputs = []
        for i, grn in enumerate(self.variable_grns):
            var_out = grn(x[:, i, :], context)
            var_outputs.append(var_out)
        var_outputs = torch.stack(var_outputs, dim=1)  # (batch, num_inputs, hidden_dim)
        flattened = var_outputs.view(var_outputs.size(0), -1)
        weights = self.weight_grn(flattened, context)
        weights = self.softmax(weights)  # (batch, num_inputs)
        # Weighted sum of variable outputs
        weighted_output = (var_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return weighted_output, weights  # (batch, hidden_dim), (batch, num_inputs)

from torch import nn
import torch
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(
        self,
        num_tabular_features,
        output_size,
        embedding_size,
    ):
        super().__init__()

        self.num_tabular_features = num_tabular_features

        # Embeddeding layer
        self.embedding_size = embedding_size
        self.embedders_tab_vars = nn.ModuleList(
            [
                nn.Linear(1, self.embedding_size)
                for _ in range(self.num_tabular_features)
            ]
        )

        self.vsn_tab_vars = VariableSelectionNetwork(
            num_inputs=num_tabular_features,
            input_dim=self.embedding_size,
            hidden_dim=num_tabular_features,  # This is the output size
        )

        # MLP for tabular data
        self.mlp = nn.Sequential(
            nn.Linear(num_tabular_features, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU()
        )

        self.regressor = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, output_size)
        )

    def forward(self, x_tab):
        # x_tab: (batch, num_tabular_features)

        tab_out = torch.stack(
            [
                self.embedders_tab_vars[i](x_tab[Ellipsis, i].unsqueeze(-1))
                for i in range(0, self.num_tabular_features)
            ],
            axis=-2,
        )  # (batch, num_features, embedding_size)

        tab_out, vsn_weights = self.vsn_tab_vars(tab_out)
        tab_out = self.mlp(tab_out)
        output = self.regressor(tab_out)
        return output.squeeze(-1), vsn_weights


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

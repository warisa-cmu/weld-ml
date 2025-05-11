import torch
import torch.nn as nn
import torch.nn.functional as F


class GRN(nn.Module):
    """Gated Residual Network"""

    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.skip = nn.Linear(input_size, output_size, bias=False)
        self.layernorm = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Gated transformation
        output = F.elu(self.linear1(x))
        output = self.dropout(self.linear2(output))
        # Residual connection
        output = self.layernorm(output + self.skip(x))
        return output


class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_sizes, hidden_size, dropout=0.1):
        super().__init__()
        self.input_sizes = input_sizes
        self.hidden_size = hidden_size

        # Prescalers for each input variable
        self.prescalers = nn.ModuleDict(
            {name: nn.Linear(size, hidden_size) for name, size in input_sizes.items()}
        )

        # GRN for variable weights calculation
        self.grn = GRN(
            input_size=hidden_size * len(input_sizes),
            hidden_size=hidden_size,
            output_size=len(input_sizes),
            dropout=dropout,
        )

    def forward(self, inputs, context=None):
        # Transform inputs through prescalers
        transformed = [self.prescalers[name](x) for name, x in inputs.items()]
        transformed = torch.stack(transformed, dim=-2)  # [batch, num_vars, hidden]

        # Compute variable weights
        flatten = transformed.flatten(start_dim=-2)  # [batch, num_vars*hidden]
        if context is not None:
            flatten = torch.cat([flatten, context], dim=-1)

        weights = self.grn(flatten)
        weights = F.softmax(weights, dim=-1)  # [batch, num_vars]

        # Apply weighted combination
        weighted = transformed * weights.unsqueeze(-1)
        output = weighted.sum(dim=-2)  # [batch, hidden]

        return output, weights


# Define input variables with different feature sizes
input_sizes = {"numeric": 10, "categorical": 5, "temporal": 8}
vsn = VariableSelectionNetwork(input_sizes, hidden_size=32)

# Sample inputs (batch_size=64)
inputs = {
    "numeric": torch.randn(64, 10),
    "categorical": torch.randn(64, 5),
    "temporal": torch.randn(64, 8),
}

output, weights = vsn(inputs)
print(f"Selected features shape: {output.shape}")  # torch.Size([64, 32])
print(f"Variable weights: {weights[0]}")  # e.g. tensor([0.4, 0.1, 0.5])

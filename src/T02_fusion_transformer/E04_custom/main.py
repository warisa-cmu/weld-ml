import torch
import torch.nn as nn
import torch.nn.functional as F


class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_sizes, hidden_size, dropout=0.1):
        super().__init__()
        self.input_sizes = input_sizes
        self.hidden_size = hidden_size

        # Variable-specific encoders
        self.encoders = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(size, hidden_size), nn.ReLU(), nn.Dropout(dropout)
                )
                for name, size in input_sizes.items()
            }
        )

        # Variable selection mechanism
        self.selection_layer = nn.Sequential(
            nn.Linear(sum(input_sizes.values()), len(input_sizes)), nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Encode each variable
        encoded_vars = {name: self.encoders[name](x[name]) for name in self.input_sizes}

        # Concatenate all inputs for selection
        selection_input = torch.cat([x[name] for name in self.input_sizes], dim=1)

        # Calculate importance weights
        importance_weights = self.selection_layer(selection_input)

        # Weight and combine variable representations
        var_outputs = torch.stack(
            [encoded_vars[name] for name in self.input_sizes], dim=1
        )
        weighted_outputs = var_outputs * importance_weights.unsqueeze(-1)
        combined_output = weighted_outputs.sum(dim=1)

        return combined_output, importance_weights


class InterpretableMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, query, key, value, need_weights=True):
        # Return both output and attention weights for interpretability
        attn_output, attn_weights = self.mha(
            query,
            key,
            value,
            need_weights=need_weights,
            average_attn_weights=False,  # Get per-head attention weights
        )
        return attn_output, attn_weights


class TimeSeriesProcessor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()

        # Variable selection for time series features
        self.variable_selection = VariableSelectionNetwork(
            input_sizes={"time_features": input_dim}, hidden_size=hidden_dim
        )

        # Positional encoding
        self.pos_encoder = nn.Embedding(
            1000, hidden_dim
        )  # Support sequences up to length 1000

        # Stack of attention layers
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attention": InterpretableMultiheadAttention(
                            hidden_dim, num_heads, dropout
                        ),
                        "norm1": nn.LayerNorm(hidden_dim),
                        "feedforward": nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim * 4),
                            nn.ReLU(),
                            nn.Linear(hidden_dim * 4, hidden_dim),
                        ),
                        "norm2": nn.LayerNorm(hidden_dim),
                    }
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, features]
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Process each time step with variable selection
        time_step_embeddings = []
        var_importances = []

        for t in range(seq_len):
            # Prepare input for variable selection
            time_step_dict = {"time_features": x[:, t, :]}
            embedding, importance = self.variable_selection(time_step_dict)
            time_step_embeddings.append(embedding)
            var_importances.append(importance)

        # Stack time step embeddings
        sequence = torch.stack(time_step_embeddings, dim=1)

        # Add positional encoding
        positions = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len)
        sequence = sequence + self.pos_encoder(positions)

        # Store attention weights for interpretability
        attention_weights = []

        # Apply attention layers
        for layer in self.layers:
            # Self-attention block
            residual = sequence
            normalized = layer["norm1"](sequence)
            attended, weights = layer["attention"](normalized, normalized, normalized)
            sequence = residual + attended
            attention_weights.append(weights)

            # Feedforward block
            residual = sequence
            normalized = layer["norm2"](sequence)
            sequence = residual + layer["feedforward"](normalized)

        # Global pooling for sequence representation
        global_repr = sequence.mean(dim=1)

        return global_repr, attention_weights, torch.stack(var_importances, dim=1)

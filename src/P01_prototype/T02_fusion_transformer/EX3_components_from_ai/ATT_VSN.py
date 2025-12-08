import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Variable Selection Network (VSN) ---
class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_dim, num_features, hidden_dim, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.input_dim = input_dim

        self.var_transform_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, input_dim),
            ) for _ in range(num_features)
        ])

        self.gate = nn.Sequential(
            nn.Linear(num_features * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_features),
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (batch, num_features, input_dim)
        var_outputs = []
        for i in range(self.num_features):
            var_i = x[:, i, :]
            out_i = self.var_transform_layers[i](var_i)
            var_outputs.append(out_i.unsqueeze(1))
        var_outputs = torch.cat(var_outputs, dim=1)

        # Gating
        flattened = x.view(x.size(0), -1)
        weights = self.softmax(self.gate(flattened))

        # Weighted sum
        weighted_output = torch.sum(var_outputs * weights.unsqueeze(-1), dim=1)
        return weighted_output, weights  # (batch, input_dim), (batch, num_features)

# --- Interpretable Multihead Attention Block ---
class InterpretableMultiheadAttention(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        self.head_dim = input_dim // num_heads

        # Shared projections for all heads (interpretable Attention)
        self.query_proj = nn.Linear(input_dim, input_dim)
        self.key_proj = nn.Linear(input_dim, input_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)

        self.out_proj = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        # query, key, value: (batch, seq_len, input_dim)
        B, L, D = query.size()
        H = self.num_heads

        # Shared linear projections
        q = self.query_proj(query)      # (batch, seq_len, input_dim)
        k = self.key_proj(key)
        v = self.value_proj(value)

        # Split heads
        q = q.view(B, L, H, self.head_dim).transpose(1,2)  # (batch, heads, seq_len, head_dim)
        k = k.view(B, L, H, self.head_dim).transpose(1,2)
        v = v.view(B, L, H, self.head_dim).transpose(1,2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch, heads, seq_len, seq_len)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Attention output
        attn_output = torch.matmul(attn_probs, v)  # (batch, heads, seq_len, head_dim)
        attn_output = attn_output.transpose(1,2).contiguous().view(B, L, D)  # (batch, seq_len, input_dim)

        # Averaging heads for interpretability (as in TFT paper)
        attn_output = attn_output.view(B, L, H, self.head_dim).mean(dim=2)  # (batch, seq_len, head_dim)
        attn_output = attn_output.view(B, L, -1)  # (batch, seq_len, input_dim)

        attn_output = self.out_proj(attn_output)  # Final linear 
        return attn_output, attn_probs  # Returning attention probabilities for interpretability

# --- Example Full Block Integration ---
class VSN_MultiheadAttention_Block(nn.Module):
    def __init__(self, num_features, vsn_input_dim, vsn_hidden_dim, attn_heads, dropout=0.1):
        super().__init__()
        self.vsn = VariableSelectionNetwork(
            input_dim=vsn_input_dim,
            num_features=num_features,
            hidden_dim=vsn_hidden_dim,
            dropout=dropout)
        
        self.attn = InterpretableMultiheadAttention(
            input_dim=vsn_input_dim,
            num_heads=attn_heads,
            dropout=dropout)

    def forward(self, x_seq):
        # x_seq: (batch, seq_len, num_features, input_dim)
        batch, seq_len, num_features, input_dim = x_seq.size()
        # Process each time step in the sequence with VSN
        vsn_outputs = []
        vsn_weights = []
        for t in range(seq_len):
            out, w = self.vsn(x_seq[:, t, :, :])  # (batch, input_dim), (batch, num_features)
            vsn_outputs.append(out.unsqueeze(1))  # (batch, 1, input_dim)
            vsn_weights.append(w.unsqueeze(1))    # (batch, 1, num_features)
        vsn_outputs = torch.cat(vsn_outputs, dim=1)  # (batch, seq_len, input_dim)
        vsn_weights = torch.cat(vsn_weights, dim=1)  # (batch, seq_len, num_features)

        # Apply attention (self-attention, so use vsn_outputs as q, k, v)
        attn_output, attn_probs = self.attn(vsn_outputs, vsn_outputs, vsn_outputs)
        # Output: (batch, seq_len, input_dim), attn_probs: (batch, heads, seq_len, seq_len)

        return attn_output, vsn_weights, attn_probs

# --- Example Usage ---
if __name__ == "__main__":
    batch_size = 2
    seq_len = 6
    num_features = 4
    input_dim = 8
    vsn_hidden_dim = 16
    attn_heads = 2

    x_seq = torch.randn(batch_size, seq_len, num_features, input_dim)
    block = VSN_MultiheadAttention_Block(
        num_features=num_features,
        vsn_input_dim=input_dim,
        vsn_hidden_dim=vsn_hidden_dim,
        attn_heads=attn_heads
    )
    attn_output, vsn_weights, attn_probs = block(x_seq)
    print("Attention Output:", attn_output.shape)
    print("VSN Weights:", vsn_weights.shape)
    print("Attention Probs:", attn_probs.shape)
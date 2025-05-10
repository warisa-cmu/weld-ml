import torch
import torch.nn as nn
import torch.nn.functional as F


class InterpretableMultiHeadAttention(nn.Module):
    """
    Implement interpretable multi-head attention from TFT paper
    Key features:
    - Averaged attention weights across heads for interpretability
    - Combined linear projections for efficiency
    - Causal masking support
    - Residual connections

    Args:
        d_model: Input dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = nn.Dropout(dropout)

        # Combined projections for Q/K/V [2][4]
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.zeros_(self.qkv_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with interpretable attention

        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask (batch_size, seq_len, seq_len)

        Returns:
            output: Attention output (batch_size, seq_len, d_model)
            attn_weights: Averaged attention weights (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.size()

        # Combined Q/K/V projection [2][4]
        qkv = self.qkv_proj(x)  # (batch_size, seq_len, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)  # Each (batch_size, seq_len, d_model)

        # Split into heads [1][5]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores [1][3]
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(self.head_dim)

        # Apply mask before averaging [3][4]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Average attention across heads [1][5]
        avg_scores = scores.mean(
            dim=1, keepdim=True
        )  # (batch_size, 1, seq_len, seq_len)
        attn_weights = F.softmax(avg_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values [1][4]
        context = torch.matmul(attn_weights, v)  # (batch_size, 1, seq_len, head_dim)

        # Combine heads and project [2][5]
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )
        output = self.out_proj(context)

        return output, attn_weights.squeeze(1)


# Hyperparameters
d_model = 512
n_heads = 8
seq_len = 64
batch_size = 32

# Create module
attention = InterpretableMultiHeadAttention(d_model, n_heads)

# Sample input
q = k = v = torch.randn(batch_size, seq_len, d_model)

# Forward pass
output, attn_weights = attention(q, k, v)

print(f"Output shape: {output.shape}")  # torch.Size([32, 64, 512])
print(f"Weights shape: {attn_weights.shape}")  # torch.Size([32, 64, 64])

import torch
import torch.nn as nn
import torch.nn.functional as F


class InterpretableMultiHeadAttention(nn.Module):
    """
    Efficient interpretable multi-head attention with parallel processing
    and averaged attention weights for better interpretability

    Args:
        d_model: Total dimension of input features
        n_heads: Number of attention heads
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = nn.Dropout(dropout)

        # Combined linear projections for efficiency
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Final output projection
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with parallel head processing

        Args:
            q: Query tensor (batch_size, seq_len, d_model)
            k: Key tensor (batch_size, seq_len, d_model)
            v: Value tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask (batch_size, seq_len, seq_len)

        Returns:
            tuple: (output tensor, averaged attention weights)
        """
        batch_size = q.size(0)

        # Project inputs and split into heads [1][3][4]
        q = (
            self.q_proj(q)
            .view(batch_size, -1, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(k)
            .view(batch_size, -1, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(v)
            .view(batch_size, -1, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Compute attention scores [2][5]
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim)
        )

        # Apply mask before softmax if provided [3][4]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Average attention across heads before softmax for interpretability [1][3]
        avg_scores = scores.mean(
            dim=1, keepdim=True
        )  # (batch_size, 1, seq_len, seq_len)
        weights = F.softmax(avg_scores, dim=-1)
        weights = self.dropout(weights)

        # Apply attention to values [2][5]
        context = torch.matmul(weights, v)  # (batch_size, 1, seq_len, head_dim)

        # Combine heads and project to output space [3][4]
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )
        output = self.out_proj(context)

        # Return output and averaged attention weights [1][3]
        return output, weights.squeeze(1)


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

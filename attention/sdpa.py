import torch
import torch.nn.functional as F


def sdpa(Q, K, V):
    """
    Q, K, V: (batch, heads, seq_len, head_dim)
    """
    return F.scaled_dot_product_attention(Q, K, V)

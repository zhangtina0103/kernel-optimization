import torch


def naive_attention(Q, K, V, scale=None):
    """
    Q: (batch, heads, seq_len, head_dim)
    K: (batch, heads, seq_len, head_dim)
    V: (batch, heads, seq_len, head_dim)
    """
    if scale is None:
        scale = Q.shape[-1] ** -0.5  # 1/sqrt(head_dim)

    # Q @ K^T
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale

    # softmax over last dimension
    P = torch.softmax(S, dim=-1)

    # weighted sum of values
    O = torch.matmul(P, V)

    return O

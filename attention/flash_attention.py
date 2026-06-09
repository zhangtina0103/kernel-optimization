import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_kernel(
    Q, K, V, O,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    N, D,
    scale,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # which batch, head, and query block am i
    batch = tl.program_id(0)
    head  = tl.program_id(1)
    q_idx = tl.program_id(2)

    # base pointers for this batch and head
    q_ptr = Q + batch * stride_qb + head * stride_qh
    k_ptr = K + batch * stride_kb + head * stride_kh
    v_ptr = V + batch * stride_vb + head * stride_vh
    o_ptr = O + batch * stride_ob + head * stride_oh

    # which query positions am i responsible for
    q_offs = q_idx * BLOCK_N + tl.arange(0, BLOCK_N)  # (BLOCK_N,)
    d_offs = tl.arange(0, BLOCK_D)                     # (BLOCK_D,)

    # load my tile of Q: shape (BLOCK_N, BLOCK_D)
    q_mask = (q_offs[:, None] < N) & (d_offs[None, :] < D)
    q = tl.load(q_ptr + q_offs[:, None] * stride_qn
                       + d_offs[None, :] * stride_qd,
                mask=q_mask, other=0.0)

    # initialize online softmax state
    m = tl.full((BLOCK_N,), float('-inf'), dtype=tl.float32)  # running max
    l = tl.zeros((BLOCK_N,), dtype=tl.float32)                # running sum
    acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)      # output accumulator

    # loop over K and V tiles
    num_blocks = (N + BLOCK_N - 1) // BLOCK_N
    for j in range(num_blocks):
        k_offs = j * BLOCK_N + tl.arange(0, BLOCK_N)

        # load tile of K shape (BLOCK_N, BLOCK_D)
        k_mask = (k_offs[:, None] < N) & (d_offs[None, :] < D)
        k = tl.load(k_ptr + k_offs[:, None] * stride_kn
                           + d_offs[None, :] * stride_kd,
                    mask=k_mask, other=0.0)

        # compute attention scores: S = Q @ K^T * scale
        # shape: (BLOCK_N, BLOCK_N)
        s = tl.dot(q, tl.trans(k)) * scale

        # mask out of bounds positions
        s = tl.where(
            (q_offs[:, None] < N) & (k_offs[None, :] < N),
            s, float('-inf')
        )

        # softmax update
        # find max of current tile
        m_new = tl.maximum(m, tl.max(s, axis=1))

        # rescale previous accumulator and sum
        alpha = tl.exp(m - m_new)        # correction factor for previous tiles
        l = l * alpha + tl.sum(tl.exp(s - m_new[:, None]), axis=1)

        # load V tile and update accumulator
        v_mask = (k_offs[:, None] < N) & (d_offs[None, :] < D)
        v = tl.load(v_ptr + k_offs[:, None] * stride_vn
                           + d_offs[None, :] * stride_vd,
                    mask=v_mask, other=0.0)

        # rescale old acc and add new contribution
        acc = acc * alpha[:, None] + tl.dot(tl.exp(s - m_new[:, None]).to(tl.float32), v)

        # update running max
        m = m_new

    # normalize accumulator by running sum
    acc = acc / l[:, None]

    # write output
    o_mask = (q_offs[:, None] < N) & (d_offs[None, :] < D)
    tl.store(o_ptr + q_offs[:, None] * stride_on
                   + d_offs[None, :] * stride_od,
             acc, mask=o_mask)


def flash_attention(Q, K, V):
    """
    Q, K, V: (batch, heads, seq_len, head_dim)
    returns O: (batch, heads, seq_len, head_dim)
    """
    batch, heads, N, D = Q.shape
    scale = D ** -0.5

    O = torch.zeros_like(Q)

    BLOCK_N = 32
    BLOCK_D = triton.next_power_of_2(D)

    grid = (batch, heads, (N + BLOCK_N - 1) // BLOCK_N)

    flash_attention_kernel[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        N, D, scale,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )

    return O

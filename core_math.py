import torch
from torch import Tensor
from einops import rearrange


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    """
    计算自注意力机制。

    该函数通过旋转位置编码（RoPE）对查询（Q）和键（K）进行编码，
    然后计算缩放点积注意力，最后将输出重塑为原始形状。

    参数:
        q (Tensor): 查询张量，形状为 (B, H, L, D)。
        k (Tensor): 键张量，形状为 (B, H, L, D)。
        v (Tensor): 值张量，形状为 (B, H, L, D)。
        pe (Tensor): 位置编码张量，形状为 (B, L, D)。

    返回:
        Tensor: 注意力机制的输出，形状为 (B, L, H*D)。
    """
    # 应用旋转位置编码（RoPE）到查询 Q 和键 K
    q, k = apply_rope(q, k, pe)  # q, k: (B, H, L, D)
    
    # 计算缩放点积注意力
    # F.scaled_dot_product_attention 是 PyTorch 1.11 及以上版本中引入的函数
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # x: (B, H, L, D)

    # 将输出重塑为 (B, L, H*D)，以匹配原始输入的形状
    x = rearrange(x, "B H L D -> B L (H D)")  # x: (B, L, H*D)

    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    """
    生成旋转位置编码（RoPE）矩阵。

    RoPE 通过对输入的位置编码进行旋转操作，引入位置信息到注意力机制中。

    参数:
        pos (Tensor): 输入的位置编码张量，形状为 (..., N)。
        dim (int): 位置编码的维度。
        theta (int): RoPE 编码中的角度参数，用于控制旋转的幅度。

    返回:
        Tensor: RoPE 矩阵，形状为 (..., N, 2, 2)。
    """
    # 确保维度是偶数，因为 RoPE 需要将每个维度分成实部和虚部
    assert dim % 2 == 0

    # 生成缩放因子，形状为 (dim/2,)
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim  # scale: (dim/2,)

    # 计算 omega，形状为 (dim/2,)
    omega = 1.0 / (theta**scale)  # omega: (dim/2,)

    # 计算旋转角度，形状为 (..., N, dim/2)
    out = torch.einsum("...n,d->...nd", pos, omega)  # out: (..., N, dim/2)

    # 生成 RoPE 矩阵，形状为 (..., N, dim/2, 2, 2)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)  # out: (..., N, dim/2, 2, 2)

    # 重塑 RoPE 矩阵为 (..., N, dim/2 * 2 * 2) = (..., N, dim)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)  # out: (..., N, dim/2, 2, 2)

    # 返回浮点类型的 RoPE 矩阵
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    """
    应用旋转位置编码（RoPE）到查询 Q 和键 K。

    该函数将 RoPE 矩阵与查询和键张量相乘，实现位置信息的引入。

    参数:
        xq (Tensor): 查询张量，形状为 (B, H, L, D)。
        xk (Tensor): 键张量，形状为 (B, H, L, D)。
        freqs_cis (Tensor): RoPE 矩阵，形状为 (B, L, D/2, 2, 2)。

    返回:
        Tuple[Tensor, Tensor]: 应用 RoPE 后的查询和键张量，形状均为 (B, H, L, D)。
    """
    # 将查询和键张量转换为浮点类型，并重塑为 (..., N, D/2, 1, 2)
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)  # xq_: (B, H, L, D/2, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)  # xk_: (B, H, L, D/2, 1, 2)

    # 应用 RoPE 矩阵到查询和键张量
    # freqs_cis[..., 0] 和 freqs_cis[..., 1] 分别对应于 cos(theta) 和 -sin(theta) 以及 sin(theta) 和 cos(theta)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]  # xq_out: (B, H, L, D/2, 1)
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]  # xk_out: (B, H, L, D/2, 1)

    # 将输出重塑回原始形状 (B, H, L, D)
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

import math
from dataclasses import dataclass
import torch
from einops import rearrange
from torch import Tensor, nn

from core_math import attention, rope


class EmbedND(nn.Module):
    """
    EmbedND 类用于对多维索引进行旋转位置编码（Rotary Position Embedding, RoPE）。

    该类通过将多维索引的每个维度分别进行 RoPE 编码，然后将它们在指定维度上拼接起来，
    生成最终的位置编码。

    参数:
        dim (int): 每个索引维度对应的嵌入维度。
        theta (int): RoPE 编码中的角度参数，用于控制旋转的幅度。
        axes_dim (List[int]): 每个索引维度的轴维度，用于确定每个维度对应的轴长度。
    """
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        # 保存每个索引维度对应的嵌入维度
        self.dim = dim
        # 保存 RoPE 编码中的角度参数
        self.theta = theta
        # 保存每个索引维度的轴维度
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        """
        前向传播方法。

        对输入的多维索引进行 RoPE 编码，并生成最终的位置编码。

        参数:
            ids (Tensor): 输入的多维索引张量，形状为 (..., n_axes)，其中 n_axes 是索引的维度数。

        返回:
            Tensor: 输出的位置编码张量，形状为 (..., n_axes, dim)。
        """
        # 获取索引的维度数
        n_axes = ids.shape[-1]
        # 对每个索引维度分别进行 RoPE 编码，并将结果在指定维度上拼接起来
        # rope 函数需要每个索引维度的轴维度和角度参数
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)], # 对第 i 个索引维度进行 RoPE 编码
            dim=-3, # 在倒数第 3 个维度上进行拼接
        )

        # 在指定维度上添加一个维度，得到最终的位置编码
        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    生成正弦时间步长嵌入。

    该函数生成基于正弦和余弦函数的时间步长嵌入，用于将时间步长信息编码为向量表示。

    参数:
        t (Tensor): 输入的时间步长张量，形状为 (N,)，每个元素表示一个批次元素的时间步长，可以是分数。
        dim (int): 输出的嵌入维度。
        max_period (int, optional): 控制嵌入的最小频率。默认为 10000。
        time_factor (float, optional): 时间步长的缩放因子，用于调整嵌入的尺度。默认为 1000.0。

    返回:
        Tensor: 输出的位置嵌入张量，形状为 (N, dim)。
    """
    # 对时间步长进行缩放
    t = time_factor * t
    # 计算嵌入维度的一半
    half = dim // 2
    # 生成频率参数
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    # 计算角度参数
    args = t[:, None].float() * freqs[None] # (N, 1) * (1, half) -> (N, half)

    # 生成嵌入向量
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1) # (N, half) -> (N, dim)

    # 如果嵌入维度为奇数，则在最后添加一个零向量以匹配维度
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1) # (N, dim) -> (N, dim + 1) -> (N, dim)
    
    # 如果输入张量是浮点数类型，则将嵌入张量转换为与输入张量相同的类型
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    """
    MLPEmbedder 类是一个多层感知机（MLP）嵌入器，用于将输入向量嵌入到高维空间。

    该嵌入器由两个线性层和一个 SiLU 激活函数组成，能够有效地捕捉输入数据的复杂非线性关系。

    参数:
        in_dim (int): 输入向量的维度。
        hidden_dim (int): 隐藏层的维度，用于控制 MLP 的容量。
    """
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        # 定义输入线性层，将输入向量映射到隐藏层
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        # 定义 SiLU 激活函数，用于引入非线性
        self.silu = nn.SiLU()
        # 定义输出线性层，将隐藏层映射回隐藏层维度
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播方法。

        将输入向量通过 MLP 嵌入器进行处理，生成嵌入后的向量。

        参数:
            x (Tensor): 输入张量，形状为 (N, in_dim)。

        返回:
            Tensor: 嵌入后的输出张量，形状为 (N, hidden_dim)。
        """
        # 通过输入线性层 (N, in_dim) -> (N, hidden_dim)
        # 通过 SiLU 激活函数 (N, hidden_dim)
        # 通过输出线性层 (N, hidden_dim) -> (N, hidden_dim)
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    """
    RMSNorm 类实现了均方根归一化（Root Mean Square Normalization）。

    RMSNorm 是一种归一化方法，通过对输入张量沿着指定维度计算均方根（RMS），然后进行缩放，
    以稳定训练过程并加速收敛。

    参数:
        dim (int): 归一化的维度。
    """
    def __init__(self, dim: int):
        super().__init__()
        # 定义缩放参数，用于调整归一化后的结果
        # 初始化为全 1 的可学习参数
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        """
        前向传播方法。

        对输入张量进行 RMS 归一化处理。

        参数:
            x (Tensor): 输入张量。

        返回:
            Tensor: 归一化后的输出张量。
        """
        # 将输入张量转换为浮点类型，以防止在计算中出现整数类型的问题
        x_dtype = x.dtype
        x = x.float()

        # 计算输入张量的均方根（RMS）
        # torch.mean(x**2, dim=-1, keepdim=True) 计算每个样本的均方值
        # torch.rsqrt 计算均方根的倒数，即 1 / sqrt(mean(x**2))
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6) # 加 1e-6 防止除零

        # 对输入张量进行缩放，得到归一化后的结果
        # 将结果转换回原始数据类型
        return (x * rrms).to(dtype=x_dtype) * self.scale  # 缩放后乘以 scale 参数


class QKNorm(torch.nn.Module):
    """
    QKNorm 类实现了查询（Q）和键（K）的归一化，用于自注意力机制。

    该类对查询和键张量分别进行 RMS 归一化处理，以确保它们具有相似的尺度，从而稳定注意力计算。

    参数:
        dim (int): 归一化的维度。
    """
    def __init__(self, dim: int):
        super().__init__()
        # 定义查询和键的 RMS 归一化层
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """
        前向传播方法。

        对查询和键张量进行归一化处理，并返回转换后的查询和键张量。

        参数:
            q (Tensor): 查询张量。
            k (Tensor): 键张量。
            v (Tensor): 值张量。

        返回:
            Tuple[Tensor, Tensor]: 归一化后的查询和键张量。
        """
        # 对查询和键张量分别进行 RMS 归一化
        q = self.query_norm(q)
        k = self.key_norm(k)
        # 将查询和键张量转换为与值张量相同的类型
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    """
    自注意力机制（Self-Attention）模块，用于捕捉输入序列中不同位置之间的关系。

    该模块通过计算查询（Q）、键（K）和值（V）来生成注意力权重，并利用这些权重对值进行加权求和，
    从而实现对输入序列的自适应表示。

    参数:
        dim (int): 输入和输出的维度。
        num_heads (int, optional): 多头注意力机制中的头数。默认为 8。
        qkv_bias (bool, optional): 查询（Q）、键（K）和值（V）线性层是否使用偏置。默认为 False。
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        # 保存多头注意力机制中的头数
        self.num_heads = num_heads
        # 计算每个头的维度
        head_dim = dim // num_heads

        # 定义查询（Q）、键（K）和值（V）的线性变换层
        # 输入维度为 dim，输出维度为 dim * 3（因为 Q, K, V 各占 dim）
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 定义 Q 和 K 的归一化层
        self.norm = QKNorm(head_dim)
        # 定义输出投影层，将注意力机制的输出映射回原始维度
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        """
        前向传播方法。

        对输入张量进行自注意力机制处理，并生成输出张量。

        参数:
            x (Tensor): 输入张量，形状为 (B, L, dim)。
            pe (Tensor): 位置编码张量，形状为 (B, L, dim)。

        返回:
            Tensor: 经过自注意力机制处理后的输出张量，形状为 (B, L, dim)。
        """
        # 通过 QKV 线性变换层，得到 Q, K, V
        qkv = self.qkv(x) # (B, L, dim * 3)

        # 重塑 Q, K, V 张量以适应多头注意力机制
        # 重塑后的形状为 (3, B, num_heads, L, head_dim)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)

        # 对 Q 和 K 进行归一化处理
        q, k = self.norm(q, k, v)

        # 计算自注意力机制的输出
        x = attention(q, k, v, pe=pe)  # (B, L, dim)

        # 通过输出投影层，将注意力输出映射回原始维度
        x = self.proj(x) # (B, L, dim)
        return x


@dataclass
class ModulationOut:
    """
    ModulationOut 数据类，用于存储调制（Modulation）模块的输出。

    属性:
        shift (Tensor): 移位张量，用于调整特征图的均值。
        scale (Tensor): 缩放张量，用于调整特征图的方差。
        gate (Tensor): 门控张量，用于控制特征的通过量。
    """
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    """
    调制（Modulation）模块，用于对输入特征图进行调制处理。

    该模块通过线性变换和 SiLU 激活函数生成移位、缩放和门控张量，
    并根据这些张量对输入特征图进行调制。

    参数:
        dim (int): 输入特征的维度。
        double (bool): 是否使用双调制。如果为 True，则生成两组调制参数；否则，只生成一组。
    """
    def __init__(self, dim: int, double: bool):
        super().__init__()
        # 是否使用双调制的标志
        self.is_double = double
        # 根据是否使用双调制，设置乘数
        self.multiplier = 6 if double else 3
        # 定义线性变换层，将输入特征映射到调制参数
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        """
        前向传播方法。

        对输入张量进行调制处理，生成调制参数。

        参数:
            vec (Tensor): 输入张量，形状为 (B, L, dim)。

        返回:
            Tuple[ModulationOut, Optional[ModulationOut]]: 一组或两组调制参数。
        """
        # 对输入张量进行 SiLU 激活
        # (B, 1, dim * multiplier) -> (B, 1, dim) * multiplier
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        # 将输出张量在最后一个维度上拆分为多组调制参数
        # 每组调制参数包含 shift, scale, gate
        # 根据是否使用双调制，返回相应的调制输出
        return (
            ModulationOut(*out[:3]), # 第一组调制参数
            ModulationOut(*out[3:]) if self.is_double else None, # 第二组调制参数（如果使用双调制）
        )


class DoubleStreamBlock(nn.Module):
    """
    双流块（DoubleStreamBlock）类，用于同时处理图像和文本两个数据流。

    该模块通过多头自注意力机制和前馈神经网络（MLP）对图像和文本特征进行并行处理，
    并通过调制（Modulation）模块对特征进行动态调整，以增强模型的表达能力。

    参数:
        hidden_size (int): 隐藏层的维度，用于控制特征图的通道数。
        num_heads (int): 多头自注意力机制中的头数，用于并行处理不同的注意力头。
        mlp_ratio (float): MLP 层中隐藏层大小的比例，用于计算 MLP 的中间层大小。
        qkv_bias (bool, optional): 查询（Q）、键（K）和值（V）线性层是否使用偏置。默认为 False。
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()

        # 计算 MLP 的隐藏层大小
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        # 保存多头数和隐藏层大小
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        # 初始化图像流的调制模块，使用双调制
        self.img_mod = Modulation(hidden_size, double=True)
        # 初始化图像流的 LayerNorm 层，不使用仿射变换，epsilon 设置为 1e-6
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # 初始化图像流的自注意力模块
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        # 初始化图像流的第二个 LayerNorm 层
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # 初始化图像流的 MLP 模块，使用 GELU 激活函数
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        # 初始化文本流的调制模块，使用双调制
        self.txt_mod = Modulation(hidden_size, double=True)
        # 初始化文本流的 LayerNorm 层，不使用仿射变换，epsilon 设置为 1e-6
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # 初始化文本流的自注意力模块
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        # 初始化文本流的第二个 LayerNorm 层
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # 初始化文本流的 MLP 模块，使用 GELU 激活函数
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        """
        前向传播方法。

        对图像和文本特征进行双流处理，包括调制、归一化、自注意力机制和 MLP。

        参数:
            img (Tensor): 输入的图像特征张量，形状为 (B, L_img, hidden_size)。
            txt (Tensor): 输入的文本特征张量，形状为 (B, L_txt, hidden_size)。
            vec (Tensor): 输入的辅助向量张量，形状为 (B, L_vec, hidden_size)。
            pe (Tensor): 位置编码张量，形状为 (B, L, hidden_size)。

        返回:
            Tuple[Tensor, Tensor]: 处理后的图像和文本特征张量。
        """
        # 对辅助向量进行调制处理，得到图像和文本的调制参数
        img_mod1, img_mod2 = self.img_mod(vec) # 图像调制参数
        txt_mod1, txt_mod2 = self.txt_mod(vec) # 文本调制参数

        # prepare image for attention
        # 准备图像特征进行注意力机制处理

        # 对图像特征进行归一化
        img_modulated = self.img_norm1(img)
        # 应用调制参数
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        # 通过 QKV 线性变换
        img_qkv = self.img_attn.qkv(img_modulated)
        # 重塑张量以适应多头注意力
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        # 对 Q 和 K 进行归一化
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        # 准备文本特征进行注意力机制处理

        # 对文本特征进行归一化
        txt_modulated = self.txt_norm1(txt)
        # 应用调制参数
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        # 通过 QKV 线性变换
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        # 重塑张量以适应多头注意力
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        # 对 Q 和 K 进行归一化
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        # 合并图像和文本的 Q, K, V 进行注意力计算

        # 在多头维度上拼接
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        # 计算注意力输出
        attn = attention(q, k, v, pe=pe)
        # 分离图像和文本的注意力输出
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        # 计算图像块的输出
        # 应用注意力输出和门控
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        # 应用 MLP 和调制参数
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        # 计算文本块的输出
        # 应用注意力输出和门控
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        # 应用 MLP 和调制参数
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """
    """
    单流块（SingleStreamBlock）类，用于处理单一数据流（如图像或文本）。

    该模块通过多头自注意力机制和前馈神经网络（MLP）对输入特征进行并行处理，
    并通过调制（Modulation）模块对特征进行动态调整，以增强模型的表达能力。

    参数:
        hidden_size (int): 隐藏层的维度，用于控制特征图的通道数。
        num_heads (int): 多头自注意力机制中的头数，用于并行处理不同的注意力头。
        mlp_ratio (float, optional): MLP 层中隐藏层大小的比例，用于计算 MLP 的中间层大小。默认为 4.0。
        qk_scale (float, optional): Q 和 K 的缩放因子。如果未指定，则使用 head_dim 的平方根的倒数。默认为 None。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()

        # 保存隐藏层大小和头数
        self.hidden_dim = hidden_size
        self.num_heads = num_heads

        # 计算每个头的维度
        head_dim = hidden_size // num_heads

        # 如果未指定 QK 缩放因子，则使用 head_dim 的平方根的倒数
        self.scale = qk_scale or head_dim**-0.5

        # 计算 MLP 的隐藏层大小
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)

        # qkv and mlp_in
        # 定义第一个线性层，用于生成 Q, K, V 和 MLP 的输入
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)

        # proj and mlp_out
        # 定义第二个线性层，用于合并注意力输出和 MLP 输出
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        # 定义 Q 和 K 的归一化层
        self.norm = QKNorm(head_dim)

        # 保存隐藏层大小
        self.hidden_size = hidden_size
        # 定义预归一化层，使用 LayerNorm，不使用仿射变换，epsilon 设置为 1e-6
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # 定义 MLP 的激活函数，使用近似为 tanh 的 GELU
        self.mlp_act = nn.GELU(approximate="tanh")
        # 定义调制模块，不使用双调制
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        """
        前向传播方法。

        对输入特征进行调制处理，通过多头自注意力机制和 MLP 进行处理，并结合调制参数生成输出。

        参数:
            x (Tensor): 输入特征张量，形状为 (B, L, hidden_size)。
            vec (Tensor): 辅助向量张量，形状为 (B, L, hidden_size)。
            pe (Tensor): 位置编码张量，形状为 (B, L, hidden_size)。

        返回:
            Tensor: 处理后的输出特征张量，形状为 (B, L, hidden_size)。
        """
        # 对辅助向量进行调制处理，得到调制参数
        mod, _ = self.modulation(vec)

        # 对输入特征进行预归一化处理，并应用调制参数
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift  # (B, L, hidden_size)

        # 通过第一个线性层生成 Q, K, V 和 MLP 输入
        # (B, L, 3 * hidden_size) 和 (B, L, mlp_hidden_dim)
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        # 重塑 Q, K, V 张量以适应多头自注意力机制
        # (3, B, num_heads, L, head_dim)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)

        # 对 Q 和 K 进行归一化处理
        # (3, B, num_heads, L, head_dim)
        q, k = self.norm(q, k, v)

        # compute attention
        # 计算自注意力机制的输出
        # (B, L, hidden_size)
        attn = attention(q, k, v, pe=pe)

        # compute activation in mlp stream, cat again and run second linear layer
        # 计算 MLP 流的激活函数，并拼接注意力输出和 MLP 输出，然后通过第二个线性层
        # (B, L, hidden_size)
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))

        # 将调制参数的门控值与输出相乘，并加到原始输入上，实现残差连接
        # (B, L, hidden_size)
        return x + mod.gate * output


class LastLayer(nn.Module):
    """
    最后一层（LastLayer）类，用于将 Transformer 块的输出映射到最终的输出通道数。

    该模块通过自适应层归一化（adaLN）和线性变换将输入特征映射到目标输出维度。

    参数:
        hidden_size (int): 隐藏层的维度。
        patch_size (int): 图像块的尺寸，用于控制输出特征图的尺寸。
        out_channels (int): 输出图像的通道数。
    """
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        # 定义最终归一化层，使用 LayerNorm，不使用仿射变换，epsilon 设置为 1e-6
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # 定义线性变换层，将隐藏层映射到输出图像的通道数
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        # 定义自适应层归一化调制模块，使用 SiLU 激活函数和线性变换生成移位和缩放参数
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        """
        前向传播方法。

        对输入特征进行自适应层归一化调制，并通过线性变换生成最终输出。

        参数:
            x (Tensor): 输入特征张量，形状为 (B, L, hidden_size)。
            vec (Tensor): 辅助向量张量，形状为 (B, L, hidden_size)。

        返回:
            Tensor: 最终输出张量，形状为 (B, L, patch_size^2 * out_channels)。
        """
        # 通过自适应层归一化调制模块生成移位和缩放参数
        # (B, L, hidden_size) 和 (B, L, hidden_size)
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)

        # 对输入特征进行自适应层归一化调制
        # (B, L, hidden_size)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]

        # 通过线性变换生成最终输出
        # (B, L, patch_size^2 * out_channels)
        x = self.linear(x)
        return x

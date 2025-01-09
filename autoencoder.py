from dataclasses import dataclass
import torch
from einops import rearrange
from torch import Tensor, nn


@dataclass
class AutoEncoderParams:
    """
    AutoEncoderParams 数据类用于存储自动编码器模型的各种参数配置。

    属性:
        resolution (int): 输入图像的分辨率（高度和宽度）。例如，256 表示 256x256 的图像。
        in_channels (int): 输入图像的通道数。通常，彩色图像为 3（RGB），灰度图像为 1。
        ch (int): 初始的隐藏通道数，用于控制模型中特征图的数量。
        out_ch (int): 输出图像的通道数，通常与 `in_channels` 相同。
        ch_mult (List[int]): 通道数乘数列表，用于在每个下采样或上采样阶段增加通道数。
                             例如，[1, 2, 4, 8] 表示每个阶段通道数依次乘以 1, 2, 4, 8。
        num_res_blocks (int): 每个阶段中残差块的数目，用于增强模型的表达能力。
        z_channels (int): 潜在空间（latent space）的通道数，用于表示编码器的输出。
        scale_factor (float): 缩放因子，用于控制潜在空间的标准差。
        shift_factor (float): 平移因子，用于控制潜在空间的均值。
    """
    resolution: int
    in_channels: int
    ch: int
    out_ch: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float


def swish(x: Tensor) -> Tensor:
    """
    Swish 激活函数。

    Swish 是一种自门控的激活函数，定义为：f(x) = x * sigmoid(x)。
    它在深层神经网络中表现出色，能够帮助模型更好地捕捉非线性关系。

    参数:
        x (Tensor): 输入张量。

    返回:
        Tensor: 经过 Swish 激活函数处理后的输出张量。
    """
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    """
    自注意力块（Attention Block），用于捕捉输入特征图的空间关系。

    该模块通过自注意力机制计算每个位置的特征表示与所有其他位置的关系，
    从而增强模型的全局感知能力。

    参数:
        in_channels (int): 输入特征图的通道数。
    """
    def __init__(self, in_channels: int):
        super().__init__()
        # 保存输入通道数
        self.in_channels = in_channels

        # 定义组归一化层，对输入特征图进行归一化处理
        # num_groups=32 表示将通道分成 32 个组进行归一化
        # eps=1e-6 用于防止除零
        # affine=True 表示使用可学习的仿射变换参数
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        # 定义查询（Query）卷积层，将输入特征图映射到查询空间
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # 定义键（Key）卷积层，将输入特征图映射到键空间
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # 定义值（Value）卷积层，将输入特征图映射到值空间
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # 定义输出投影卷积层，将注意力机制的输出映射回原始特征空间
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        """
        自注意力机制的实现。

        参数:
            h_ (Tensor): 输入特征图，形状为 (B, C, H, W)。

        返回:
            Tensor: 注意力机制的输出，形状为 (B, C, H, W)。
        """
        # 对输入特征图进行归一化处理
        h_ = self.norm(h_)

        # 通过查询、键和值卷积层得到相应的特征表示
        q = self.q(h_)  # (B, C, H, W)
        k = self.k(h_)  # (B, C, H, W)
        v = self.v(h_)  # (B, C, H, W)

        # 获取张量的维度
        b, c, h, w = q.shape

        # 重塑查询、键和值张量以适应注意力机制
        # 将 (B, C, H, W) 重塑为 (B, 1, H*W, C)
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()

        # 计算缩放点积注意力
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        # 将输出重塑回原始形状 (B, C, H, W)
        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播方法。

        将输入特征图通过自注意力机制进行处理，并将其与原始输入相加，实现残差连接。

        参数:
            x (Tensor): 输入特征图，形状为 (B, C, H, W)。

        返回:
            Tensor: 自注意力块的输出，形状为 (B, C, H, W)。
        """
        # 将输入特征图通过自注意力机制处理后，与原始输入相加，实现残差连接
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    """
    ResNet 残差块（ResnetBlock），用于构建深层神经网络。

    残差块通过引入跳跃连接（skip connection），使得梯度能够更有效地传播，
    从而缓解深层网络中的梯度消失问题。

    参数:
        in_channels (int): 输入特征的通道数。
        out_channels (Optional[int]): 输出特征的通道数。如果为 None，则输出通道数与输入通道数相同。
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # 保存输入通道数
        self.in_channels = in_channels
        # 如果输出通道数未指定，则默认为输入通道数
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        # 定义第一个组归一化层，对输入特征进行归一化处理
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        # 定义第一个卷积层，将输入特征映射到输出通道数
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # 定义第二个组归一化层，对第一个卷积层的输出进行归一化处理
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        # 定义第二个卷积层，将第一个卷积层的输出映射回输出通道数
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # 如果输入和输出通道数不同，则定义一个 1x1 卷积层进行通道数的匹配
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        前向传播方法。

        通过两个卷积层和跳跃连接实现残差块的前向传播。

        参数:
            x (Tensor): 输入张量，形状为 (N, in_channels, H, W)。

        返回:
            Tensor: 输出张量，形状为 (N, out_channels, H, W)。
        """
        # 保存输入张量作为跳跃连接
        h = x

        # 通过第一个组归一化层和 Swish 激活函数
        h = self.norm1(h)
        h = swish(h)
        # 通过第一个卷积层
        h = self.conv1(h)

        # 通过第二个组归一化层和 Swish 激活函数
        h = self.norm2(h)
        h = swish(h)
        # 通过第二个卷积层
        h = self.conv2(h)

        # 如果输入和输出通道数不同，则通过 1x1 卷积层进行通道数的匹配
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        # 将跳跃连接与卷积层的输出相加，实现残差连接
        return x + h


class Downsample(nn.Module):
    """
    下采样模块，用于在神经网络中降低特征图的空间分辨率。

    该模块通过步幅为 2 的卷积层实现下采样，同时保持通道数不变。

    参数:
        in_channels (int): 输入特征的通道数。
    """
    def __init__(self, in_channels: int):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        # 由于 PyTorch 的卷积层不支持非对称填充，因此需要手动进行填充
        # 使用 3x3 卷积核，步幅为 2，实现下采样
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        """
        前向传播方法。

        对输入张量进行填充后，通过卷积层实现下采样。

        参数:
            x (Tensor): 输入张量，形状为 (N, in_channels, H, W)。

        返回:
            Tensor: 输出张量，形状为 (N, in_channels, H/2, W/2)。
        """
        # 对输入张量进行填充，填充宽度为 (左, 右, 上, 下) = (0, 1, 0, 1)
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        # 通过卷积层实现下采样
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    """
    上采样模块，用于在神经网络中提高特征图的空间分辨率。

    该模块首先使用最近邻插值进行上采样，然后通过卷积层进行特征融合。

    参数:
        in_channels (int): 输入特征的通道数。
    """
    def __init__(self, in_channels: int):
        super().__init__()
        # 定义卷积层，用于融合上采样后的特征
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        """
        前向传播方法。

        对输入张量进行最近邻插值上采样后，通过卷积层进行特征融合。

        参数:
            x (Tensor): 输入张量，形状为 (N, in_channels, H, W)。

        返回:
            Tensor: 输出张量，形状为 (N, in_channels, 2H, 2W)。
        """
        # 使用最近邻插值将特征图的空间分辨率提高 2 倍
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        # 通过卷积层进行特征融合
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    """
    编码器（Encoder）类，用于将输入图像编码到潜在空间（latent space）。

    该编码器采用多分辨率的架构，通过一系列的卷积、残差块和下采样操作，
    将输入图像逐步压缩到低维的潜在表示。

    参数:
        resolution (int): 输入图像的分辨率（高度和宽度）。
        in_channels (int): 输入图像的通道数。例如，彩色图像通常为 3（RGB）。
        ch (int): 初始的隐藏通道数，用于控制模型中特征图的数量。
        ch_mult (List[int]): 通道数乘数列表，用于在每个下采样阶段增加通道数。
                             例如，[1, 2, 4] 表示每个阶段通道数依次乘以 1, 2, 4。
        num_res_blocks (int): 每个分辨率阶段中残差块的数目，用于增强模型的表达能力。
        z_channels (int): 潜在空间（latent space）的通道数，用于表示编码器的输出。
    """
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        # 保存初始的隐藏通道数
        self.ch = ch
        # 计算分辨率阶段的数目
        self.num_resolutions = len(ch_mult)
        # 保存残差块的数目
        self.num_res_blocks = num_res_blocks
        # 保存输入图像的分辨率
        self.resolution = resolution
        # 保存输入图像的通道数
        self.in_channels = in_channels

        # downsampling
        # 下采样阶段
        # 定义输入卷积层，将输入图像映射到初始隐藏通道数
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        # 计算每个分辨率阶段的输入通道数乘数
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult

        # 定义下采样模块列表
        self.down = nn.ModuleList()
        # 初始化当前块的输入通道数
        block_in = self.ch

        # 遍历每个分辨率阶段
        for i_level in range(self.num_resolutions):
            # 定义当前分辨率阶段的残差块列表
            block = nn.ModuleList()
            # 定义当前分辨率阶段的注意力块列表（当前实现中未使用）
            attn = nn.ModuleList()
            # 计算当前块的输入和输出通道数
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            # 遍历每个残差块
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out

            # 定义当前分辨率阶段的下采样模块
            down = nn.Module()
            down.block = block
            down.attn = attn

            # 如果不是最后一个分辨率阶段，则添加下采样操作
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2  # 更新当前分辨率
            # 将当前阶段的下采样模块添加到下采样模块列表中
            self.down.append(down)

        # middle
        # 中间阶段
        self.mid = nn.Module()
        # 定义中间阶段的第一个残差块
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        # 定义中间阶段的注意力块
        self.mid.attn_1 = AttnBlock(block_in)
        # 定义中间阶段的第二个残差块
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        # 输出阶段
        # 定义输出归一化层
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        # 定义输出卷积层，将特征图映射到潜在空间
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播方法。

        将输入图像通过编码器网络，生成潜在空间的表示。

        参数:
            x (Tensor): 输入图像张量，形状为 (N, in_channels, H, W)。

        返回:
            Tensor: 编码器的输出张量，形状为 (N, 2 * z_channels, H_enc, W_enc)。
        """
        # 下采样阶段
        # 将输入图像通过输入卷积层
        # 存储每个阶段的输出
        hs = [self.conv_in(x)]

        # 遍历每个分辨率阶段
        for i_level in range(self.num_resolutions):
            # 遍历每个残差块
            for i_block in range(self.num_res_blocks):
                # 通过残差块处理
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    # 如果有注意力块，则通过注意力块处理
                    h = self.down[i_level].attn[i_block](h)
                # 将当前块的输出添加到列表中
                hs.append(h)

            # 如果不是最后一个分辨率阶段，则进行下采样
            if i_level != self.num_resolutions - 1:
                # 通过下采样模块处理
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        # 中间阶段
        # 获取最后一个阶段的输出
        h = hs[-1]
        # 通过第一个残差块
        h = self.mid.block_1(h)
        # 通过注意力块
        h = self.mid.attn_1(h)
        # 通过第二个残差块
        h = self.mid.block_2(h)

        # end
        # 输出阶段
        # 通过归一化层
        h = self.norm_out(h)
        # 通过 Swish 激活函数
        h = swish(h)
        # 通过输出卷积层，生成最终输出
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    """
    解码器（Decoder）类，用于将潜在空间的表示解码回原始图像。

    该解码器采用多分辨率的架构，通过一系列的上采样、残差块和卷积操作，
    将潜在空间的表示逐步恢复为高分辨率的图像。

    参数:
        ch (int): 初始的隐藏通道数，用于控制模型中特征图的数量。
        out_ch (int): 输出图像的通道数。例如，彩色图像通常为 3（RGB）。
        ch_mult (List[int]): 通道数乘数列表，用于在每个上采样阶段增加通道数。
                             例如，[1, 2, 4] 表示每个阶段通道数依次乘以 1, 2, 4。
        num_res_blocks (int): 每个分辨率阶段中残差块的数目，用于增强模型的表达能力。
        in_channels (int): 输入潜在空间的通道数。
        resolution (int): 输出图像的分辨率（高度和宽度）。
        z_channels (int): 输入潜在空间的通道数。
    """
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ):
        super().__init__()
        # 保存初始的隐藏通道数
        self.ch = ch
        # 计算分辨率阶段的数目
        self.num_resolutions = len(ch_mult)
        # 保存残差块的数目
        self.num_res_blocks = num_res_blocks
        # 保存输出图像的分辨率
        self.resolution = resolution
        # 保存输入潜在空间的通道数
        self.in_channels = in_channels
        # 计算上采样因子
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # compute in_ch_mult, block_in and curr_res at lowest res
        # 计算最低分辨率阶段的通道数和分辨率
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        # 潜在空间的形状
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        # 定义从潜在空间到特征图的卷积层
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        # 中间阶段
        self.mid = nn.Module()
        # 定义中间阶段的第一个残差块
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        # 定义中间阶段的注意力块
        self.mid.attn_1 = AttnBlock(block_in)
        # 定义中间阶段的第二个残差块
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        # 上采样阶段
        self.up = nn.ModuleList()
        # 逆序遍历每个分辨率阶段
        for i_level in reversed(range(self.num_resolutions)):
            # 定义当前分辨率阶段的残差块列表
            block = nn.ModuleList()
            # 定义当前分辨率阶段的注意力块列表（当前实现中未使用）
            attn = nn.ModuleList()

            # 计算当前块的输出通道数
            block_out = ch * ch_mult[i_level]
            # 遍历每个残差块（包括一个额外的残差块）
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            # 定义当前分辨率阶段的上采样模块
            up = nn.Module()
            up.block = block
            up.attn = attn

            # 如果不是第一个分辨率阶段，则添加上采样操作
            if i_level != 0:
                up.upsample = Upsample(block_in)
                # 更新当前分辨率
                curr_res = curr_res * 2 
            # 将当前阶段的上采样模块插入到上采样模块列表的前面，以保持顺序一致
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        # 输出阶段
        # 定义输出归一化层
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        # 定义输出卷积层，将特征图映射到输出图像
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        """
        前向传播方法。

        将潜在空间的表示通过解码器网络，解码回原始图像。

        参数:
            z (Tensor): 输入潜在空间张量，形状为 (N, z_channels, H_z, W_z)。

        返回:
            Tensor: 解码器的输出图像张量，形状为 (N, out_ch, resolution, resolution)。
        """
        # 将潜在空间的表示通过输入卷积层，映射到特征图
        # z to block_in
        h = self.conv_in(z)

        # middle
        # 中间阶段
        # 通过第一个残差块
        h = self.mid.block_1(h)
        # 通过注意力块
        h = self.mid.attn_1(h)
        # 通过第二个残差块
        h = self.mid.block_2(h)

        # upsampling
        # 上采样阶段
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                # 通过残差块处理
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    # 如果有注意力块，则通过注意力块处理
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                # 通过上采样模块处理
                h = self.up[i_level].upsample(h)

        # end
        # 输出阶段
        # 通过归一化层
        h = self.norm_out(h)
        # 通过 Swish 激活函数
        h = swish(h)
        # 通过输出卷积层，生成最终输出
        h = self.conv_out(h)
        return h


class DiagonalGaussian(nn.Module):
    """
    对角高斯分布（Diagonal Gaussian）模块，用于从潜在空间生成样本。

    该模块假设潜在空间中的每个维度都是独立的，并且服从高斯分布。
    通过对输入的均值和对数方差进行操作，可以生成样本或仅返回均值。

    参数:
        sample (bool, optional): 是否生成样本。如果为 True，则从高斯分布中采样；否则，仅返回均值。默认为 True。
        chunk_dim (int, optional): 分割维度的索引，用于将输入张量分割为均值和对数方差。默认为 1。
    """
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        # 是否生成样本的标志
        self.sample = sample
        # 用于分割输入张量的维度索引
        self.chunk_dim = chunk_dim

    def forward(self, z: Tensor) -> Tensor:
        """
        前向传播方法。

        将输入张量分割为均值和对数方差，并根据参数决定是否生成样本。

        参数:
            z (Tensor): 输入张量，假设其形状为 (N, 2 * latent_dim, ...)。

        返回:
            Tensor: 如果 sample 为 True，则返回从高斯分布中采样的样本；否则，返回均值。
        """
        # 将输入张量沿指定维度分割为均值和对数方差
        # 假设输入张量的形状为 (N, 2 * latent_dim, ...)，分割后每个部分的形状为 (N, latent_dim, ...)
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            # 计算标准差
            std = torch.exp(0.5 * logvar)
            # 从标准正态分布中生成与 mean 形状相同的随机噪声
            # 生成样本：mean + std * noise
            return mean + std * torch.randn_like(mean)
        else:
            # 仅返回均值
            return mean


class AutoEncoder(nn.Module):
    """
    自动编码器（AutoEncoder）类，结合编码器、解码器和正则化模块，实现数据的压缩和重构。

    该自动编码器使用指定的编码器和解码器架构，并通过对角高斯分布对潜在空间进行正则化处理。
    通过缩放和平移因子，可以对潜在空间进行进一步的调整。

    参数:
        params (AutoEncoderParams): 自动编码器的参数配置，包含分辨率、通道数等超参数。
    """
    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        # 初始化编码器
        self.encoder = Encoder(
            resolution=params.resolution,      # 输入图像的分辨率
            in_channels=params.in_channels,    # 输入图像的通道数
            ch=params.ch,                      # 初始的隐藏通道数
            ch_mult=params.ch_mult,            # 通道数乘数列表
            num_res_blocks=params.num_res_blocks,  # 每个分辨率阶段中残差块的数目
            z_channels=params.z_channels,      # 潜在空间的通道数
        )

        # 初始化解码器
        self.decoder = Decoder(
            resolution=params.resolution,      # 输出图像的分辨率
            in_channels=params.in_channels,    # 输入图像的通道数
            ch=params.ch,                      # 初始的隐藏通道数
            out_ch=params.out_ch,              # 输出图像的通道数
            ch_mult=params.ch_mult,            # 通道数乘数列表
            num_res_blocks=params.num_res_blocks,  # 每个分辨率阶段中残差块的数目
            z_channels=params.z_channels,      # 潜在空间的通道数
        )

        # 初始化对角高斯分布模块，用于对潜在空间进行正则化处理
        self.reg = DiagonalGaussian()

        # 初始化缩放和平移因子，用于调整潜在空间
        self.scale_factor = params.scale_factor
        self.shift_factor = params.shift_factor

    def encode(self, x: Tensor) -> Tensor:
        """
        编码方法。

        将输入数据通过编码器编码到潜在空间，并对潜在空间进行缩放和平移调整。

        参数:
            x (Tensor): 输入数据张量。

        返回:
            Tensor: 调整后的潜在空间表示。
        """
        # 通过编码器将输入数据编码到潜在空间
        # 通过对角高斯分布模块处理潜在空间
        z = self.reg(self.encoder(x))
        # 对潜在空间进行缩放和平移调整
        z = self.scale_factor * (z - self.shift_factor)
        return z

    def decode(self, z: Tensor) -> Tensor:
        """
        解码方法。

        将潜在空间的表示通过解码器解码回原始数据空间，并对潜在空间进行逆缩放和平移调整。

        参数:
            z (Tensor): 潜在空间的表示。

        返回:
            Tensor: 解码后的输出数据张量。
        """
        # 对潜在空间进行逆缩放和平移调整
        z = z / self.scale_factor + self.shift_factor
        # 通过解码器将潜在空间解码回原始数据空间
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播方法。

        将输入数据通过编码器编码到潜在空间，再通过解码器解码回原始数据空间，实现自动编码过程。

        参数:
            x (Tensor): 输入数据张量。

        返回:
            Tensor: 自动编码器的重构输出。
        """
        # 先编码再解码，实现自动编码过程
        return self.decode(self.encode(x))

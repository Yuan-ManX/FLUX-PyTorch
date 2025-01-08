from dataclasses import dataclass
import torch
from torch import Tensor, nn

from layers import DoubleStreamBlock, EmbedND, LastLayer, MLPEmbedder, SingleStreamBlock, timestep_embedding
from lora import LinearLora, replace_linear_with_lora


@dataclass
class FluxParams:
    """
    Flux 模型所需的参数配置。

    该数据类包含了构建 Flux 模型所需的所有超参数配置。

    属性:
        in_channels (int): 输入图像的通道数。
        out_channels (int): 输出图像的通道数。
        vec_in_dim (int): 辅助向量的输入维度。
        context_in_dim (int): 文本输入的上下文维度。
        hidden_size (int): Transformer 模型的隐藏层大小。
        mlp_ratio (float): MLP 层中隐藏层大小的比例，用于计算 MLP 的中间层大小。
        num_heads (int): 多头自注意力机制中的头数。
        depth (int): 双流 Transformer 块的层数。
        depth_single_blocks (int): 单流 Transformer 块的层数。
        axes_dim (List[int]): 位置编码中各个轴的维度列表。
        theta (int): 位置编码中的角度参数。
        qkv_bias (bool): 在查询、键和值投影中是否使用偏置。
        guidance_embed (bool): 是否启用引导信息。
    """
    in_channels: int
    out_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """
    """
    Transformer 模型用于序列上的流匹配（Flow Matching）。

    该模型旨在处理包含图像、文本、时间步长和辅助信息（如引导）的多模态数据。
    通过多层 Transformer 块，模型能够捕捉不同模态之间的复杂关系，并生成最终的输出。

    参数:
        params (FluxParams): 模型参数，包含模型配置的各种超参数。
    """

    def __init__(self, params: FluxParams):
        super().__init__()

        # 保存参数以便后续使用
        self.params = params

        # 输入和输出的通道数
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels

        # 检查隐藏层大小是否可以被头数整除
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        
        # 计算每个头部的位置编码维度
        pe_dim = params.hidden_size // params.num_heads

        # 检查位置编码维度是否与轴维度之和匹配
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        
        # 初始化模型的其他超参数
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads

        # 初始化位置编码嵌入器
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)

        # 初始化图像输入线性层，将输入图像特征映射到隐藏层大小
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)

        # 初始化时间步长嵌入器，使用 MLP 将时间步长嵌入到隐藏层大小
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)

        # 初始化向量输入嵌入器，将辅助向量特征映射到隐藏层大小
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)

        # 初始化引导嵌入器（可选），如果启用引导则使用 MLP，否则使用恒等映射
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )

        # 初始化文本输入线性层，将输入文本特征映射到隐藏层大小
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        # 初始化双流 Transformer 块列表，根据深度参数重复堆叠
        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        # 初始化单流 Transformer 块列表，根据单流块深度参数重复堆叠
        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        # 初始化最后一层，将 Transformer 块的输出映射到最终的输出通道数
        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
    ) -> Tensor:
        """
        前向传播方法。

        参数:
            img (Tensor): 输入图像张量，形状为 (N, C, T)。
            img_ids (Tensor): 输入图像的标识符张量，形状为 (N, T)。
            txt (Tensor): 输入文本张量，形状为 (N, L, D_txt)。
            txt_ids (Tensor): 输入文本的标识符张量，形状为 (N, L)。
            timesteps (Tensor): 时间步长张量，形状为 (N,)。
            y (Tensor): 辅助向量张量，形状为 (N, D_y)。
            guidance (Optional[Tensor]): 可选的引导张量，形状为 (N,)。如果启用引导，则必须提供。

        返回:
            Tensor: 模型的输出，形状为 (N, T, patch_size ** 2 * out_channels)。
        """

        # 检查输入张量的维度
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        # 对图像输入进行线性变换，映射到隐藏层大小
        img = self.img_in(img) # (N, C, T) -> (N, T, hidden_size)

        # 对时间步长进行嵌入，使用时间步长嵌入函数
        vec = self.time_in(timestep_embedding(timesteps, 256)) # (N, hidden_size)

        # 如果启用了引导嵌入，则将引导信息嵌入并添加到 vec 中
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256)) # (N, hidden_size)

        # 将辅助向量 y 嵌入并添加到 vec 中
        vec = vec + self.vector_in(y) # (N, hidden_size)

        # 对文本输入进行线性变换，映射到隐藏层大小
        txt = self.txt_in(txt) # (N, L, D_txt) -> (N, L, hidden_size)

        # 将文本和图像的标识符拼接起来，用于位置编码
        ids = torch.cat((txt_ids, img_ids), dim=1) # (N, T + L)

        # 生成位置编码
        pe = self.pe_embedder(ids) # (N, T + L, pe_dim)

        # 通过所有双流 Transformer 块
        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe) # img: (N, T, hidden_size), txt: (N, L, hidden_size)

        # 将文本和图像特征在通道维度上拼接
        img = torch.cat((txt, img), 1) # (N, T + L, hidden_size)

        # 通过所有单流 Transformer 块
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe) # (N, T + L, hidden_size)

        # 去除文本部分，只保留图像部分
        img = img[:, txt.shape[1] :, ...] # (N, T, hidden_size)

        # 通过最后一层，生成最终输出
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img


class FluxLoraWrapper(Flux):
    """
    Flux 模型的可训练低秩自适应（LoRA）包装器。

    通过在 Flux 模型的基础上应用 LoRA 技术，可以在保持原始模型参数不变的情况下，
    通过添加低秩矩阵来微调模型，从而减少训练参数量并加速训练过程。

    参数:
        lora_rank (int): LoRA 矩阵的秩，决定了低秩近似的程度。默认值为 128。
        lora_scale (float): LoRA 矩阵的缩放因子，用于控制低秩矩阵的贡献。默认值为 1.0。
        *args: 传递给 Flux 基类的位置参数。
        **kwargs: 传递给 Flux 基类的关键字参数。
    """
    def __init__(
        self,
        lora_rank: int = 128,
        lora_scale: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        # 调用 Flux 基类的初始化方法，传递所有位置和关键字参数
        super().__init__(*args, **kwargs)

        # 保存 LoRA 矩阵的秩
        self.lora_rank = lora_rank

        # 替换模型中的所有线性层为带有 LoRA 的线性层
        replace_linear_with_lora(
            self,  # 要修改的模型实例
            max_rank=lora_rank,  # LoRA 矩阵的最大秩
            scale=lora_scale,  # LoRA 矩阵的缩放因子
        )

    def set_lora_scale(self, scale: float) -> None:
        """
        设置 LoRA 矩阵的缩放因子。

        该方法遍历模型的所有子模块，找到所有 LinearLora 类型的模块，并设置其缩放因子。

        参数:
            scale (float): 新的缩放因子值。
        """
        # 遍历模型的所有子模块
        for module in self.modules():
            # 检查当前模块是否是 LinearLora 类型
            if isinstance(module, LinearLora):
                # 设置 LinearLora 模块的缩放因子
                module.set_scale(scale=scale)

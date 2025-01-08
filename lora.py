import torch
from torch import nn


def replace_linear_with_lora(
    module: nn.Module,
    max_rank: int,
    scale: float = 1.0,
) -> None:
    """
    递归遍历给定的 PyTorch 模块及其子模块，将所有 nn.Linear 层替换为带有 LoRA 的 LinearLora 层。

    LoRA（Low-Rank Adaptation）是一种用于高效微调大型模型的技术，通过在预训练模型的基础上添加低秩矩阵，
    可以在保持原始模型参数不变的情况下，减少训练参数量并加速训练过程。

    参数:
        module (nn.Module): 要进行替换操作的 PyTorch 模块。
        max_rank (int): LoRA 矩阵的最大秩，决定了低秩近似的程度。
        scale (float, optional): LoRA 矩阵的缩放因子，用于调节低秩矩阵的贡献。默认值为 1.0。
    """
    # 遍历模块的所有直接子模块及其名称
    for name, child in module.named_children():
        # 检查当前子模块是否是 nn.Linear 类型的线性层
        if isinstance(child, nn.Linear):
            # 创建一个新的 LinearLora 层，复制原线性层的属性
            new_lora = LinearLora(
                in_features=child.in_features,    # 输入特征的维度
                out_features=child.out_features,  # 输出特征的维度
                bias=child.bias,                  # 是否使用偏置
                rank=max_rank,                    # LoRA 矩阵的秩
                scale=scale,                      # LoRA 矩阵的缩放因子
                dtype=child.weight.dtype,         # 权重张量的数据类型
                device=child.weight.device,       # 权重张量的设备（CPU/GPU）
            )

            # 将原线性层的权重赋值给新的 LinearLora 层
            new_lora.weight = child.weight
            # 如果原线性层有偏置，则将偏置也赋值给新的 LinearLora 层
            new_lora.bias = child.bias if child.bias is not None else None

            # 使用新的 LinearLora 层替换原线性层
            setattr(module, name, new_lora)
        else:
            # 如果当前子模块不是线性层，则递归调用 replace_linear_with_lora 函数，
            # 对子模块的子模块进行替换操作
            replace_linear_with_lora(
                module=child,        # 当前子模块
                max_rank=max_rank,   # 传递最大秩参数
                scale=scale,         # 传递缩放因子参数
            )


class LinearLora(nn.Linear):
    """
    LinearLora 类继承自 nn.Linear，添加了低秩自适应（LoRA）机制。

    LoRA 通过在原始线性层的基础上添加低秩矩阵来实现高效微调，从而减少训练参数量并加速训练过程。
    该类在前向传播过程中，将原始线性层的输出与 LoRA 矩阵的输出进行加和，实现低秩适应的效果。

    参数:
        in_features (int): 输入特征的维度。
        out_features (int): 输出特征的维度。
        bias (bool): 是否使用偏置。如果为 True，则添加偏置参数。
        rank (int): LoRA 矩阵的秩，决定了低秩近似的程度。
        dtype (torch.dtype): 参数的数据类型。
        device (torch.device): 参数所在的设备（CPU 或 GPU）。
        lora_bias (bool, optional): LoRA 矩阵的偏置标志。默认为 True，表示使用偏置。
        scale (float, optional): LoRA 矩阵的缩放因子，用于调节低秩矩阵的贡献。默认为 1.0。
        *args: 传递给 nn.Linear 基类的位置参数。
        **kwargs: 传递给 nn.Linear 基类的关键字参数。
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        rank: int,
        dtype: torch.dtype,
        device: torch.device,
        lora_bias: bool = True,
        scale: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        # 调用 nn.Linear 基类的初始化方法
        super().__init__(
            in_features=in_features,         # 传递输入特征维度
            out_features=out_features,       # 传递输出特征维度
            bias=bias is not None,           # 设置是否使用偏置
            device=device,                   # 传递设备信息
            dtype=dtype,                     # 传递数据类型
            *args,                           # 传递其他位置参数
            **kwargs,                        # 传递其他关键字参数
        )

        # 确保 scale 是一个浮点数
        assert isinstance(scale, float), "scale must be a float"

        # 保存缩放因子
        self.scale = scale
        # 保存 LoRA 矩阵的秩
        self.rank = rank
        # 保存 LoRA 矩阵的偏置标志
        self.lora_bias = lora_bias
        # 保存数据类型和设备信息
        self.dtype = dtype
        self.device = device

        # 计算 LoRA 矩阵的最大允许秩，避免秩超过输入或输出特征的最小维度
        if rank > (new_rank := min(self.out_features, self.in_features)):
            self.rank = new_rank

        # 初始化 LoRA 矩阵 A，形状为 (in_features, rank)
        self.lora_A = nn.Linear(
            in_features=in_features,          # 输入特征维度
            out_features=self.rank,           # 输出特征维度为 LoRA 矩阵的秩
            bias=False,                       # 不使用偏置
            dtype=dtype,                      # 数据类型与主线性层一致
            device=device,                    # 设备与主线性层一致
        )

        # 初始化 LoRA 矩阵 B，形状为 (rank, out_features)
        self.lora_B = nn.Linear(
            in_features=self.rank,            # 输入特征维度为 LoRA 矩阵的秩
            out_features=out_features,        # 输出特征维度与主线性层一致
            bias=self.lora_bias,              # 使用 LoRA 矩阵的偏置标志
            dtype=dtype,                      # 数据类型与主线性层一致
            device=device,                    # 设备与主线性层一致
        )

    def set_scale(self, scale: float) -> None:
        """
        设置 LoRA 矩阵的缩放因子。

        参数:
            scale (float): 新的缩放因子值。
        """
        # 确保 scale 是一个浮点数
        assert isinstance(scale, float), "scalar value must be a float"
        # 更新缩放因子
        self.scale = scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        前向传播方法。

        计算原始线性层的输出与 LoRA 矩阵的输出的加和。

        参数:
            input (Tensor): 输入张量，形状为 (N, in_features)。

        返回:
            Tensor: 输出张量，形状为 (N, out_features)。
        """
        # 计算原始线性层的输出
        base_out = super().forward(input)  # (N, out_features)

        # 计算 LoRA 矩阵的输出
        _lora_out_B = self.lora_B(self.lora_A(input))  # (N, rank) -> (N, out_features)

        # 对 LoRA 矩阵的输出进行缩放
        lora_update = _lora_out_B * self.scale  # (N, out_features)

        # 将原始线性层的输出与 LoRA 矩阵的输出相加
        return base_out + lora_update  # (N, out_features)

from torch import Tensor, nn
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer


class HFEmbedder(nn.Module):
    """
    HFEmbedder 类用于使用 Hugging Face 的预训练模型对文本进行嵌入。

    该类支持两种类型的预训练模型：
    1. OpenAI 的 CLIP 模型，用于图像和文本的多模态嵌入。
    2. T5 模型，用于文本的编码和嵌入。

    参数:
        version (str): 预训练模型的版本或名称，例如 "openai/clip-vit-base-patch32" 或 "t5-base"。
        max_length (int): 文本的最大长度，用于截断或填充文本。
        **hf_kwargs: 其他传递给 Hugging Face 模型加载方法的关键字参数。
    """
    def __init__(self, version: str, max_length: int, **hf_kwargs):
        super().__init__()
        # 判断模型版本是否以 "openai" 开头，以确定是否使用 CLIP 模型
        self.is_clip = version.startswith("openai")
        # 保存文本的最大长度
        self.max_length = max_length
        # 根据模型类型设置输出的键名
        # CLIP 模型使用 "pooler_output" 作为输出，而 T5 模型使用 "last_hidden_state"
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            # 如果是 CLIP 模型，加载 CLIPTokenizer 和 CLIPTextModel
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(version, max_length=max_length)
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(version, **hf_kwargs)
        else:
            # 如果不是 CLIP 模型，加载 T5Tokenizer 和 T5EncoderModel
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(version, max_length=max_length)
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(version, **hf_kwargs)

        # 将模型设置为评估模式，并冻结其参数，防止在训练过程中更新
        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> Tensor:
        """
        前向传播方法，对输入的文本列表进行嵌入。

        参数:
            text (List[str]): 输入的文本列表，每个元素是一个字符串。

        返回:
            Tensor: 文本的嵌入表示，形状为 (N, hidden_size) 或 (N, sequence_length, hidden_size)。
        """
        # 使用分词器对文本进行编码
        batch_encoding = self.tokenizer(
            text,                             # 输入的文本列表
            truncation=True,                  # 启用截断，截断超过最大长度的文本
            max_length=self.max_length,       # 设置最大长度
            return_length=False,              # 不返回长度信息
            return_overflowing_tokens=False,  # 不返回超出最大长度的部分
            padding="max_length",             # 对文本进行填充，使其长度相同
            return_tensors="pt",              # 返回 PyTorch 张量
        )

        # 如果是 CLIP 模型，则不需要提供 attention_mask，因为 CLIP 的 tokenizer 已经处理了填充
        # 如果是 T5 模型，则可以传递 attention_mask
        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )

        # 返回指定键的输出
        # 对于 CLIP 模型，返回 "pooler_output"；对于 T5 模型，返回 "last_hidden_state"
        return outputs[self.output_key]

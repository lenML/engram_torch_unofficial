from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EngramConfig:
    # 基础配置
    hidden_size: int = 1024  # 主干模型的 hidden_dim
    engram_vocab_size: int = 129280  # 原始 Tokenizer 的词表大小

    # Engram 核心参数
    max_ngram_size: int = 3  # 最大 N-gram (如 3-gram)
    # 这里的 vocab_size 是指 Engram 哈希表的槽位总数 (越大越好，但占内存)
    engram_table_size: List[int] = field(
        default_factory=lambda: [129280 * 5, 129280 * 5]
    )
    n_embed_per_ngram: int = 512  # Engram 内部查表得到的向量维度
    n_head_per_ngram: int = 8  # 多头哈希的头数

    # 结构参数
    kernel_size: int = 4  # 卷积核大小
    hc_mult: int = 1  # Dense 模型默认为 1，如果是 DeepSeek-V3 设为 4

    # 系统参数
    pad_id: int = 2
    seed: int = 42

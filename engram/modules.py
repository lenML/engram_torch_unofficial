import torch
import torch.nn as nn
import math
from typing import List, Optional
from .config import EngramConfig


class ScalableEmbedding(nn.Module):
    """支持 CPU Offload 的巨型 Embedding"""

    def __init__(self, sizes: List[int], dim: int):
        super().__init__()
        self.offsets = [0]
        for s in sizes[:-1]:
            self.offsets.append(self.offsets[-1] + s)
        self.register_buffer(
            "offsets_buf", torch.tensor(self.offsets, dtype=torch.long)
        )

        total_size = sum(sizes)
        self.embedding = nn.Embedding(total_size, dim)

    def forward(self, hash_ids: "torch.Tensor") -> "torch.Tensor":
        # hash_ids: [B, L, Num_Heads]
        device = hash_ids.device

        # 加上偏移量，对应到总表中的绝对位置
        flat_ids = hash_ids + self.offsets_buf.to(device)

        embed_device = self.embedding.weight.device
        flat_ids = flat_ids.to(embed_device)

        embeds_cpu: "torch.Tensor" = self.embedding(flat_ids)
        return embeds_cpu.to(device)


class EngramModule(nn.Module):
    def __init__(self, config: EngramConfig, layer_vocab_sizes: List[List[int]]):
        """
        layer_vocab_sizes: 来自 Tokenizer 计算出的该层所有 Head 的词表大小列表
        """
        super().__init__()
        self.cfg = config

        # 1. Embedding 层
        # 展平 layer_vocab_sizes (e.g. [[size_2gram_h1, ...], [size_3gram_h1...]])
        flat_sizes = [s for sublist in layer_vocab_sizes for s in sublist]
        self.embed_dim = config.n_embed_per_ngram // config.n_head_per_ngram

        self.memory = ScalableEmbedding(sizes=flat_sizes, dim=self.embed_dim)

        # Engram 拼接后的总维度
        total_engram_dim = len(flat_sizes) * self.embed_dim

        # 2. 投影层
        self.value_proj = nn.Linear(total_engram_dim, config.hidden_size)
        # 支持 Dense (hc_mult=1) 和 DeepSeek (hc_mult=4)
        self.key_projs = nn.ModuleList(
            [
                nn.Linear(total_engram_dim, config.hidden_size)
                for _ in range(config.hc_mult)
            ]
        )

        # 3. 门控归一化
        self.norm_key = nn.ModuleList(
            [nn.RMSNorm(config.hidden_size) for _ in range(config.hc_mult)]
        )
        self.norm_query = nn.ModuleList(
            [nn.RMSNorm(config.hidden_size) for _ in range(config.hc_mult)]
        )

        # 4. 短卷积 (Depthwise)
        # 输入通道 = hidden_size * hc_mult
        channels = config.hidden_size * config.hc_mult
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=config.kernel_size,
            groups=channels,
            padding=config.kernel_size - 1,
        )
        self.act = nn.SiLU()

        self.reset_parameters()

    def reset_parameters(self):
        """零初始化，保证插入后不影响原始模型"""
        nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        # 也可以初始化 Value Proj 为极小值
        nn.init.normal_(self.value_proj.weight, std=0.001)
        nn.init.zeros_(self.value_proj.bias)

    def forward(self, hidden_states: "torch.Tensor", hash_ids: "torch.Tensor"):
        """
        hidden_states: [B, L, D] (Dense) 或 [B, L, G, D] (DeepSeek)
        hash_ids: [B, L, Num_Heads] (预先计算好的)
        """
        B, L = hidden_states.shape[:2]

        # 1. 适配 Dense 模型的维度 [B, L, D] -> [B, L, 1, D]
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.unsqueeze(2)  # Add group dim

        assert hidden_states.shape[2] == self.cfg.hc_mult

        # 2. 查表 (Static Memory Lookup)
        # output: [B, L, Heads, EmbedDim] -> flatten -> [B, L, TotalDim]
        mem_embeds = self.memory(hash_ids).flatten(start_dim=-2)

        # 3. 门控计算 (Context-Aware Gating)
        value = self.value_proj(mem_embeds)  # [B, L, D]

        gated_values = []
        for i in range(self.cfg.hc_mult):
            # Key & Query
            k = self.norm_key[i](self.key_projs[i](mem_embeds))
            q = self.norm_query[i](hidden_states[:, :, i, :])

            # Dot Product
            score = (q * k).sum(dim=-1, keepdim=True) / math.sqrt(self.cfg.hidden_size)
            # 论文中的处理: abs().sqrt() * sign() 增强非线性
            gate = score.abs().clamp_min(1e-6).sqrt() * score.sign()
            gate = torch.sigmoid(gate)

            # Apply Gate to Value
            gated_values.append(gate * value)  # Value 是共享的

        # [B, L, G, D]
        out = torch.stack(gated_values, dim=2)

        # 4. 短卷积 (Short Conv)
        # Reshape for Conv1d: [B, L, G, D] -> [B, G*D, L]
        x_conv: "torch.Tensor" = out.flatten(start_dim=2).permute(0, 2, 1)
        # Causal Padding logic implies looking only back, usually handled by padding size
        # Simple implementation:
        x_conv = self.conv(x_conv)
        x_conv = x_conv[:, :, :L]  # Trim padding
        x_conv = self.act(x_conv)

        # Back to [B, L, G, D]
        out = out + x_conv.permute(0, 2, 1).view(B, L, self.cfg.hc_mult, -1)

        # 如果是 Dense 模型，squeeze 回去
        if self.cfg.hc_mult == 1:
            out = out.squeeze(2)

        return out

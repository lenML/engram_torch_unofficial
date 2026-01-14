import torch
import torch.nn as nn
from typing import List

from .embed import OffloadEmbedding


class MultiHeadEmbedding(nn.Module):
    def __init__(
        self,
        sizes: List[int],
        dim: int,
        use_offload: bool = False,
        offload_cache_lines: int = 1024,
    ):
        super().__init__()
        self.offsets = [0]
        for s in sizes[:-1]:
            self.offsets.append(self.offsets[-1] + s)
        self.register_buffer(
            "offsets_buf", torch.tensor(self.offsets, dtype=torch.long)
        )

        total_size = sum(sizes)
        if use_offload:
            self.embedding = OffloadEmbedding(
                num_embeddings=total_size,
                embedding_dim=dim,
                cache_lines=offload_cache_lines,
            )
        else:
            self.embedding = nn.Embedding(total_size, dim)

        self.sizes = sizes

    def forward(
        self,
        # hash_ids: [B, L, Num_Heads]
        hash_ids: "torch.Tensor",
    ) -> "torch.Tensor":
        flat_ids = hash_ids + self.offsets_buf
        embeds = self.embedding(flat_ids)
        return embeds

    def preload(self, hash_ids: torch.Tensor):
        if isinstance(self.embedding, OffloadEmbedding):
            flat_ids = hash_ids + self.offsets_buf
            self.embedding._load_missing(flat_ids)

import torch
import torch.nn as nn
from typing import List
from collections import OrderedDict


class MultiHeadEmbedding(nn.Module):
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

    def forward(
        self,
        # hash_ids: [B, L, Num_Heads]
        hash_ids: "torch.Tensor",
    ) -> "torch.Tensor":
        flat_ids = hash_ids + self.offsets_buf
        embeds = self.embedding(flat_ids)
        return embeds


class OffloadMultiHeadEmbedding(nn.Module):
    def __init__(
        self,
        sizes: List[int],
        dim: int,
        cache_size: int = -1,
        device: str = "cuda",
    ):
        super().__init__()

        if cache_size == -1:
            cache_size = sizes[0] // 2

        # offsets
        offsets = [0]
        for s in sizes[:-1]:
            offsets.append(offsets[-1] + s)
        self.register_buffer("offsets_buf", torch.tensor(offsets, dtype=torch.long))

        self.total_size = sum(sizes)
        self.dim = dim
        self.device = torch.device(device)

        # CPU embedding（主存）
        self.cpu_embedding = nn.Embedding(self.total_size, dim, device="cpu")

        # GPU cache
        self.cache_size = cache_size
        self.gpu_weight = torch.empty(cache_size, dim, device=self.device)

        # cache bookkeeping
        self.cache_index = {}  # global_id -> slot
        self.lru = OrderedDict()  # global_id -> None
        self.free_slots = list(range(cache_size))

    @torch.no_grad()
    def _evict_if_needed(self):
        if self.free_slots:
            return
        # LRU eviction
        evict_id, _ = self.lru.popitem(last=False)
        slot = self.cache_index.pop(evict_id)
        self.free_slots.append(slot)

    @torch.no_grad()
    def _load_to_cache(self, global_ids: torch.Tensor):
        """
        global_ids: 1D unique ids on CPU
        """
        for gid in global_ids.tolist():
            if gid in self.cache_index:
                # refresh LRU
                self.lru.move_to_end(gid)
                continue

            self._evict_if_needed()

            slot = self.free_slots.pop()
            self.gpu_weight[slot].copy_(self.cpu_embedding.weight[gid].to(self.device))

            self.cache_index[gid] = slot
            self.lru[gid] = None

    def preload(self, hash_ids: torch.Tensor):
        """
        显式 preload（例如 warmup）
        ids: 任意 shape 的 hash_ids（未加 offset）
        """
        flat = (hash_ids + self.offsets_buf).view(-1).unique().cpu()
        self._load_to_cache(flat)

    def forward(self, hash_ids: torch.Tensor) -> torch.Tensor:
        # hash_ids: [B, L, H]
        flat_ids = (hash_ids + self.offsets_buf).view(-1)

        unique_ids = flat_ids.unique()
        self._load_to_cache(unique_ids.cpu())

        # map global_id -> cache_slot
        slots = torch.empty_like(flat_ids)
        for i, gid in enumerate(flat_ids.tolist()):
            slot = self.cache_index[gid]
            slots[i] = slot
            self.lru.move_to_end(gid)

        embeds = self.gpu_weight[slots].view(*hash_ids.shape, self.dim)
        return embeds

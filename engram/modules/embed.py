import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Optional


class OffloadEmbedding(nn.Embedding):
    """
    Drop-in replacement for nn.Embedding with CPU Offloading and LRU Caching.
    Solves OOM issues for massive vocabularies.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        cache_lines: Optional[int] = None,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Optional[torch.Tensor] = None,
        device=None,
        dtype=None,
    ) -> None:
        # Force parent to initialize on CPU to save GPU memory
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            _weight,
            device="cpu",
            dtype=dtype,
        )

        self.device_target = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Configure Cache
        if cache_lines is None:
            cache_lines = int(num_embeddings * 0.1)
        self.cache_lines = min(cache_lines, num_embeddings)

        # 1. Master Weights on CPU (Pinned for speed)
        self.weight.requires_grad_(False)
        self.weight.data = self.weight.data.pin_memory()

        # 2. GPU Cache
        self.cache_weight = nn.Parameter(
            torch.zeros(
                (self.cache_lines, embedding_dim),
                dtype=dtype,
                device=self.device_target,
            ),
            requires_grad=False,
        )

        # 3. Map Table: global_id -> cache_slot. -1 means Miss.
        self.register_buffer(
            "map_table",
            torch.full(
                (num_embeddings,), -1, dtype=torch.long, device=self.device_target
            ),
        )

        # 4. LRU Tracker (CPU)
        self.lru_tracker = OrderedDict()
        self.free_slots = set(range(self.cache_lines))

    @classmethod
    def from_embedding(
        cls, emb: nn.Embedding, cache_lines: Optional[int] = None
    ) -> "OffloadEmbedding":
        obj = cls(
            num_embeddings=emb.num_embeddings,
            embedding_dim=emb.embedding_dim,
            cache_lines=cache_lines,
            padding_idx=emb.padding_idx,
            max_norm=emb.max_norm,
            norm_type=emb.norm_type,
            scale_grad_by_freq=emb.scale_grad_by_freq,
            sparse=emb.sparse,
            _weight=None,
            device=emb.weight.device if emb.weight.device.type == "cuda" else None,
            dtype=emb.weight.dtype,
        )
        # Copy weights
        with torch.no_grad():
            obj.weight.data.copy_(emb.weight.data.cpu())

        # Ensure device correctness
        if emb.weight.device.type == "cuda":
            obj.to(emb.weight.device)

        return obj

    def to(self, *args, **kwargs):
        """Override .to() to keep master weights on CPU."""
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )

        if device is not None and device.type == "cuda":
            self.device_target = device
            self.cache_weight.data = self.cache_weight.data.to(
                device, dtype=dtype, non_blocking=non_blocking
            )
            self.map_table = self.map_table.to(device, non_blocking=non_blocking)
            # self.weight stays on CPU
        else:
            super().to(*args, **kwargs)
        return self

    @torch.inference_mode()
    def _load_missing(self, flat_ids: torch.Tensor):
        """Logic to move embeddings from CPU to GPU Cache."""
        # 1. Check GPU Map
        slots = self.map_table[flat_ids]
        is_missing = slots == -1

        if not is_missing.any():
            return

        # 2. CPU Scheduling
        misses_cpu = flat_ids[is_missing].cpu().numpy()
        # Filter padding_idx (never load padding into cache)
        if self.padding_idx is not None:
            misses_cpu = misses_cpu[misses_cpu != self.padding_idx]
            if len(misses_cpu) == 0:
                return

        slots_to_use = []
        evict_ids = []
        num_needed = len(misses_cpu)

        # Strategy A: Use Free Slots
        while num_needed > 0 and self.free_slots:
            slots_to_use.append(self.free_slots.pop())
            num_needed -= 1

        # Strategy B: Evict LRU
        while num_needed > 0 and self.lru_tracker:
            gid, slot = self.lru_tracker.popitem(last=False)  # Pop oldest
            slots_to_use.append(slot)
            evict_ids.append(gid)
            num_needed -= 1

        # 3. Execution
        # Invalidate evicted
        if evict_ids:
            self.map_table[torch.tensor(evict_ids, device=self.device_target)] = -1

        # Safety check: if cache is strictly smaller than misses, we must truncate to avoid crash
        # (This happens if fallback didn't catch it, or partial loading)
        valid_count = len(slots_to_use)
        if valid_count < len(misses_cpu):
            misses_cpu = misses_cpu[:valid_count]

        if len(misses_cpu) > 0:
            misses_t = torch.tensor(misses_cpu, dtype=torch.long)
            slots_t = torch.tensor(
                slots_to_use, device=self.device_target, dtype=torch.long
            )

            # Async Copy
            src = self.weight.index_select(0, misses_t)
            self.cache_weight.index_copy_(
                0, slots_t, src.to(self.device_target, non_blocking=True)
            )

            # Update Map
            self.map_table[misses_t.to(self.device_target)] = slots_t

            # Update Tracker
            for i, gid in enumerate(misses_cpu):
                self.lru_tracker[gid] = slots_to_use[i]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Flatten and unique for analysis
        flat_input = input.flatten().unique()

        # -----------------------------------------------------------
        # SAFEGUARD 1: Capacity Overflow -> CPU Fallback
        # -----------------------------------------------------------
        # If the batch needs more unique tokens than the entire cache size,
        # we cannot use the GPU cache mechanism. Fallback to CPU lookup.
        if flat_input.numel() > self.cache_lines:
            # print(f"Warning: Batch unique tokens ({flat_input.numel()}) > Cache size ({self.cache_lines}). Falling back to CPU.")
            return F.embedding(
                input.cpu(),
                self.weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            ).to(self.device_target)

        # -----------------------------------------------------------
        # Standard Path: GPU Cache
        # -----------------------------------------------------------
        self._load_missing(flat_input)

        indices = self.map_table[input]

        # -----------------------------------------------------------
        # SAFEGUARD 2: Padding & Invalid Index Handling
        # -----------------------------------------------------------
        # indices might contain -1 for:
        # a) padding_idx (we didn't load it)
        # b) logic errors (shouldn't happen with Safeguard 1)

        # If we have -1s, we must mask them before passing to F.embedding to avoid CUDA Assert
        if (indices == -1).any():
            mask = indices == -1
            # Temporarily point to slot 0 (safe) so kernel doesn't crash
            # We will overwrite the result with zeros later
            safe_indices = indices.clone()
            safe_indices[mask] = 0

            out = F.embedding(
                safe_indices,
                self.cache_weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )

            # Manually zero out the masked positions
            # (Assuming standard padding behavior is zero-vector)
            out[mask] = 0.0
            return out
        else:
            return F.embedding(
                indices,
                self.cache_weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )

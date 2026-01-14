import torch
import torch.nn as nn
import unittest
import numpy as np
from engram import OffloadEmbedding


class TestOffloadConsistency(unittest.TestCase):

    def setUp(self):
        # Reset CUDA to ensure clean state if possible (optional)
        torch.cuda.empty_cache()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_embeddings = 10000
        self.embedding_dim = 64
        # Cache 设小一点，方便触发淘汰
        self.cache_lines = 100

        self.ref_emb = nn.Embedding(self.num_embeddings, self.embedding_dim)
        nn.init.normal_(self.ref_emb.weight)

    def test_creation_from_embedding(self):
        """Test factory method and weight sync"""
        if self.device == "cpu":
            return
        print("\n=== Test: Creation from Embedding ===")

        # Ensure ref is on GPU to test cross-device copy logic
        ref_gpu = self.ref_emb.to(self.device)
        off_emb = OffloadEmbedding.from_embedding(ref_gpu, cache_lines=self.cache_lines)

        self.assertEqual(off_emb.weight.device.type, "cpu")
        self.assertEqual(off_emb.cache_weight.device.type, "cuda")
        self.assertTrue(torch.allclose(off_emb.weight, self.ref_emb.weight.cpu()))
        print("Pass: Weight synced and devices correct.")

    def test_behavior_consistency(self):
        """Test output consistency with random input"""
        if self.device == "cpu":
            return
        print("\n=== Test: Behavior Consistency ===")

        ref_gpu = self.ref_emb.to(self.device)
        off_emb = OffloadEmbedding.from_embedding(ref_gpu, cache_lines=self.cache_lines)

        # Case 1: Small Batch (Fits in Cache)
        input_small = torch.randint(0, self.num_embeddings, (10, 5), device=self.device)
        with torch.no_grad():
            y_ref = ref_gpu(input_small)
            y_off = off_emb(input_small)

        self.assertTrue(torch.allclose(y_ref, y_off, atol=1e-4), "Small batch mismatch")
        print("Pass: Small batch consistent.")

        # Case 2: Large Batch (Overflows Cache -> CPU Fallback)
        # Input has 200 unique tokens, Cache is 100
        input_large = torch.arange(0, 200, device=self.device).unsqueeze(0)
        with torch.no_grad():
            y_ref_l = ref_gpu(input_large)
            y_off_l = off_emb(input_large)  # This triggers CPU fallback path

        self.assertTrue(
            torch.allclose(y_ref_l, y_off_l, atol=1e-4),
            "Large batch (fallback) mismatch",
        )
        print("Pass: Large batch (CPU Fallback) consistent.")

    def test_padding_handling(self):
        """Test padding_idx handling to ensure no crash on -1 indices"""
        if self.device == "cpu":
            return
        print("\n=== Test: Padding Handling ===")

        pad_idx = 0
        ref_pad = nn.Embedding(1000, 32, padding_idx=pad_idx).to(self.device)
        off_pad = OffloadEmbedding.from_embedding(
            ref_pad, cache_lines=50
        )  # Small cache

        # Input containing padding index and normal index
        x = torch.tensor([[pad_idx, 10, 20]], device=self.device)

        y_ref = ref_pad(x)
        y_off = off_pad(x)

        # Check if padding vector is zero
        self.assertTrue(torch.all(y_off[0, 0] == 0))
        # Check consistency
        self.assertTrue(torch.allclose(y_ref, y_off, atol=1e-4))
        print("Pass: Padding handled correctly without crash.")

    def test_module_integration(self):
        """Test inside a nn.Sequential"""
        if self.device == "cpu":
            return
        print("\n=== Test: Module Integration ===")

        model = nn.Sequential(
            OffloadEmbedding.from_embedding(
                self.ref_emb.to(self.device), cache_lines=200
            ),
            nn.Linear(self.embedding_dim, 10).to(self.device),
        )

        x = torch.randint(0, 1000, (4, 32), device=self.device)
        out = model(x)
        self.assertEqual(out.shape, (4, 32, 10))
        print("Pass: Works in nn.Sequential.")


if __name__ == "__main__":
    unittest.main()

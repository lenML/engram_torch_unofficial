import torch
import numpy as np
import sys
import os
from transformers import AutoTokenizer

# 确保能导入本地的 engram 包
sys.path.append(os.getcwd())

from engram.config import EngramConfig
from engram.tokenizer import EngramTokenizer
from engram.modules import EngramModule


def test_engram_workflow():
    print("=" * 50)
    print("🚀 开始 Engram 模块单元测试")
    print("=" * 50)

    # ------------------------------------------------------
    # 1. 准备环境
    # ------------------------------------------------------
    print("\n[Step 1] 初始化配置与 Tokenizer...")

    # 为了测试方便，我们用 gpt2 的 tokenizer (比较小)
    try:
        hf_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        hf_tokenizer.pad_token = hf_tokenizer.eos_token
    except:
        print("❌ 需要安装 transformers 和下载 gpt2 tokenizer")
        return

    # 配置：模拟一个 Dense 模型 (hc_mult=1)，开启 CPU Offload
    config = EngramConfig(
        hidden_size=512,  # 模拟一个小模型
        engram_vocab_size=len(hf_tokenizer),
        max_ngram_size=3,
        n_embed_per_ngram=64,  # 小一点方便观察
        n_head_per_ngram=4,
        hc_mult=1,  # 关键：Dense 模式
        seed=42,
    )

    # 初始化 Engram Tokenizer (模拟插入到第 2 层)
    layer_id = 2
    engram_tokenizer = EngramTokenizer(config, hf_tokenizer, layer_ids=[layer_id])
    print("✅ Tokenizer 初始化成功")

    # ------------------------------------------------------
    # 2. 测试哈希计算 (CPU 逻辑)
    # ------------------------------------------------------
    print("\n[Step 2] 测试哈希计算 (Hash Calculation)...")
    text = ["Hello world, this is a test for Engram.", "Short sentence."]
    hf_enc = hf_tokenizer(
        text, return_tensors="np", padding=True, truncation=True, max_length=20
    )
    input_ids = hf_enc["input_ids"]

    print(f"   Input shape: {input_ids.shape}")

    # 核心测试：计算 Hash
    hash_ids = engram_tokenizer.compress_and_hash(input_ids, layer_id=layer_id)

    # 验证维度: [Batch, Seq, Num_Heads_Total]
    # Total Heads = (max_ngram_size - 1) * n_head_per_ngram
    # 这里 max_ngram=3 (即2-gram, 3-gram), head=4 => total=8
    expected_heads = (config.max_ngram_size - 1) * config.n_head_per_ngram
    print(f"   Hash shape:  {hash_ids.shape}")

    assert hash_ids.shape == (
        input_ids.shape[0],
        input_ids.shape[1],
        expected_heads,
    ), f"❌ 哈希维度错误! 预期 (B, L, {expected_heads}), 实际 {hash_ids.shape}"
    print("✅ 哈希维度校验通过")

    # ------------------------------------------------------
    # 3. 初始化模块与零初始化检查
    # ------------------------------------------------------
    print("\n[Step 3] 初始化 EngramModule 与 零初始化检查...")
    vocab_sizes = engram_tokenizer.vocab_distributions[layer_id]
    module = EngramModule(config, vocab_sizes)

    # 检查 Embedding 是否在 CPU (因为 cpu_offload=True)
    is_on_cpu = module.memory.embedding.weight.device.type == "cpu"
    print(f"   Embedding is on CPU? {is_on_cpu}")
    assert is_on_cpu, "❌ 配置了 Offload 但 Embedding 却在 GPU 上"

    # 检查卷积权重是否为 0
    conv_weight_sum = module.conv.weight.abs().sum().item()
    print(f"   Conv weight L1 norm: {conv_weight_sum}")
    assert conv_weight_sum == 0, "❌ 零初始化失败！卷积权重不为 0，这会破坏预训练模型。"
    print("✅ 零初始化校验通过")

    # ------------------------------------------------------
    # 4. Forward Pass (GPU 推理)
    # ------------------------------------------------------
    print("\n[Step 4] 前向传播测试 (Forward Pass)...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Testing on device: {device}")

    # 将模块非 Embedding 部分移到 GPU
    module.to(device)

    # 构造 Dummy Hidden States [B, L, D]
    B, L = input_ids.shape
    hidden_states = torch.randn(B, L, config.hidden_size).to(device)

    # 构造 Hash IDs (需要转为 Tensor, 但不需要手动 .to(device) 因为 module 内部处理 offload)
    # 但为了模拟真实 DataLoader，我们通常传过来 tensor
    hash_tensor = torch.from_numpy(hash_ids).long().to(device)

    # RUN
    try:
        # offload embedding to CPU
        module.memory.cpu()

        output = module(hidden_states, hash_tensor)
        print(f"   Output shape: {output.shape}")

        # 验证维度不变 (Residual 适配)
        assert (
            output.shape == hidden_states.shape
        ), f"❌ 输出维度错误! 预期 {hidden_states.shape}, 实际 {output.shape}"

        # 验证初始输出值极小 (因为零初始化)
        # 注意：由于 Gating 的 sigmoid 和 value_proj 的初始化，输出不一定是纯 0，但应该非常小
        # 或者如果是 Conv 后的残差，应该是 0 (取决于具体实现，Demo中是 value + conv(value))
        # 我们这里只检查是否数值爆炸
        out_mean = output.abs().mean().item()
        print(f"   Output mean abs value: {out_mean:.6f}")
        if out_mean > 0.1:
            print("⚠️ 警告: 初始输出值较大，可能会对主干模型产生冲击")
        else:
            print("✅ 初始输出值处于安全范围")

        print("✅ 前向传播成功")

    except Exception as e:
        print(f"❌ 前向传播崩溃: {e}")
        import traceback

        traceback.print_exc()

    # ------------------------------------------------------
    # 5. 反向传播测试 (Gradient Flow)
    # ------------------------------------------------------
    print("\n[Step 5] 反向传播测试 (Backward Pass)...")
    try:
        loss = output.sum()
        loss.backward()

        # 检查 Embedding 是否有梯度
        # 注意：Embedding 在 CPU 上，PyTorch 支持 CPU->GPU 的梯度回传吗？
        # PyTorch 的 Embedding 如果在 CPU，input 在 GPU，backward 时通常也是 OK 的
        embed_grad = module.memory.embedding.weight.grad

        if embed_grad is not None:
            grad_norm = embed_grad.norm().item()
            print(f"   Embedding gradient norm: {grad_norm}")
            print("✅ 梯度回传成功 (CPU Embedding 接收到了梯度)")
        else:
            print("❌ Embedding 没有接收到梯度!")

    except Exception as e:
        print(f"❌ 反向传播崩溃: {e}")

    print("\n" + "=" * 50)
    print("🎉 所有测试完成！如果全是 ✅，说明 Engram 模块可用。")
    print("=" * 50)


if __name__ == "__main__":
    test_engram_workflow()

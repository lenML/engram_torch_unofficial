# engram_torch_unofficial

```py
from engram import EngramConfig, EngramTokenizer, EngramModule, ScalableEmbedding
from transformers import AutoTokenizer

hf_tokenizer = AutoTokenizer.from_pretrained("gpt2")

k = 5 # Knowledge scaling
config = EngramConfig(
    hidden_size=1024,
    engram_vocab_size=len(hf_tokenizer),
    engram_table_size=[len(hf_tokenizer) * k, len(hf_tokenizer) * k],
    max_ngram_size=3,
    n_embed_per_ngram=512,
    n_head_per_ngram=8,
    hc_mult=1,  # 1 for Dense model, 4 for DeepSeek-V3
    pad_id=hf_tokenizer.pad_token,
    seed=42,
)
layer_ids = [2, 6]  # gpt2 has 12 layers, we use layer 2 and 6 for testing
engram_tokenizer = EngramTokenizer(config, hf_tokenizer, layer_ids=layer_ids)

vocab_sizes_2 = engram_tokenizer.vocab_distributions[layer_ids[0]]
engram_2 = EngramModule(config, vocab_sizes_2)
vocab_sizes_6 = engram_tokenizer.vocab_distributions[layer_ids[1]]
engram_6 = EngramModule(config, vocab_sizes_6)

# you can offload memory(embeddings) to cpu:
engram_2.to("cuda")
engram_2.memory.to("cpu")

```

WIP

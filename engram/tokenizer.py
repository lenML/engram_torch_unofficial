import numpy as np
from tokenizers import normalizers, Regex
from transformers import PreTrainedTokenizer
from typing import List, Dict
from .utils import find_next_prime
from .config import EngramConfig

# 默认的归一化器
SENTINEL = "\ue000"
DEFAULT_NORMALIZER = normalizers.Sequence(
    [
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), SENTINEL),
        normalizers.Strip(),
        normalizers.Replace(SENTINEL, " "),
    ]
)


class EngramTokenizer:
    def __init__(
        self,
        config: EngramConfig,
        hf_tokenizer: PreTrainedTokenizer,
        layer_ids: List[int],
        normalizer: normalizers.Sequence = DEFAULT_NORMALIZER,
    ):
        """
        config: EngramConfig 对象
        hf_tokenizer: HuggingFace Tokenizer 实例
        layer_ids: 需要插入 Engram 的层索引列表 (用于生成不同的随机种子)
        normalizer: 归一化器，默认为 DEFAULT_NORMALIZER
        """
        self.cfg = config
        self.layer_ids = layer_ids
        self.tokenizer = hf_tokenizer
        self.normalizer = normalizer

        self.lookup_table = self._build_compression_table()
        self.compressed_vocab_size = int(self.lookup_table.max() + 1)

        self.layer_params = self._init_hash_params()
        self.vocab_distributions = self._calc_vocab_distributions()

    def _build_compression_table(self):
        old2new: Dict[int, int] = {}
        key2new: Dict[str, int] = {}
        new_tokens = []

        vocab_size = len(self.tokenizer)
        for tid in range(vocab_size):
            text = self.tokenizer.decode([tid], skip_special_tokens=False)

            if "�" in text:
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text

            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid

        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]

        return lookup

    def _init_hash_params(self):
        # 为每一层生成不同的随机乘数
        params = {}
        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // self.compressed_vocab_size)
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007

        for lid in self.layer_ids:
            base_seed = int(self.cfg.seed + PRIME_1 * int(lid))
            rng = np.random.default_rng(base_seed)
            # 生成 max_ngram_size 个乘数
            multipliers = rng.integers(
                0, half_bound, size=(self.cfg.max_ngram_size,), dtype=np.int64
            )
            params[lid] = multipliers * 2 + 1
        return params

    def _calc_vocab_distributions(self):
        # 计算每一层、每个 N-gram、每个 Head 的模数 (素数)
        # 返回结构: {layer_id: [[head1_prime, head2_prime...], ...]}
        dists = {}
        seen_primes = set()

        for lid in self.layer_ids:
            layer_dists = []  # [N-gram-2, N-gram-3...]
            for n_idx in range(len(self.cfg.engram_table_size)):
                target_size = self.cfg.engram_table_size[n_idx]
                heads = []
                curr = target_size // self.cfg.n_head_per_ngram
                for _ in range(self.cfg.n_head_per_ngram):
                    p = find_next_prime(curr, seen_primes)
                    seen_primes.add(p)
                    heads.append(p)
                    curr = p
                layer_dists.append(heads)
            dists[lid] = layer_dists
        return dists

    def compress_and_hash(self, input_ids: np.ndarray, layer_id: int) -> np.ndarray:
        """
        输入: 原始 Token IDs [Batch, Seq] (Numpy)
        输出: 哈希索引 [Batch, Seq, Num_Heads] (Numpy)
        """
        x = np.asarray(input_ids, dtype=np.int64)
        B, T = x.shape

        multipliers = self.layer_params[layer_id]

        def shift_k(k: int) -> np.ndarray:
            if k == 0:
                return x
            shifted = np.pad(
                x, ((0, 0), (k, 0)), mode="constant", constant_values=self.cfg.pad_id
            )[:, :T]
            return shifted

        base_shifts = [shift_k(k) for k in range(self.cfg.max_ngram_size)]

        all_hashes = []

        for n in range(2, self.cfg.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]
            mix = tokens[0] * multipliers[0]
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
            num_heads_for_this_ngram = self.cfg.n_head_per_ngram
            head_vocab_sizes = self.vocab_distributions[layer_id][n_gram_index]

            for j in range(num_heads_for_this_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = mix % mod
                all_hashes.append(head_hash.astype(np.int64, copy=False))

        return np.stack(all_hashes, axis=2)

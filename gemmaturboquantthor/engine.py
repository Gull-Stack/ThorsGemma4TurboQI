"""GemmaTurboQuantThor engine — load Gemma 4 + TurboQuant in one call.

Gemma 4 31B architecture:
- 60 layers total: 50 sliding-attention (window=1024) + 10 full-attention (global)
- Full attention layers appear every 6th layer (indices 5,11,17,23,29,35,41,47,53,59)
- head_dim=256, global_head_dim=512, GQA with 32 query / 16 KV heads
- TurboQuant targets the 10 full-attention layers (global KV cache, unbounded growth)
- Sliding-attention layers have bounded cache (window=1024) — less benefit from compression

Usage:
    from gemmaturboquantthor.engine import GemmaEngine

    engine = GemmaEngine()  # loads model + creates TQ cache
    response = engine.generate("Explain quantum computing.")
    print(response.text)
"""

import time
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx

from gemmaturboquantthor.cache.cache import TurboQuantKVCache
from gemmaturboquantthor.gemma4_cache import make_gemma4_turboquant_cache


# Gemma 4 model variants (MLX 4-bit quantized, loaded via mlx-vlm)
MODELS = {
    "31b": "mlx-community/gemma-4-31b-it-4bit",
    "27b-a4b": "mlx-community/gemma-4-26B-A4B-it-4bit",
    "4b": "mlx-community/gemma-4-4b-it-4bit",
    "2b": "mlx-community/gemma-4-2b-it-4bit",
}

DEFAULT_MODEL = "27b-a4b"

# Gemma 4 layer types from config
GEMMA4_FULL_ATTENTION_INDICES = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59]


@dataclass
class GenerationResult:
    """Result from a generation call."""
    text: str
    tokens: int
    prefill_time: float
    decode_time: float
    prefill_tps: float
    decode_tps: float
    cache_memory_mb: float
    compression_ratio: float
    n_tq_layers: int = 0
    n_sliding_layers: int = 0


@dataclass
class EngineConfig:
    """Configuration for the Gemma + TurboQuant engine."""
    model: str = DEFAULT_MODEL
    key_bits: int = 3
    value_bits: int = 4
    layer_adaptive: bool = True
    compress_sliding: bool = False  # Also compress sliding-attention layers (bounded cache)
    sparse_v: bool = False
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    seed: int = 42


class GemmaEngine:
    """Gemma 4 inference engine with TurboQuant KV cache compression.

    Gemma 4 uses a hybrid sliding/full-attention architecture:
    - 50 sliding-attention layers (window=1024, bounded KV cache)
    - 10 full-attention layers (global, unbounded KV cache growth)

    TurboQuant compresses the full-attention layer caches by default,
    which are the memory bottleneck for long context. Sliding layers
    can optionally be compressed too via compress_sliding=True.

    Args:
        config: EngineConfig or None for defaults
        model_id: Override model (HuggingFace ID or shorthand like "31b")
    """

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        model_id: Optional[str] = None,
    ):
        self.config = config or EngineConfig()

        if model_id:
            self.config.model = model_id

        # Resolve shorthand model names
        resolved = MODELS.get(self.config.model, self.config.model)

        print(f"Loading {resolved}...")
        self._load_model(resolved)
        print(f"  {self.n_layers} layers ({self.n_full_attn} full-attention + {self.n_sliding} sliding)")

        # Detect head_dim
        self.head_dim = self._detect_head_dim()
        print(f"  head_dim={self.head_dim}")

        # Apply sparse V patch if enabled
        if self.config.sparse_v:
            apply_turboquant(self._text_model, sparse_v=True)

        print(f"  TurboQuant: keys={self.config.key_bits}-bit, values={self.config.value_bits}-bit, "
              f"layer_adaptive={self.config.layer_adaptive}")
        print(f"  Compressing: {'all layers' if self.config.compress_sliding else 'full-attention layers only (10/60)'}")
        print("Ready.")

    def _load_model(self, model_id: str):
        """Load model via mlx-vlm (supports Gemma 4's multimodal architecture)."""
        from mlx_vlm.utils import load
        self._vlm_model, self._processor = load(model_id)

        # Extract the text/language model for direct inference
        if hasattr(self._vlm_model, 'language_model'):
            self._text_model = self._vlm_model.language_model
            if hasattr(self._text_model, 'model'):
                self._inner_model = self._text_model.model
            else:
                self._inner_model = self._text_model
        else:
            self._text_model = self._vlm_model
            self._inner_model = self._vlm_model

        self.layers = self._inner_model.layers
        self.n_layers = len(self.layers)

        # Detect layer types from config
        self.layer_types = getattr(self._inner_model.config if hasattr(self._inner_model, 'config') else
                                   self._vlm_model.config.text_config if hasattr(self._vlm_model, 'config') else
                                   None, 'layer_types', None)

        if self.layer_types:
            self.full_attn_indices = [i for i, t in enumerate(self.layer_types) if t == "full_attention"]
            self.sliding_indices = [i for i, t in enumerate(self.layer_types) if t == "sliding_attention"]
        else:
            # Fallback: assume all layers are full attention
            self.full_attn_indices = list(range(self.n_layers))
            self.sliding_indices = []

        self.n_full_attn = len(self.full_attn_indices)
        self.n_sliding = len(self.sliding_indices)

        # Set up tokenizer from processor
        self.tokenizer = self._processor.tokenizer if hasattr(self._processor, 'tokenizer') else self._processor

    def _detect_head_dim(self) -> int:
        for layer in self.layers:
            for attr in ('self_attn', 'attention'):
                attn = getattr(layer, attr, None)
                if attn is not None and hasattr(attn, 'head_dim'):
                    return attn.head_dim
        return 256  # Gemma 4 default

    def _make_cache(self):
        """Create TurboQuant cache — compresses full-attention layers, preserves sliding layers."""
        return make_gemma4_turboquant_cache(
            self._text_model,
            key_bits=self.config.key_bits,
            value_bits=self.config.value_bits,
            layer_adaptive=self.config.layer_adaptive,
            seed=self.config.seed,
        )

    def _make_standard_cache(self):
        """Create standard (uncompressed) cache for benchmarking."""
        from mlx_lm.models.cache import KVCache
        return [KVCache() for _ in range(self.n_layers)]

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
    ) -> GenerationResult:
        """Generate a response to a prompt.

        Args:
            prompt: User message
            max_tokens: Override max tokens
            system: Optional system prompt
            temperature: Override temperature (0 = greedy)
            stream: If True, yields tokens as they're generated

        Returns:
            GenerationResult with text, timing, and memory stats
        """
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature if temperature is not None else self.config.temperature

        # Build messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        input_ids = mx.array(self.tokenizer.encode(text))[None]

        # Fresh cache per generation
        cache = self._make_cache()

        # Count TQ vs standard layers
        n_tq = sum(1 for c in cache if isinstance(c, TurboQuantKVCache))
        n_std = len(cache) - n_tq

        # Prefill
        t0 = time.perf_counter()
        out = self._text_model(input_ids, cache=cache)
        logits = out.logits if hasattr(out, 'logits') else out
        mx.eval(logits)
        t_prefill = time.perf_counter() - t0
        prefill_toks = input_ids.shape[1]

        # Decode
        generated_tokens = []

        if temperature <= 0:
            token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        else:
            logits_last = logits[:, -1, :] / temperature
            token = mx.random.categorical(logits_last)[None]

        generated_tokens.append(token.item())

        t_decode_start = time.perf_counter()
        for _ in range(max_tokens - 1):
            out = self._text_model(token, cache=cache)
            logits = out.logits if hasattr(out, 'logits') else out
            mx.eval(logits)

            if temperature <= 0:
                token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            else:
                logits_last = logits[:, -1, :] / temperature
                token = mx.random.categorical(logits_last)[None]

            tok_id = token.item()
            generated_tokens.append(tok_id)
            if tok_id == self.tokenizer.eos_token_id:
                break
        t_decode = time.perf_counter() - t_decode_start

        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        decode_toks = len(generated_tokens)

        # Cache stats
        cache_mem = sum(c.nbytes for c in cache if hasattr(c, 'nbytes')) / 1024 / 1024
        tq_layers = [c for c in cache if isinstance(c, TurboQuantKVCache) and not c.empty()]
        avg_ratio = sum(c.compression_ratio for c in tq_layers) / len(tq_layers) if tq_layers else 0

        return GenerationResult(
            text=output_text,
            tokens=decode_toks,
            prefill_time=t_prefill,
            decode_time=t_decode,
            prefill_tps=prefill_toks / t_prefill if t_prefill > 0 else 0,
            decode_tps=decode_toks / t_decode if t_decode > 0 else 0,
            cache_memory_mb=cache_mem,
            compression_ratio=avg_ratio,
            n_tq_layers=n_tq,
            n_sliding_layers=n_std,
        )

    def benchmark(self, prompt: str, max_tokens: int = 200) -> dict:
        """Run standard vs TurboQuant comparison.

        Returns dict with both results and comparison stats.
        """
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        input_ids = mx.array(self.tokenizer.encode(text))[None]

        def _run(cache_list, label):
            t0 = time.perf_counter()
            out = self._text_model(input_ids, cache=cache_list)
            logits = out.logits if hasattr(out, 'logits') else out
            mx.eval(logits)
            t_prefill = time.perf_counter() - t0

            generated = []
            token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            generated.append(token.item())

            t_dec = time.perf_counter()
            for _ in range(max_tokens - 1):
                out = self._text_model(token, cache=cache_list)
                logits = out.logits if hasattr(out, 'logits') else out
                mx.eval(logits)
                token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
                tok_id = token.item()
                generated.append(tok_id)
                if tok_id == self.tokenizer.eos_token_id:
                    break
            t_decode = time.perf_counter() - t_dec

            mem = sum(c.nbytes for c in cache_list if hasattr(c, 'nbytes'))
            output = self.tokenizer.decode(generated, skip_special_tokens=True)

            return {
                "label": label,
                "text": output,
                "prefill_tps": input_ids.shape[1] / t_prefill,
                "decode_tps": len(generated) / t_decode if t_decode > 0 else 0,
                "cache_mb": mem / 1024 / 1024,
                "tokens": len(generated),
            }

        # Standard
        std = _run(self._make_standard_cache(), "standard_fp16")
        mx.clear_cache()

        # TurboQuant
        tq = _run(self._make_cache(), "turboquant")

        total_tokens = input_ids.shape[1] + std["tokens"]
        std_32k = (std["cache_mb"] / total_tokens * 32768) / 1024 if total_tokens > 0 else 0
        tq_32k = (tq["cache_mb"] / total_tokens * 32768) / 1024 if total_tokens > 0 else 0

        return {
            "standard": std,
            "turboquant": tq,
            "architecture": {
                "total_layers": self.n_layers,
                "full_attention": self.n_full_attn,
                "sliding_attention": self.n_sliding,
            },
            "projection_32k": {
                "standard_gb": std_32k,
                "turboquant_gb": tq_32k,
                "savings_gb": std_32k - tq_32k,
            },
        }

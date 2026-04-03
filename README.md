# GemmaTurboQuantThor

**Gemma 4 31B + TurboQuant KV cache compression for Apple Silicon.**

Run Google's largest open model on a 24GB M4 Pro with 4x compressed KV cache, unlocking 32K+ context windows that would otherwise OOM.

## What This Does

Combines two technologies:
- **Gemma 4 31B** — Google's most capable open model (Apache 2.0), 31B dense transformer
- **TurboQuant** — KV cache compression (ICLR 2026) via asymmetric quantization: 3-bit keys, 4-bit values

Every layer in Gemma 4 is full attention (no hybrid/MoE routing), so TurboQuant compresses the entire KV cache — maximum savings.

| Config | KV Cache (32K ctx) | Max Context (24GB) |
|--------|-------------------|-------------------|
| Standard fp16 | ~8 GB | ~8K (OOM) |
| TurboQuant 3/4-bit | ~1.8 GB | ~32K+ |

## Quick Start

```bash
git clone https://github.com/Gull-Stack/gemmaturboquantthor.git
cd gemmaturboquantthor

python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Generate
gemmatq generate "Explain quantum computing in simple terms."

# Interactive chat
gemmatq chat

# Benchmark standard vs TurboQuant
gemmatq benchmark

# OpenAI-compatible API server
pip install -e ".[server]"
gemmatq serve --port 8080
```

## Python API

```python
from gemmaturboquantthor.engine import GemmaEngine

engine = GemmaEngine()
result = engine.generate("What is the meaning of life?")
print(result.text)
print(f"{result.decode_tps:.1f} tok/s, {result.compression_ratio:.1f}x compressed")
```

## Configuration

```python
from gemmaturboquantthor.engine import GemmaEngine, EngineConfig

config = EngineConfig(
    model="31b",          # "31b", "27b-a4b", "4b", "2b"
    key_bits=3,           # 3-bit keys (direction — compresses well)
    value_bits=4,         # 4-bit values (magnitude — more sensitive)
    layer_adaptive=True,  # Last 20% of layers get +1 bit
    sparse_v=False,       # Sparse V decode optimization
    max_tokens=512,
    temperature=0.0,
)
engine = GemmaEngine(config)
```

## Architecture

```
gemmaturboquantthor/
├── core/               # TurboQuant compression engine
│   ├── codebook.py     # Lloyd-Max optimal codebooks for N(0,1)
│   ├── rotation.py     # Walsh-Hadamard + random sign rotation
│   ├── quantizer.py    # MSE quantizer + asymmetric K/V
│   ├── packing.py      # Bit packing (2/3/4-bit into uint32)
│   ├── sparse_v.py     # Adaptive sparse V dequantization
│   └── metal_kernels*  # Fused Metal GPU kernels for M4 Pro
├── cache/              # MLX integration
│   ├── cache.py        # TurboQuantKVCache (drop-in for mlx-lm)
│   ├── attention.py    # SDPA with sparse V
│   └── patch.py        # mlx-lm monkey-patch
├── engine.py           # Main engine: load + generate + benchmark
├── server.py           # OpenAI-compatible /v1/chat/completions
├── openclaw_skill.py   # OpenClaw agent skill integration
└── cli.py              # CLI entry point (gemmatq)
```

## OpenClaw Integration

Expose Gemma 4 + TurboQuant as a skill for any OpenClaw agent:

```bash
# Start the server
gemmatq serve --port 8080

# Register as an OpenClaw skill
cp gemmaturboquantthor/openclaw_skill.py ~/.openclaw/workspace/skills/gemmatq/handler.py
```

## How TurboQuant Works

1. **Normalize** each KV vector, store norm as float32
2. **Rotate** via Walsh-Hadamard Transform (each coordinate becomes Gaussian)
3. **Quantize** with Lloyd-Max optimal codebooks (information-theoretically optimal)
4. **Pack** indices into bit-packed uint32 (3-bit: 10 values per word)

Asymmetric K/V: Keys carry direction (Q·K dot products) and compress to 3-bit with cos_sim=1.000. Values carry magnitude and need 4-bit for <1% quality loss.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4) — Metal GPU required
- Python 3.10+
- 24GB+ unified memory for 31B model

## License

MIT

## Credits

- [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- [Gemma 4](https://ai.google.dev/gemma) (Google DeepMind, Apache 2.0)
- [TurboQuant-Thor](https://github.com/Gull-Stack/turboquant-thor) (Gull-Stack)
- [MLX](https://github.com/ml-explore/mlx) (Apple)

Built by [Gull-Stack](https://gullstack.com)

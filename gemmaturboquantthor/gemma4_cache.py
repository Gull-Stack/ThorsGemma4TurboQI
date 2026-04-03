"""Gemma 4 specific TurboQuant cache factory.

Gemma 4 has a hybrid architecture with two different attention types:
- Sliding attention (50 layers): head_dim=256, n_kv_heads=16
- Full attention (10 layers): head_dim=512 (global_head_dim), n_kv_heads=4

TurboQuantKVCache must be created with the correct per-layer dimensions.
"""

from gemmaturboquantthor.cache.cache import TurboQuantKVCache


def make_gemma4_turboquant_cache(
    model,
    key_bits: int = 3,
    value_bits: int = 4,
    layer_adaptive: bool = True,
    seed: int = 42,
):
    """Create per-layer TurboQuant cache for Gemma 4's hybrid architecture.

    Detects each layer's head_dim, n_kv_heads, and attention type from
    the model's actual attention modules, then creates correctly-sized
    TurboQuantKVCache instances.

    Args:
        model: The Gemma 4 language model (LanguageModel or Gemma4TextModel)
        key_bits: Bits for key quantization
        value_bits: Bits for value quantization
        layer_adaptive: Use higher precision for final layers
        seed: Random seed for rotation

    Returns:
        List of cache objects, one per layer
    """
    from mlx_lm.models.cache import KVCache

    # Get the inner model with .layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        inner = model.model
    elif hasattr(model, 'layers'):
        inner = model
    else:
        raise ValueError("Cannot find model layers")

    layers = inner.layers
    n_layers = len(layers)

    # Identify full-attention layer indices for layer-adaptive calculation
    full_attn_indices = []
    for i, layer in enumerate(layers):
        attn = getattr(layer, 'self_attn', None)
        if attn and getattr(attn, 'layer_type', '') == 'full_attention':
            full_attn_indices.append(i)

    n_full_attn = len(full_attn_indices)

    caches = []
    full_attn_pos = 0

    for i, layer in enumerate(layers):
        attn = getattr(layer, 'self_attn', None)

        if attn is None:
            # No attention — use standard cache
            caches.append(KVCache())
            continue

        layer_type = getattr(attn, 'layer_type', 'sliding_attention')
        head_dim = getattr(attn, 'head_dim', 256)

        if layer_type == 'full_attention':
            # Full-attention layers: head_dim=512, 4 KV heads
            # These are the memory bottleneck — compress with TurboQuant
            caches.append(TurboQuantKVCache(
                key_bits=key_bits,
                value_bits=value_bits,
                head_dim=head_dim,
                layer_idx=full_attn_pos,
                n_layers=n_full_attn,
                layer_adaptive=layer_adaptive,
                seed=seed,
            ))
            full_attn_pos += 1
        else:
            # Sliding-attention layers: bounded cache (window=1024)
            # Standard cache is fine — small and bounded
            caches.append(KVCache())

    return caches

"""GemmaTurboQuantThor CLI — run Gemma 4 + TurboQuant from the command line.

Usage:
    gemmatq generate "What is the meaning of life?"
    gemmatq chat
    gemmatq benchmark
    gemmatq serve --port 8080
"""

import argparse
import sys

from gemmaturboquantthor.engine import GemmaEngine, EngineConfig, MODELS


def cmd_generate(args):
    config = EngineConfig(
        model=args.model,
        key_bits=args.key_bits,
        value_bits=args.value_bits,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    engine = GemmaEngine(config)
    result = engine.generate(args.prompt, system=args.system)

    print()
    print(result.text)
    print()
    print(f"--- {result.tokens} tokens | "
          f"prefill {result.prefill_tps:.1f} tok/s | "
          f"decode {result.decode_tps:.1f} tok/s | "
          f"cache {result.cache_memory_mb:.1f} MB ({result.compression_ratio:.1f}x compressed)")


def cmd_chat(args):
    config = EngineConfig(
        model=args.model,
        key_bits=args.key_bits,
        value_bits=args.value_bits,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    engine = GemmaEngine(config)

    print("\nGemma 4 + TurboQuant — type 'quit' to exit\n")
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not prompt or prompt.lower() in ("quit", "exit", "q"):
            break

        result = engine.generate(prompt, system=args.system)
        print(f"\nGemma: {result.text}")
        print(f"  [{result.tokens} tok | {result.decode_tps:.1f} tok/s | "
              f"{result.cache_memory_mb:.1f} MB cache ({result.compression_ratio:.1f}x)]\n")


def cmd_benchmark(args):
    config = EngineConfig(
        model=args.model,
        key_bits=args.key_bits,
        value_bits=args.value_bits,
    )
    engine = GemmaEngine(config)

    prompt = args.prompt or "Explain how KV cache compression works in large language models in 3 sentences."
    print(f"\nBenchmarking: \"{prompt[:80]}...\"" if len(prompt) > 80 else f"\nBenchmarking: \"{prompt}\"")
    print()

    results = engine.benchmark(prompt, max_tokens=args.max_tokens)

    std = results["standard"]
    tq = results["turboquant"]
    proj = results["projection_32k"]

    print("=" * 60)
    print(f"{'':30s} {'Standard':>12s} {'TurboQuant':>12s}")
    print("-" * 60)
    print(f"{'Prefill (tok/s)':30s} {std['prefill_tps']:12.1f} {tq['prefill_tps']:12.1f}")
    print(f"{'Decode (tok/s)':30s} {std['decode_tps']:12.1f} {tq['decode_tps']:12.1f}")
    print(f"{'Cache memory (MB)':30s} {std['cache_mb']:12.2f} {tq['cache_mb']:12.2f}")
    print("-" * 60)
    print(f"{'32K context projection (GB)':30s} {proj['standard_gb']:12.2f} {proj['turboquant_gb']:12.2f}")
    print(f"{'32K savings (GB)':30s} {'':12s} {proj['savings_gb']:12.2f}")
    print("=" * 60)

    print(f"\nStandard output:\n  {std['text'][:300]}")
    print(f"\nTurboQuant output:\n  {tq['text'][:300]}")


def cmd_serve(args):
    from gemmaturboquantthor.server import create_app
    import uvicorn

    config = EngineConfig(
        model=args.model,
        key_bits=args.key_bits,
        value_bits=args.value_bits,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    app = create_app(config)
    print(f"\nStarting server on http://0.0.0.0:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


def main():
    parser = argparse.ArgumentParser(
        prog="gemmatq",
        description="Gemma 4 + TurboQuant KV cache compression",
    )
    parser.add_argument("--model", default="31b", choices=list(MODELS.keys()),
                        help="Gemma 4 model variant (default: 31b)")
    parser.add_argument("--key-bits", type=int, default=3, help="Key quantization bits (default: 3)")
    parser.add_argument("--value-bits", type=int, default=4, help="Value quantization bits (default: 4)")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0=greedy)")

    sub = parser.add_subparsers(dest="command")

    # generate
    gen = sub.add_parser("generate", help="Generate a single response")
    gen.add_argument("prompt", help="Input prompt")
    gen.add_argument("--system", help="System prompt")

    # chat
    ch = sub.add_parser("chat", help="Interactive chat")
    ch.add_argument("--system", help="System prompt")

    # benchmark
    bm = sub.add_parser("benchmark", help="Compare standard vs TurboQuant")
    bm.add_argument("--prompt", help="Benchmark prompt")

    # serve
    sv = sub.add_parser("serve", help="Start OpenAI-compatible API server")
    sv.add_argument("--port", type=int, default=8080, help="Server port")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

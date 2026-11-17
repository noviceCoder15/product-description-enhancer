"""Command line interface for the Product Description Enhancer.

    python -m app.cli data                      # write the starter dataset
    python -m app.cli train                     # fine-tune GPT-2 with LoRA
    python -m app.cli generate --name "Aero Earbuds" --spec "Battery life: 12 hours" --compare
    python -m app.cli review                    # base vs fine-tuned report for manual review
    python -m app.cli merge                     # save a standalone merged model
    python -m app.cli assess --spec ... --text "..."   # score any description

The Streamlit app (app/main.py) offers the same features in a browser.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import core, storage


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _adapter_dir(args) -> str:
    return str(Path(args.adapters) / storage.safe_name(args.adapter))


def _print_metrics(metrics: dict) -> None:
    print(f"specs mentioned : {metrics['spec_coverage']:.0%}")
    print(f"keywords used   : {metrics['keyword_coverage']:.0%}")
    print(f"words           : {metrics['words']}")
    print(f"repetition      : {metrics['repetition']:.0%}")
    print("flags           : " + ("; ".join(metrics["flags"]) if metrics["flags"] else "none"))


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_data(args) -> int:
    examples = core.synthesize_dataset(args.n, args.seed)
    storage.write_jsonl(args.out, examples)
    print(f"wrote {len(examples)} examples to {args.out}")
    print(core.dataset_summary(examples))
    return 0


def cmd_assess(args) -> int:
    specs = core.parse_specs("\n".join(args.spec or []))
    metrics = core.assess(args.text, specs, core.parse_keywords(args.keywords),
                          args.name or "", args.category or "")
    _print_metrics(metrics)
    return 0


def cmd_train(args) -> int:
    examples, problems = storage.read_dataset(args.data)
    if problems:
        print(f"skipped {len(problems)} unusable rows, for example: {problems[0]}")
    if len(examples) < 10:
        raise ValueError(f"only {len(examples)} usable examples - LoRA needs at least about 10 (50 or more is better)")
    adapter_dir = _adapter_dir(args)
    from . import llm

    summary = llm.fine_tune(
        args.model, examples, adapter_dir, rank=args.rank, alpha=args.alpha, dropout=args.dropout,
        targets=tuple(t.strip() for t in args.targets.split(",") if t.strip()), steps=args.steps,
        batch_size=args.batch_size, lr=args.lr, max_length=args.max_length, seed=args.seed,
    )
    print(f"\nsaved adapter to {adapter_dir} ({summary['seconds']}s)")
    print(f"validation loss {summary['val_loss']:.3f}" if summary["val_loss"] is not None else "")
    return 0


def cmd_generate(args) -> int:
    specs = core.parse_specs("\n".join(args.spec or []))
    if not args.name.strip() or not specs:
        raise ValueError("give --name and at least one --spec, like --spec 'Battery life: 12 hours'")
    keywords = core.parse_keywords(args.keywords)
    prompt = core.build_prompt(args.name, args.category, specs, keywords, args.tone)
    adapter_dir = _adapter_dir(args) if args.adapter else None
    if adapter_dir:
        storage.read_adapter_config(adapter_dir)          # fail early if it does not exist
    from . import llm

    def write(model, tokenizer):
        return llm.generate_description(model, tokenizer, prompt, args.max_tokens, args.temperature, seed=args.seed)

    model, tokenizer = llm.load_for_inference(None if adapter_dir else args.model, adapter_dir, not args.no_merge)
    text = write(model, tokenizer)
    print(("Fine-tuned: " if adapter_dir else "Base model: ") + (text or "(empty)"))
    print()
    _print_metrics(core.assess(text, specs, keywords, args.name, args.category))
    if args.compare and adapter_dir:
        base_name = storage.read_adapter_config(adapter_dir)["base_model"]
        base_model, base_tokenizer = llm.load_for_inference(base_name, None)
        print("\nBase model: " + (write(base_model, base_tokenizer) or "(empty)"))
    return 0


def cmd_merge(args) -> int:
    adapter_dir = _adapter_dir(args)
    storage.read_adapter_config(adapter_dir)
    from . import llm

    out = llm.export_merged(adapter_dir, args.out)
    print(f"saved the merged model to {out}")
    print("Load it like any Hugging Face model, for example: --model " + out)
    return 0


def cmd_review(args) -> int:
    adapter_dir = _adapter_dir(args)
    storage.read_adapter_config(adapter_dir)
    examples, _ = storage.read_dataset(args.data)
    from . import llm

    report = llm.review(adapter_dir, examples, args.count, args.temperature, args.max_tokens, args.seed)
    storage.write_text(args.out, report)
    print(f"wrote {args.out} - open it, read the descriptions, and fill in the scores")
    return 0


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", default=_env("DATA_PATH", "data/products.jsonl"), help="dataset (.jsonl or .csv)")
    common.add_argument("--model", default=_env("BASE_MODEL", "gpt2"), help="base model name or local folder")
    common.add_argument("--adapters", default=_env("ADAPTER_DIR", "adapters"), help="folder that holds adapters")
    common.add_argument("--adapter", default="default", help="adapter name (default: default)")
    common.add_argument("--seed", type=int, default=None)

    product = argparse.ArgumentParser(add_help=False)
    product.add_argument("--name", default="", help="product name")
    product.add_argument("--category", default="general")
    product.add_argument("--spec", action="append", help="'Name: value'; repeat for each spec")
    product.add_argument("--keywords", default="", help="comma-separated target keywords")

    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="Fine-tune a language model with LoRA to write product descriptions from specs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="write the synthetic starter dataset")
    data.add_argument("--n", type=int, default=240)
    data.add_argument("--seed", type=int, default=7)
    data.add_argument("--out", default="data/products.jsonl")
    data.set_defaults(func=cmd_data)

    assess = sub.add_parser("assess", parents=[product], help="check a description against its specs")
    assess.add_argument("--text", required=True, help="the description to check")
    assess.set_defaults(func=cmd_assess)

    train = sub.add_parser("train", parents=[common], help="fine-tune with LoRA and save the adapter")
    train.add_argument("--rank", type=int, default=8)
    train.add_argument("--alpha", type=float, default=16)
    train.add_argument("--dropout", type=float, default=0.05)
    train.add_argument("--targets", default="c_attn,c_proj,c_fc", help="layer names to adapt")
    train.add_argument("--steps", type=int, default=150)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--lr", type=float, default=5e-4)
    train.add_argument("--max-length", type=int, default=256)
    train.set_defaults(func=cmd_train, seed=1337)

    gen = sub.add_parser("generate", parents=[common, product], help="write a description")
    gen.add_argument("--tone", default="professional", help="professional, playful, luxury, or your own")
    gen.add_argument("--temperature", type=float, default=0.7)
    gen.add_argument("--max-tokens", type=int, default=110)
    gen.add_argument("--no-merge", action="store_true", help="keep the adapter separate (slower)")
    gen.add_argument("--compare", action="store_true", help="also show the base model's output")
    gen.set_defaults(func=cmd_generate)
    # generate uses the saved adapter by default; pass --adapter '' to use the base model alone.

    merge = sub.add_parser("merge", parents=[common], help="save a standalone model with the adapter merged in")
    merge.add_argument("--out", default="models/merged")
    merge.set_defaults(func=cmd_merge)

    rev = sub.add_parser("review", parents=[common], help="write a base vs fine-tuned report for manual review")
    rev.add_argument("--count", type=int, default=8)
    rev.add_argument("--temperature", type=float, default=0.7)
    rev.add_argument("--max-tokens", type=int, default=110)
    rev.add_argument("--out", default="reports/review.md")
    rev.set_defaults(func=cmd_review, seed=1)

    return parser


def main(argv=None) -> int:
    storage.load_env()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ModuleNotFoundError as exc:
        if exc.name not in ("torch", "transformers"):
            raise
        print(f"error: {exc.name} is needed for this command - install it with: pip install {exc.name}",
              file=sys.stderr)
        return 1
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""LoRA fine-tuning from scratch, on top of a pre-trained Hugging Face language model.

LoRA in one paragraph: instead of updating a big weight matrix W, keep it frozen
and learn a small correction B @ A, where A is (r x in) and B is (out x r) with a
small rank r. The layer computes  y = x W + scale * (x A^T) B^T  with
scale = alpha / r. B starts at zero, so training begins from the original model.
Only A and B are trained and saved. For fast inference the correction is merged
into W once (W' = W + scale * B A), and the extra layers disappear.

Needs PyTorch. `transformers` is imported only when a model is loaded.
"""
from __future__ import annotations

import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import core, storage

DEFAULT_MODEL = "gpt2"
# GPT-2 layer names: the fused q/k/v projection, the output projections and the MLP input.
# For Llama-style models use q_proj,v_proj (or more) instead.
DEFAULT_TARGETS = ("c_attn", "c_proj", "c_fc")


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_base(model_name: str = DEFAULT_MODEL, device=None):
    """Load a pre-trained causal language model and its tokenizer (a name or a local folder)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:                 # GPT-2 has no padding token
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device or pick_device())
    model.eval()
    return model, tokenizer


# --------------------------------------------------------------------------
# The LoRA layer
# --------------------------------------------------------------------------

def _is_conv1d(layer) -> bool:
    # GPT-2 uses its own Conv1D layer, which stores its weight as (in, out).
    return type(layer).__name__ == "Conv1D"


def _supported(layer) -> bool:
    return isinstance(layer, nn.Linear) or _is_conv1d(layer)


def layer_dims(layer) -> tuple[int, int]:
    """(in_features, out_features) for an nn.Linear or a GPT-2 Conv1D."""
    if isinstance(layer, nn.Linear):
        return layer.in_features, layer.out_features
    if _is_conv1d(layer):
        return layer.weight.shape[0], layer.weight.shape[1]
    raise ValueError(f"LoRA supports nn.Linear and Conv1D layers, not {type(layer).__name__}")


class LoRALinear(nn.Module):
    """A frozen layer plus a trainable low-rank update."""

    def __init__(self, base: nn.Module, r: int = 8, alpha: float = 16, dropout: float = 0.0):
        super().__init__()
        if r < 1:
            raise ValueError("the LoRA rank must be at least 1")
        in_features, out_features = layer_dims(base)
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r
        options = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.A = nn.Parameter(torch.empty(r, in_features, **options))
        self.B = nn.Parameter(torch.zeros(out_features, r, **options))   # zero: the update starts as a no-op
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)
        self.merged = False

    def forward(self, x):
        out = self.base(x)
        if self.merged:
            return out
        update = F.linear(F.linear(self.drop(x), self.A), self.B)       # x A^T B^T
        return out + self.scale * update

    def _delta(self) -> torch.Tensor:
        delta = (self.B @ self.A) * self.scale                          # (out, in)
        return delta.T if _is_conv1d(self.base) else delta              # Conv1D weights are (in, out)

    @torch.no_grad()
    def merge(self) -> None:
        """Fold the update into the base weight."""
        if not self.merged:
            self.base.weight += self._delta()
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if self.merged:
            self.base.weight -= self._delta()
            self.merged = False


def inject_lora(model: nn.Module, targets=DEFAULT_TARGETS, r: int = 8, alpha: float = 16,
                dropout: float = 0.0) -> list[str]:
    """Freeze the model and wrap every layer named in `targets` with a LoRA layer.

    Returns the names of the wrapped layers.
    """
    targets = tuple(targets)
    for parameter in model.parameters():
        parameter.requires_grad = False
    replaced = []
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name in targets and _supported(child) and not isinstance(child, LoRALinear):
                setattr(parent, child_name, LoRALinear(child, r, alpha, dropout))
                replaced.append(f"{parent_name}.{child_name}" if parent_name else child_name)
    if not replaced:
        raise ValueError(f"no layers named {', '.join(targets)} were found in this model")
    return replaced


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def lora_state(model: nn.Module) -> dict:
    """Just the A and B matrices, keyed by layer name."""
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.A"] = module.A.detach().cpu().clone()
            state[f"{name}.B"] = module.B.detach().cpu().clone()
    return state


@torch.no_grad()
def load_lora_state(model: nn.Module, state: dict) -> None:
    used = set()
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            for key, parameter in ((f"{name}.A", module.A), (f"{name}.B", module.B)):
                if key not in state:
                    raise ValueError(f"adapter is missing '{key}' - was it trained for this model?")
                if tuple(state[key].shape) != tuple(parameter.shape):
                    raise ValueError(f"adapter shape mismatch for '{key}' - check the rank and base model")
                parameter.copy_(state[key])
                used.add(key)
    unused = set(state) - used
    if unused:
        raise ValueError(f"adapter has layers this model does not: {sorted(unused)[:3]}")


@torch.no_grad()
def merge_lora(model: nn.Module) -> nn.Module:
    """Merge every adapter into its base weights and remove the LoRA layers.

    Afterwards the model is a plain pre-trained model (same size and speed as the
    original) that behaves like the fine-tuned one.
    """
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                child.merge()
                setattr(parent, name, child.base)
    return model


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def lr_at(step: int, total_steps: int, base_lr: float, warmup: int = 20, floor: float = 0.1) -> float:
    """Short linear warm-up, then cosine decay down to `floor` x base."""
    warmup = max(1, min(warmup, total_steps // 10))
    if step <= warmup:
        return base_lr * step / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))


def _tensors(batch: dict, device) -> dict:
    return {key: torch.tensor(value, dtype=torch.long, device=device) for key, value in batch.items()}


@torch.no_grad()
def evaluate_loss(model, items: list[dict], pad_id: int, batch_size: int = 8) -> float:
    """Average loss on the description tokens of tokenized examples."""
    device = next(model.parameters()).device
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(items), batch_size):
        batch = _tensors(core.pad_batch(items[start : start + batch_size], pad_id), device)
        n_tokens = int((batch["labels"][:, 1:] != -100).sum())
        if n_tokens:
            total += model(**batch).loss.item() * n_tokens
            count += n_tokens
    return total / max(1, count)


def train_lora(model, tokenizer, train_examples, val_examples, steps=150, batch_size=4, lr=5e-4,
               max_length=256, seed=1337, eval_every=50, progress=None, log=print) -> dict:
    """Train only the LoRA matrices. The loss covers the description tokens only."""
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def encode(example):
        return core.tokenize_example(tokenizer, core.example_prompt(example),
                                     core.example_completion(example), max_length)

    train_items = [encode(e) for e in train_examples]
    val_items = [encode(e) for e in val_examples]
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("no trainable parameters - call inject_lora first")
    if not train_items:
        raise ValueError("no training examples")

    batch_size = min(batch_size, len(train_items))
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    order: list[int] = []
    position = 0
    history, recent = [], []
    started = time.time()

    for step in range(1, steps + 1):
        if len(order) - position < batch_size:           # start a new pass over the data
            order = list(range(len(train_items)))
            rng.shuffle(order)
            position = 0
        chosen = order[position : position + batch_size]
        position += batch_size

        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, steps, lr)
        model.train()
        batch = _tensors(core.pad_batch([train_items[i] for i in chosen], pad_id), device)
        loss = model(**batch).loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()

        recent = (recent + [loss.item()])[-10:]
        if progress:
            progress(step, steps, loss.item())
        if step == 1 or step % eval_every == 0 or step == steps:
            val = evaluate_loss(model, val_items, pad_id) if val_items else None
            history.append({"step": step, "train": loss.item(), "val": val})
            shown = "n/a" if val is None else f"{val:.3f}"
            log(f"step {step:>4}/{steps}  train {loss.item():.3f}  val {shown}")
    model.eval()
    return {
        "history": history,
        "train_loss": sum(recent) / len(recent),
        "val_loss": history[-1]["val"] if history else None,
        "seconds": time.time() - started,
    }


def fine_tune(model_name, examples, adapter_dir, rank=8, alpha=16, dropout=0.05, targets=DEFAULT_TARGETS,
              steps=150, batch_size=4, lr=5e-4, max_length=256, val_fraction=0.1, seed=1337,
              progress=None, log=print) -> dict:
    """Load the base model, add LoRA, train, and save the adapter. Returns a summary."""
    torch.manual_seed(seed)
    model, tokenizer = load_base(model_name)
    train_examples, val_examples = core.split_examples(examples, val_fraction, seed)
    layers = inject_lora(model, targets, rank, alpha, dropout)
    trainable, total = count_parameters(model)
    log(f"LoRA added to {len(layers)} layers: {trainable:,} trainable of {total:,} parameters "
        f"({100 * trainable / total:.2f}%)")
    result = train_lora(model, tokenizer, train_examples, val_examples, steps, batch_size, lr,
                        max_length, seed, progress=progress, log=log)
    config = {
        "base_model": model_name, "rank": rank, "alpha": alpha, "dropout": dropout,
        "targets": list(targets), "steps": steps, "batch_size": batch_size, "lr": lr,
        "max_length": max_length, "val_fraction": val_fraction, "seed": seed,
        "examples": len(examples), "train_loss": result["train_loss"],
        "val_loss": result["val_loss"], "seconds": round(result["seconds"], 1),
    }
    storage.save_adapter(adapter_dir, lora_state(model), config)
    return {**config, "history": result["history"], "layers": len(layers),
            "trainable_parameters": trainable, "total_parameters": total}


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

def load_for_inference(model_name, adapter_dir=None, merge=True, device=None):
    """Load the base model, optionally with a saved adapter (merged for speed by default).

    With an adapter, the model it was trained on is used.
    """
    config = None
    if adapter_dir:
        weights, config = storage.load_adapter(adapter_dir)
        if model_name and model_name != config["base_model"]:
            raise ValueError(f"this adapter was trained on '{config['base_model']}', not '{model_name}'")
        model_name = config["base_model"]
    model, tokenizer = load_base(model_name or DEFAULT_MODEL, device)
    if config:
        inject_lora(model, tuple(config["targets"]), config["rank"], config["alpha"], 0.0)
        load_lora_state(model, weights)
        if merge:
            merge_lora(model)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_description(model, tokenizer, prompt: str, max_new_tokens: int = 110,
                         temperature: float = 0.7, top_p: float = 0.9, seed: int | None = None) -> str:
    """Continue the prompt (it ends with 'Description:') and return the cleaned description."""
    if temperature < 0:
        raise ValueError("temperature cannot be negative")
    model.eval()
    device = next(model.parameters()).device
    config = model.config
    limit = getattr(config, "n_positions", None) or getattr(config, "max_position_embeddings", 1024)
    if max_new_tokens >= limit:
        raise ValueError(f"max_new_tokens must be below the model's context size ({limit})")
    ids = tokenizer.encode(prompt)[-(limit - max_new_tokens):]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    settings = {"max_new_tokens": max_new_tokens, "pad_token_id": pad_id, "eos_token_id": tokenizer.eos_token_id}
    if temperature == 0:
        settings["do_sample"] = False
    else:
        settings.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=50)
    if seed is not None:
        torch.manual_seed(seed)
    output = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), **settings)
    new_ids = output[0, input_ids.shape[1]:].tolist()
    return core.finish_description(tokenizer.decode(new_ids, skip_special_tokens=True))


def export_merged(adapter_dir: str, out_dir: str, model_name: str | None = None) -> str:
    """Save a standalone model with the adapter merged in (loadable like any Hugging Face model)."""
    model, tokenizer = load_for_inference(model_name, adapter_dir, merge=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir


# --------------------------------------------------------------------------
# Review: base model vs fine-tuned, side by side
# --------------------------------------------------------------------------

def review(adapter_dir: str, dataset: list[dict], count: int = 8, temperature: float = 0.7,
           max_new_tokens: int = 110, seed: int = 1) -> str:
    """Generate for held-out products with and without the adapter and return a Markdown report."""
    weights, config = storage.load_adapter(adapter_dir)
    train, held_out = core.split_examples(dataset, config.get("val_fraction", 0.1), config.get("seed", 1337))
    sample = held_out[:count]
    reference = core.ngram_set([e["description"] for e in train])
    model, tokenizer = load_base(config["base_model"])

    def run():
        results = []
        for example in sample:
            text = generate_description(model, tokenizer, core.example_prompt(example),
                                        max_new_tokens, temperature, seed=seed)
            metrics = core.assess(text, example["specs"], example["keywords"], example["name"],
                                  example["category"], reference)
            results.append({"text": text, "metrics": metrics})
        return results

    base_results = run()                                  # before the adapter is added
    inject_lora(model, tuple(config["targets"]), config["rank"], config["alpha"], 0.0)
    load_lora_state(model, weights)
    merge_lora(model)
    tuned_results = run()
    rows = [{"example": e, "base": b, "tuned": t} for e, b, t in zip(sample, base_results, tuned_results)]
    meta = {"base model": config["base_model"], "adapter": adapter_dir, "rank": config["rank"],
            "held-out products reviewed": len(rows), "temperature": temperature, "seed": seed}
    return core.format_review(rows, meta)

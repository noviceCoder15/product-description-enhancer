[![Built with askThili](https://msrit.ask.thili.ai/api/v1/public/projects/wv25jrkwyd/badge.svg)](https://msrit.ask.thili.ai/b/wv25jrkwyd)
# Product Description Enhancer

Turn basic product specs into a compelling, keyword-aware description. The tool fine-tunes
a pre-trained language model (GPT-2) with **LoRA**, written from scratch in PyTorch, then
merges the adapter into the model for fast inference. It comes with a Streamlit app and a
command-line interface, and runs on a laptop CPU with no API keys.

```
Product: Aero Wireless Earbud Set
Category: audio
Specs: Battery life: 12 hours; Water resistance: IPX5; Charging: USB-C
Keywords: wireless earbuds, sports earbuds
Tone: playful
Description:  ->  the fine-tuned model writes the description
```

## What LoRA does

Instead of updating a big weight matrix `W`, LoRA keeps it frozen and learns a small
correction `B @ A`, where `A` is `r x in` and `B` is `out x r` with a small rank `r`. The layer
computes `y = x W + (alpha / r) * x A^T B^T`. `B` starts at zero, so training begins from the
original model. Only `A` and `B` are trained and saved (about 1% of GPT-2's parameters).
For inference the correction is merged once (`W' = W + (alpha / r) * B A`) and the extra
layers disappear.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pytest

python -m app.cli train                 # fine-tune GPT-2 with LoRA (the first run downloads GPT-2)
python -m app.cli review                # base vs fine-tuned report for manual review
streamlit run app/main.py               # the app, at http://localhost:8501
```

The starter dataset is already in `data/products.jsonl`. Re-create it with
`python -m app.cli data`.

## The Streamlit app

- **Generate:** enter a product name, category, specs (one `name: value` per line), target
  keywords and a tone. Each result shows automatic checks: the share of specs and keywords it
  used, a warning for any number that is not in your input, and the word count. Tick
  *Also show the base model's output* to compare with the model before fine-tuning.
- **Fine-tune:** train an adapter on the bundled dataset or on your own CSV or JSONL upload,
  with a progress bar. A second button merges the selected adapter into a standalone model.
- **Dataset:** browse the bundled examples.
- **Sidebar:** pick the adapter, the temperature, the length and a seed, and choose whether to
  merge the adapter for faster inference.

## Command line

| Command | What it does |
|---|---|
| `data` | Write the synthetic starter dataset (`--n`, `--seed`, `--out`). |
| `train` | Fine-tune with LoRA and save the adapter to `adapters/<name>/`. |
| `generate` | Write a description: `--name`, `--spec "Battery life: 12 hours"` (repeat), `--keywords`, `--tone`, `--compare`. |
| `review` | Generate for held-out products with and without the adapter and write `reports/review.md`. |
| `merge` | Save a standalone model with the adapter merged in, to `models/merged`. |
| `assess` | Score any description against its specs and keywords (needs no model). |

`train` options (defaults in brackets): `--model` (gpt2), `--adapter` (default), `--rank` (8),
`--alpha` (16), `--steps` (150), `--batch-size` (4), `--lr` (5e-4), `--targets`
(c_attn,c_proj,c_fc). Defaults can also be set in a `.env` file (copy `.env.example`).

Example:

```bash
python -m app.cli generate --name "Aero Wireless Earbud Set" --category audio \
  --spec "Battery life: 12 hours" --spec "Water resistance: IPX5" --spec "Charging: USB-C" \
  --keywords "wireless earbuds, sports earbuds" --tone playful --compare
```

`assess` runs without PyTorch. Real output:

```
$ python -m app.cli assess --name "Aero Earbud Set" --category audio \
    --spec "Battery life: 12 hours" --spec "Water resistance: IPX5" \
    --keywords "wireless earbuds, sports earbuds" \
    --text "These wireless earbuds last 30 hours and sound great."
specs mentioned : 0%
keywords used   : 50%
words           : 9
repetition      : 0%
flags           : short (9 words); leaves out specs: Battery life, Water resistance;
                  leaves out keywords: sports earbuds; numbers not in the input: 30
```

## Milestones

| Milestone | Where |
|---|---|
| Dataset of product specs and descriptions | `data/products.jsonl` (240 examples), `synthesize_dataset` in `app/core.py` |
| LoRA injection and training on a pre-trained LLM | `LoRALinear`, `inject_lora`, `train_lora`, `fine_tune` in `app/llm.py` |
| Streamlit UI | `app/main.py` |
| Assess creativity and relevance | `assess` in `app/core.py`, `review` in `app/llm.py`, the manual review below |
| Adapter merging for inference | `LoRALinear.merge`, `merge_lora`, `export_merged` in `app/llm.py` |
| Stretch: tone and keywords | Tone and Keywords lines in the prompt; both are in the UI and the dataset |
| Stretch: upload your own dataset | Fine-tune tab (CSV or JSONL) |
| Stretch: other pre-trained models | `--model` (and the Base model box in the app) |

## How the code fits together

- **`app/core.py`** (no PyTorch): prompt format, dataset parsing, the synthetic dataset, token
  labelling, and the quality checks.
- **`app/llm.py`** (PyTorch): the LoRA layer, injection, training loop, merging, generation.
- **`app/storage.py`:** datasets, adapter files, reports.
- **`app/cli.py`** and **`app/main.py`:** the command line and the Streamlit app.

Training details: the model sees `prompt + description + end token`, but the loss is computed
only on the description, so it learns to write descriptions and not to copy the prompt.
GPT-2 stores its layers as `Conv1D` (weights shaped `in x out`), which the LoRA layer handles
alongside `nn.Linear`. Only the adapter is saved, so an adapter is a few MB.

## Merging

`merge_lora` folds each `B A` into its base weight and removes the LoRA layers, so inference
costs exactly what the original model costs. `LoRALinear.unmerge` reverses it. The tests check
that merged and unmerged outputs match. `python -m app.cli merge` saves a standalone model that
loads like any Hugging Face model (use its folder as `--model`).

## Assessing quality

**Automatic checks** (`core.assess`) are cheap and catch the usual failures: specs or keywords
left out, numbers that are not in the input (a sign of invented facts), very short or very
long text, repetition, and *novelty* (how much of the text is new compared with the training
descriptions).

**Manual review** is the real test. Run `python -m app.cli review`, open `reports/review.md`,
and for each held-out product read the base and fine-tuned descriptions side by side. Score
each from 1 to 5 for **relevance** (matches the specs, invents nothing) and **creativity**
(engaging, not just a template), and write notes. Keep the filled-in report in the repo as
evidence.

## Things to know

- **The starter dataset is synthetic.** Its descriptions are assembled from templates written
  for this project, which is why they always match their specs. The model will therefore write
  fairly formulaic copy, and the *novelty* score will be low. Real marketing copy gives more
  varied results: upload your own in the Fine-tune tab, or edit `data/products.jsonl`.
- **GPT-2 is small.** Even fine-tuned, it can invent a feature or get a number wrong. Read the
  flags under each result and review the text before using it in a shop.
- **Training time.** Expect roughly 10 to 25 minutes for the default 150 steps on a laptop CPU
  (an estimate, so check on your machine). The first run also downloads GPT-2 (about 500 MB).
  Use `--steps 50` for a quick trial.
- **Other models.** `distilgpt2` uses the same layer names, so it should work with the default
  settings (I have not tried it). Models with other layer names
  need `--targets`, for example `--targets q_proj,v_proj` for Llama-style models.
- **Spelling.** The starter dataset uses British spelling, so the model will too.

## Docker

```bash
docker build -t product-description-enhancer .
docker run -p 8501:8501 \
  -v "$PWD/adapters:/app/adapters" -v "$PWD/hf-cache:/root/.cache/huggingface" \
  product-description-enhancer
```

The two volumes keep your trained adapters and the downloaded model between runs.

## Tests

```bash
pytest
```

The tests cover prompts, dataset parsing, the synthetic dataset, label masking, the quality
checks, the LoRA layer (zero start, trainable parameters, merge equals unmerged for both
`Linear` and GPT-2 `Conv1D`), injection into a small GPT-2, saving and loading adapters,
generation, every command that needs no model, and the Streamlit app. `.github/workflows/ci.yml`
runs them on GitHub after each push.

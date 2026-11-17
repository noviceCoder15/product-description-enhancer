"""Streamlit app: turn product specs into descriptions with a LoRA fine-tuned model.

Run it from the project folder with:  streamlit run app/main.py
Heavy libraries (PyTorch, transformers) are imported only when a button needs them.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app import core, storage  # noqa: E402

st.set_page_config(page_title="Product Description Enhancer", page_icon="\U0001F6CD", layout="wide")
storage.load_env()

BASE_MODEL = os.environ.get("BASE_MODEL", "gpt2")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "adapters")
DATA_PATH = os.environ.get("DATA_PATH", "data/products.jsonl")
NO_ADAPTER = "(none - base model)"


@st.cache_resource(show_spinner=False)
def load_model(model_name: str, adapter_dir: str, merge: bool, stamp: float):
    """Load once and reuse. `stamp` changes when an adapter is retrained."""
    from app import llm

    return llm.load_for_inference(model_name or None, adapter_dir or None, merge)


def write(model_name, adapter_dir, merge, prompt, temperature, max_tokens, seed):
    from app import llm

    model, tokenizer = load_model(model_name, adapter_dir, merge, storage.adapter_stamp(adapter_dir))
    return llm.generate_description(model, tokenizer, prompt, max_tokens, temperature, seed=seed)


def show_result(title, text, specs, keywords, name, category):
    st.subheader(title)
    if not text:
        st.warning("The model returned no text. Try again, or raise the temperature.")
        return
    st.write(text)
    metrics = core.assess(text, specs, keywords, name, category)
    first, second, third = st.columns(3)
    first.metric("Specs mentioned", f"{metrics['spec_coverage']:.0%}")
    second.metric("Keywords used", f"{metrics['keyword_coverage']:.0%}")
    third.metric("Words", metrics["words"])
    for flag in metrics["flags"]:
        st.warning(flag)
    if metrics["ok"]:
        st.success("Mentions every spec and keyword, and invents no numbers.")


st.title("Product Description Enhancer")
st.caption("Turn basic product specs into a compelling, keyword-aware description with a "
           "GPT-2 model fine-tuned using LoRA.")

# --------------------------------------------------------------------------
# Sidebar: model and generation settings
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("Model")
    adapters = storage.list_adapters(ADAPTER_DIR)
    choice = st.selectbox("LoRA adapter", [NO_ADAPTER] + adapters, index=1 if adapters else 0,
                          help="Train one in the Fine-tune tab or with `python -m app.cli train`.")
    adapter_dir = "" if choice == NO_ADAPTER else str(Path(ADAPTER_DIR) / choice)
    base_model = st.text_input("Base model", value=BASE_MODEL,
                               help="Used when no adapter is selected. An adapter always uses the model it was trained on.")
    merge = st.checkbox("Merge the adapter for faster inference", value=True)

    st.header("Generation")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05,
                            help="0 gives the most likely text every time. Higher values are more varied.")
    max_tokens = st.slider("Maximum length (tokens)", 40, 200, 110, 10)
    seed_value = int(st.number_input("Seed (0 = random)", min_value=0, value=0, step=1))

model_name = "" if adapter_dir else base_model.strip()
adapter_base = storage.read_adapter_config(adapter_dir)["base_model"] if adapter_dir else base_model.strip()

tab_generate, tab_train, tab_data = st.tabs(["Generate", "Fine-tune", "Dataset"])

# --------------------------------------------------------------------------
# Generate
# --------------------------------------------------------------------------

with tab_generate:
    left, right = st.columns(2)
    with left:
        name = st.text_input("Product name", "Aero Wireless Earbud Set")
        category = st.text_input("Category", "audio")
        specs_text = st.text_area("Specs (one per line: name: value)", height=150,
                                  value="Battery life: 12 hours\nWater resistance: IPX5\nCharging: USB-C")
        keywords_text = st.text_input("Target keywords (comma separated)", "wireless earbuds, sports earbuds")
        tone = st.selectbox("Tone", list(core.TONES))
        compare = st.checkbox("Also show the base model's output", disabled=not adapter_dir)
        go = st.button("Generate description", type="primary")

    with right:
        if go:
            try:
                specs = core.parse_specs(specs_text)
                if not name.strip():
                    raise ValueError("enter a product name")
                if not specs:
                    raise ValueError("enter at least one spec, like 'Battery life: 12 hours'")
            except ValueError as exc:
                st.error(str(exc))
            else:
                keywords = core.parse_keywords(keywords_text)
                prompt = core.build_prompt(name, category, specs, keywords, tone)
                seed = seed_value or None
                try:
                    with st.spinner("Writing (the first run loads the model, which can take a while)..."):
                        text = write(model_name, adapter_dir, merge, prompt, temperature, max_tokens, seed)
                    show_result("Enhanced description" if adapter_dir else "Base model output",
                                text, specs, keywords, name, category)
                    if compare and adapter_dir:
                        with st.spinner("Writing with the base model..."):
                            base_text = write(adapter_base, "", True, prompt, temperature, max_tokens, seed)
                        show_result("Base model, for comparison", base_text, specs, keywords, name, category)
                except ModuleNotFoundError as exc:
                    st.error(f"{exc.name} is not installed: run `pip install -r requirements.txt`.")
                except (OSError, ValueError, RuntimeError) as exc:
                    st.error(f"Could not generate: {exc}")
        else:
            st.info("Fill in the product details and press **Generate description**. "
                    "The checks below each result show whether it used every spec and keyword.")

# --------------------------------------------------------------------------
# Fine-tune
# --------------------------------------------------------------------------

with tab_train:
    st.markdown("Fine-tune the base model with **LoRA**: the model stays frozen and only small adapter "
                "matrices are trained, so it works on a laptop CPU.")
    source = st.radio("Training data", ["Bundled starter dataset", "Upload my own (CSV or JSONL)"], horizontal=True)
    examples, problems = [], []
    if source.startswith("Bundled"):
        try:
            examples, problems = storage.read_dataset(DATA_PATH)
        except FileNotFoundError as exc:
            st.error(f"{exc}. Create it with `python -m app.cli data`.")
    else:
        upload = st.file_uploader("Dataset file", type=["csv", "jsonl"])
        st.caption("Fields: name, specs ('Key: value; Key: value'), description, and optionally "
                   "category, keywords (comma separated) and tone. Use a few dozen examples or more.")
        if upload is not None:
            try:
                examples, problems = core.parse_dataset_text(upload.getvalue().decode("utf-8-sig"),
                                                             upload.name.rsplit(".", 1)[-1])
            except (ValueError, UnicodeDecodeError) as exc:
                st.error(f"Could not read the file: {exc}")
    if examples:
        st.success(f"{len(examples)} usable examples")
    if problems:
        st.warning(f"{len(problems)} rows were skipped")
        st.code("\n".join(problems[:10]))

    first, second, third, fourth, fifth = st.columns(5)
    steps = int(first.number_input("Steps", 10, 1000, 150, 10))
    batch_size = int(second.number_input("Batch size", 1, 16, 4))
    rank = int(third.number_input("LoRA rank", 1, 64, 8))
    alpha = float(fourth.number_input("LoRA alpha", 1, 128, 16))
    lr = float(fifth.number_input("Learning rate", 1e-5, 1e-2, 5e-4, format="%.5f"))
    adapter_name = st.text_input("Adapter name", "default")

    if st.button("Start fine-tuning", type="primary", disabled=len(examples) < 10):
        try:
            from app import llm

            target = str(Path(ADAPTER_DIR) / storage.safe_name(adapter_name))
            bar = st.progress(0.0, text="Loading the base model...")

            def on_step(step, total, loss):
                bar.progress(step / total, text=f"Step {step}/{total}  loss {loss:.3f}")

            summary = llm.fine_tune(base_model.strip() or "gpt2", examples, target, rank=rank, alpha=alpha,
                                    steps=steps, batch_size=batch_size, lr=lr, progress=on_step,
                                    log=lambda *_: None)
            bar.progress(1.0, text="Done")
            st.success(f"Saved adapter '{Path(target).name}' in {summary['seconds']} seconds. "
                       "Choose it in the sidebar to use it.")
            st.write(f"Trained {summary['trainable_parameters']:,} of {summary['total_parameters']:,} parameters "
                     f"({100 * summary['trainable_parameters'] / summary['total_parameters']:.2f}%).")
            if summary["val_loss"] is not None:
                st.write(f"Validation loss: {summary['val_loss']:.3f}")
        except ModuleNotFoundError as exc:
            st.error(f"{exc.name} is not installed: run `pip install -r requirements.txt`.")
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(f"Training failed: {exc}")

    st.divider()
    st.markdown("**Merge for deployment.** Save a standalone model with the selected adapter merged in. "
                "It has no extra layers, so it runs at the speed of the original model.")
    out_dir = st.text_input("Save the merged model to", "models/merged")
    if st.button("Merge and save", disabled=not adapter_dir):
        try:
            from app import llm

            with st.spinner("Merging..."):
                saved = llm.export_merged(adapter_dir, out_dir)
            st.success(f"Saved to {saved}. Use it as the base model name to load it.")
        except ModuleNotFoundError as exc:
            st.error(f"{exc.name} is not installed: run `pip install -r requirements.txt`.")
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(f"Could not merge: {exc}")

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

with tab_data:
    try:
        bundled, bundled_problems = storage.read_dataset(DATA_PATH)
    except FileNotFoundError as exc:
        st.error(f"{exc}. Create it with `python -m app.cli data`.")
    else:
        summary = core.dataset_summary(bundled)
        st.write(f"**{summary['examples']} examples**, about {summary['average_words']} words per description.")
        st.write("Categories: " + ", ".join(f"{k} ({v})" for k, v in summary["categories"].items()))
        st.write("Tones: " + ", ".join(f"{k} ({v})" for k, v in summary["tones"].items()))
        st.dataframe([{"name": e["name"], "tone": e["tone"], "specs": core.format_specs(e["specs"]),
                       "description": e["description"]} for e in bundled[:50]])
        st.caption("The starter dataset is synthetic: templated descriptions written for this project. "
                   "Real marketing copy gives more varied results. Upload your own in the Fine-tune tab.")

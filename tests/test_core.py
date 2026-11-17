import json
import math

import pytest

from app import core

SPECS = {"Battery life": "12 hours", "Water resistance": "IPX5", "Charging": "USB-C"}
EXAMPLE = {
    "name": "Aero Earbud Set", "category": "audio", "specs": SPECS,
    "keywords": ["wireless earbuds", "sports earbuds"], "tone": "playful",
    "description": "Say hello to the Aero Earbud Set, perfect wireless earbuds! "
                   "Up to 12 hours of battery life and an IPX5 rating keep you going. "
                   "Charge via USB-C. Grab yours today for sports earbuds you will love!",
}


class FakeTokenizer:
    """Character-level stand-in for a Hugging Face tokenizer."""

    eos_token_id = 1
    pad_token_id = 1

    def encode(self, text):
        return [2 + (ord(c) % 60) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + (i % 26)) for i in ids if not (skip_special_tokens and i == 1))


# ---- specs, keywords and the prompt ----------------------------------------

def test_parse_specs_accepts_lines_and_semicolons():
    assert core.parse_specs("Battery life: 12 hours\nCharging = USB-C; Weight: 48 g") == {
        "Battery life": "12 hours", "Charging": "USB-C", "Weight": "48 g",
    }


def test_parse_specs_keeps_colons_inside_values():
    assert core.parse_specs("Ratio: 16:9") == {"Ratio": "16:9"}


def test_parse_specs_rejects_lines_without_a_value():
    for bad in ("just words", "Battery life:", ": 12 hours"):
        with pytest.raises(ValueError):
            core.parse_specs(bad)


def test_parse_keywords_cleans_and_deduplicates():
    assert core.parse_keywords(" a, B ;a\n c ") == ["a", "B", "c"]
    assert core.parse_keywords(None) == []
    assert core.parse_keywords("") == []
    assert len(core.parse_keywords(",".join(str(i) for i in range(20)))) == 6


def test_prompt_has_a_fixed_layout_and_ends_at_description():
    prompt = core.build_prompt("Aero", "audio", SPECS, ["a", "b"], "Playful")
    assert prompt.splitlines() == [
        "Product: Aero", "Category: audio",
        "Specs: Battery life: 12 hours; Water resistance: IPX5; Charging: USB-C",
        "Keywords: a, b", "Tone: playful", "Description:",
    ]
    assert "Keywords: none" in core.build_prompt("Aero", "audio", SPECS, [], "playful")


def test_completion_starts_with_a_space_so_tokens_match_at_inference():
    assert core.example_completion(EXAMPLE).startswith(" Say hello")


# ---- reading datasets --------------------------------------------------------

def test_normalise_record_accepts_dict_or_string_specs():
    as_dict, problems = core.normalise_record({**EXAMPLE})
    assert not problems and as_dict["specs"] == SPECS
    as_text, problems = core.normalise_record({**EXAMPLE, "specs": "Battery life: 12 hours; Charging: USB-C"})
    assert not problems and as_text["specs"] == {"Battery life": "12 hours", "Charging": "USB-C"}


def test_normalise_record_fills_defaults_and_is_case_insensitive():
    example, problems = core.normalise_record(
        {"NAME": "Lamp", "Specs": "Power: 8 W", "Description": "A reliable lamp for every desk and every room."})
    assert not problems
    assert example["category"] == "general" and example["tone"] == "professional" and example["keywords"] == []


def test_normalise_record_reports_every_problem():
    example, problems = core.normalise_record({"name": "", "specs": "", "description": "short"})
    assert example is None
    assert "missing name" in problems and "missing specs" in problems
    assert any("description" in p for p in problems)


def test_parse_jsonl_skips_bad_rows_and_says_why():
    good = json.dumps(EXAMPLE)
    text = "\n".join([good, "not json", json.dumps([1, 2]), json.dumps({"name": "x"}), "", good])
    examples, problems = core.parse_dataset_text(text, ".jsonl")
    assert len(examples) == 2
    assert any("line 2" in p and "JSON" in p for p in problems)
    assert any("line 3" in p for p in problems)
    assert any("line 4" in p for p in problems)


def test_parse_csv_reads_specs_and_keywords_from_cells():
    text = ("name,category,specs,keywords,tone,description\n"
            '"Aero Earbuds",audio,"Battery life: 12 hours; Charging: USB-C","wireless earbuds, sports earbuds",playful,'
            '"Say hello to the Aero Earbuds, with 12 hours of battery life and USB-C charging."\n'
            ',audio,"Charging: USB-C",,playful,"A description without a product name goes here."\n')
    examples, problems = core.parse_dataset_text(text, "csv")
    assert len(examples) == 1
    assert examples[0]["specs"] == {"Battery life": "12 hours", "Charging": "USB-C"}
    assert examples[0]["keywords"] == ["wireless earbuds", "sports earbuds"]
    assert any("row 3" in p and "missing name" in p for p in problems)


def test_unknown_dataset_format_is_rejected():
    with pytest.raises(ValueError):
        core.parse_dataset_text("{}", "xlsx")


def test_split_examples_is_reproducible_and_disjoint():
    items = [{"i": i} for i in range(20)]
    train, val = core.split_examples(items, 0.1, seed=3)
    assert len(val) == 2 and len(train) == 18
    assert not [i for i in train if i in val]
    assert core.split_examples(items, 0.1, seed=3) == (train, val)
    with pytest.raises(ValueError):
        core.split_examples(items[:1])


# ---- the starter dataset -----------------------------------------------------

def test_synthetic_dataset_is_reproducible_and_balanced():
    data = core.synthesize_dataset(60, seed=1)
    assert data == core.synthesize_dataset(60, seed=1)
    assert data != core.synthesize_dataset(60, seed=2)
    assert {e["category"] for e in data} == set(core.CATALOG)
    assert {e["tone"] for e in data} == set(core.TONES)


def test_every_synthetic_description_honours_its_inputs():
    for example in core.synthesize_dataset(120, seed=5):
        assert core.normalise_record(example) == (example, [])
        result = core.assess(example["description"], example["specs"], example["keywords"],
                             example["name"], example["category"])
        assert result["ok"], (result["flags"], example["description"])


# ---- tokenization for training ------------------------------------------------

def test_loss_is_only_on_the_description():
    item = core.tokenize_example(FakeTokenizer(), "PROMPT:", " the answer", max_length=64)
    prompt_length = len("PROMPT:")
    assert item["input_ids"][-1] == 1 and item["labels"][-1] == 1            # ends with the end token
    assert item["labels"][:prompt_length] == [-100] * prompt_length
    assert item["labels"][prompt_length:] == item["input_ids"][prompt_length:]
    assert len(item["input_ids"]) == prompt_length + len(" the answer") + 1


def test_long_examples_are_truncated_but_the_prompt_is_never_cut():
    item = core.tokenize_example(FakeTokenizer(), "PROMPT:", " " + "word " * 100, max_length=40)
    assert len(item["input_ids"]) == 40
    with pytest.raises(ValueError):
        core.tokenize_example(FakeTokenizer(), "x" * 50, " answer", max_length=40)


def test_pad_batch_pads_inputs_masks_and_labels():
    batch = core.pad_batch([{"input_ids": [5, 6, 7], "labels": [-100, 6, 7]},
                            {"input_ids": [8], "labels": [8]}], pad_id=1)
    assert batch["input_ids"] == [[5, 6, 7], [8, 1, 1]]
    assert batch["attention_mask"] == [[1, 1, 1], [1, 0, 0]]
    assert batch["labels"] == [[-100, 6, 7], [8, -100, -100]]


# ---- cleaning and measuring generated text ---------------------------------------

def test_finish_description_stops_at_a_new_example():
    raw = " A great product. It lasts.\nProduct: Another thing\nCategory: x"
    assert core.finish_description(raw) == "A great product. It lasts."
    assert core.finish_description("Good.<|endoftext|>More text") == "Good."
    assert core.finish_description("First paragraph.\n\nSecond paragraph.") == "First paragraph."


def test_finish_description_drops_a_dangling_fragment():
    assert core.finish_description("It is light. It lasts all da") == "It is light."
    assert core.finish_description("no sentence end here") == "no sentence end here"
    assert core.finish_description("   ") == ""


def test_spec_coverage_finds_values_and_numbers():
    coverage, missing = core.spec_coverage("Up to 12 hours with IPX5 and USB-C.", SPECS)
    assert coverage == 1.0 and missing == []
    coverage, missing = core.spec_coverage("A 12-hour battery and USB-C.", SPECS)
    assert missing == ["Water resistance"] and coverage == pytest.approx(2 / 3)


def test_unsupported_numbers_flag_invented_figures():
    found = core.unsupported_numbers("Lasts 30 hours with IPX5.", ["Aero", "Battery life: 12 hours; IPX5"])
    assert found == ["30"]


def test_repetition_and_novelty_measures():
    assert core.repetition_rate("a b c d e f g") == 0.0
    assert core.repetition_rate("go go go go go go") > 0.5
    reference = core.ngram_set(["the quick brown fox jumps over the lazy dog"])
    assert core.copy_rate("the quick brown fox jumps over", reference) == 1.0
    assert core.copy_rate("a completely different sentence appears here", reference) == 0.0


def test_assess_flags_each_kind_of_problem():
    good = core.assess(EXAMPLE["description"], SPECS, EXAMPLE["keywords"], "Aero Earbud Set", "audio")
    assert good["ok"] and good["spec_coverage"] == 1.0 and good["keyword_coverage"] == 1.0

    bad = core.assess("Lasts 40 hours. Lasts 40 hours. Lasts 40 hours. Lasts 40 hours.", SPECS,
                      EXAMPLE["keywords"], "Aero Earbud Set", "audio")
    assert not bad["ok"]
    text = " | ".join(bad["flags"])
    for expected in ("short", "leaves out specs", "leaves out keywords", "numbers not in the input", "repeats"):
        assert expected in text


def test_assess_reports_novelty_only_when_given_training_text():
    assert core.assess("Some text.", {}, [], "n", "c")["novelty"] is None
    reference = core.ngram_set([EXAMPLE["description"]])
    result = core.assess(EXAMPLE["description"], SPECS, [], "Aero Earbud Set", "audio", reference)
    assert result["novelty"] == pytest.approx(0.0)


def test_review_report_lists_every_example_and_summary_row():
    metrics = core.assess(EXAMPLE["description"], SPECS, EXAMPLE["keywords"], "Aero Earbud Set", "audio")
    empty = core.assess("", SPECS, EXAMPLE["keywords"], "Aero Earbud Set", "audio")
    rows = [{"example": EXAMPLE, "base": {"text": "", "metrics": empty},
             "tuned": {"text": EXAMPLE["description"], "metrics": metrics}}]
    report = core.format_review(rows, {"base model": "gpt2"})
    assert "# Product description review" in report and "base model:** gpt2" in report
    assert "| specs mentioned | 0% | 100% |" in report
    assert "### 1. Aero Earbud Set (playful)" in report
    assert "(empty)" in report and "Relevance (1-5)" in report


# ---- LoRA: the layer, injection, merging (needs PyTorch) ----------------------------

def torch_llm():
    torch = pytest.importorskip("torch")
    from app import llm

    return torch, llm


def make_conv1d(torch, n_in, n_out):
    class Conv1D(torch.nn.Module):          # same layout as GPT-2's Conv1D: weight is (in, out)
        def __init__(self, nf, nx):
            super().__init__()
            self.nf = nf
            self.weight = torch.nn.Parameter(torch.randn(nx, nf) * 0.1)
            self.bias = torch.nn.Parameter(torch.zeros(nf))

        def forward(self, x):
            return x @ self.weight + self.bias

    return Conv1D(n_out, n_in)


def random_update(llm, model):
    for module in model.modules():
        if isinstance(module, llm.LoRALinear):
            module.B.data.normal_(0, 0.05)


def test_lora_starts_as_a_no_op():
    torch, llm = torch_llm()
    torch.manual_seed(0)
    base = torch.nn.Linear(16, 24)
    lora = llm.LoRALinear(base, r=4, alpha=8)
    x = torch.randn(3, 5, 16)
    assert torch.allclose(lora(x), base(x))


def test_only_the_adapter_matrices_are_trainable():
    torch, llm = torch_llm()
    lora = llm.LoRALinear(torch.nn.Linear(16, 24), r=4)
    trainable = {name for name, p in lora.named_parameters() if p.requires_grad}
    assert trainable == {"A", "B"}
    assert lora.A.shape == (4, 16) and lora.B.shape == (24, 4)


def test_rank_must_be_positive_and_layer_must_be_supported():
    torch, llm = torch_llm()
    with pytest.raises(ValueError):
        llm.LoRALinear(torch.nn.Linear(4, 4), r=0)
    with pytest.raises(ValueError):
        llm.LoRALinear(torch.nn.ReLU(), r=2)


def check_merge_round_trip(kind):
    torch, llm = torch_llm()
    torch.manual_seed(1)
    base = torch.nn.Linear(16, 24) if kind == "linear" else make_conv1d(torch, 16, 24)
    lora = llm.LoRALinear(base, r=4, alpha=8)
    lora.B.data.normal_(0, 0.1)
    x = torch.randn(2, 7, 16)
    before = lora(x)
    original_weight = base.weight.detach().clone()
    lora.merge()
    assert lora.merged and torch.allclose(lora(x), before, atol=1e-5)
    assert not torch.allclose(base.weight, original_weight)
    lora.unmerge()
    assert not lora.merged and torch.allclose(base.weight, original_weight, atol=1e-6)
    assert torch.allclose(lora(x), before, atol=1e-5)


def test_merging_a_linear_layer_gives_the_same_output_and_can_be_undone():
    check_merge_round_trip("linear")


def test_merging_a_gpt2_conv1d_layer_gives_the_same_output_and_can_be_undone():
    check_merge_round_trip("conv1d")


def tiny_gpt2():
    pytest.importorskip("transformers")
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    config = GPT2Config(vocab_size=64, n_positions=128, n_embd=32, n_layer=2, n_head=2,
                        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    return GPT2LMHeadModel(config).eval()


def test_injection_wraps_the_gpt2_layers_without_changing_the_output():
    torch, llm = torch_llm()
    model = tiny_gpt2()
    ids = torch.randint(0, 64, (2, 10))
    with torch.no_grad():
        before = model(ids).logits
    layers = llm.inject_lora(model, llm.DEFAULT_TARGETS, r=4, alpha=8)
    assert len(layers) == 8                                   # c_attn, attn c_proj, c_fc, mlp c_proj x 2 layers
    with torch.no_grad():
        after = model(ids).logits
    assert torch.allclose(before, after, atol=1e-5)
    trainable, total = llm.count_parameters(model)
    assert 0 < trainable < total * 0.5
    assert all(name.endswith((".A", ".B")) for name, p in model.named_parameters() if p.requires_grad)


def test_unknown_target_names_are_reported():
    _, llm = torch_llm()
    with pytest.raises(ValueError):
        llm.inject_lora(tiny_gpt2(), ("no_such_layer",))


def test_merged_model_matches_the_adapter_model_and_has_no_lora_layers_left():
    torch, llm = torch_llm()
    model = tiny_gpt2()
    llm.inject_lora(model, llm.DEFAULT_TARGETS, r=4, alpha=8)
    random_update(llm, model)
    ids = torch.randint(0, 64, (2, 10))
    with torch.no_grad():
        with_adapter = model(ids).logits
    llm.merge_lora(model)
    assert not any(isinstance(m, llm.LoRALinear) for m in model.modules())
    with torch.no_grad():
        merged = model(ids).logits
    assert torch.allclose(with_adapter, merged, atol=1e-4)


def test_saved_adapter_state_restores_the_same_model():
    torch, llm = torch_llm()
    import copy

    original = tiny_gpt2()
    first, second = copy.deepcopy(original), copy.deepcopy(original)
    llm.inject_lora(first, llm.DEFAULT_TARGETS, r=4, alpha=8)
    random_update(llm, first)
    state = llm.lora_state(first)
    llm.inject_lora(second, llm.DEFAULT_TARGETS, r=4, alpha=8)
    llm.load_lora_state(second, state)
    ids = torch.randint(0, 64, (1, 9))
    with torch.no_grad():
        assert torch.allclose(first(ids).logits, second(ids).logits, atol=1e-6)


def test_adapter_with_the_wrong_rank_is_rejected():
    _, llm = torch_llm()
    import copy

    original = tiny_gpt2()
    first, second = copy.deepcopy(original), copy.deepcopy(original)
    llm.inject_lora(first, llm.DEFAULT_TARGETS, r=4, alpha=8)
    llm.inject_lora(second, llm.DEFAULT_TARGETS, r=2, alpha=8)
    with pytest.raises(ValueError):
        llm.load_lora_state(second, llm.lora_state(first))


def test_training_updates_only_the_adapter():
    torch, llm = torch_llm()
    model = tiny_gpt2()
    llm.inject_lora(model, llm.DEFAULT_TARGETS, r=4, alpha=8)
    frozen_before = model.transformer.h[0].attn.c_attn.base.weight.detach().clone()
    examples = [{"name": f"P{i}", "category": "x", "specs": {"k": str(i)}, "keywords": [], "tone": "playful",
                 "description": "A tiny description used for testing."} for i in range(6)]
    result = llm.train_lora(model, FakeTokenizer(), examples[:5], examples[5:], steps=4, batch_size=2,
                            lr=1e-2, max_length=120, eval_every=2, log=lambda *_: None)
    assert len(result["history"]) >= 2
    assert all(math.isfinite(h["train"]) for h in result["history"])
    assert math.isfinite(result["val_loss"])
    assert torch.equal(model.transformer.h[0].attn.c_attn.base.weight, frozen_before)
    assert any(m.B.abs().sum() > 0 for m in model.modules() if isinstance(m, llm.LoRALinear))


def test_generation_is_reproducible_and_temperature_zero_is_greedy():
    _, llm = torch_llm()
    model, tokenizer = tiny_gpt2(), FakeTokenizer()
    prompt = "Product: A\nDescription:"
    first = llm.generate_description(model, tokenizer, prompt, max_new_tokens=6, temperature=0.8, seed=3)
    assert first == llm.generate_description(model, tokenizer, prompt, max_new_tokens=6, temperature=0.8, seed=3)
    greedy = llm.generate_description(model, tokenizer, prompt, max_new_tokens=6, temperature=0, seed=1)
    assert greedy == llm.generate_description(model, tokenizer, prompt, max_new_tokens=6, temperature=0, seed=2)
    with pytest.raises(ValueError):
        llm.generate_description(model, tokenizer, prompt, temperature=-1)
    with pytest.raises(ValueError):
        llm.generate_description(model, tokenizer, prompt, max_new_tokens=500)


def test_learning_rate_warms_up_then_decays():
    _, llm = torch_llm()
    assert llm.lr_at(1, 100, 1e-3) < llm.lr_at(10, 100, 1e-3)
    assert llm.lr_at(10, 100, 1e-3) == pytest.approx(1e-3)
    assert llm.lr_at(100, 100, 1e-3) == pytest.approx(1e-4)

import json
from pathlib import Path

import pytest

from app import cli, core, storage

APP_SCRIPT = Path(__file__).resolve().parent.parent / "app" / "main.py"


def run(capsys, *argv):
    code = cli.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ---- the bundled dataset ----------------------------------------------------

def test_the_bundled_dataset_is_valid_and_consistent():
    examples, problems = storage.read_dataset("data/products.jsonl")
    assert problems == []
    assert len(examples) == 240
    assert {e["tone"] for e in examples} == set(core.TONES)
    for example in examples:
        lowered = example["description"].casefold()
        assert all(keyword.casefold() in lowered for keyword in example["keywords"])
        assert all(value.casefold() in lowered for value in example["specs"].values())


def test_the_bundled_dataset_matches_the_generator():
    examples, _ = storage.read_dataset("data/products.jsonl")
    assert examples == core.synthesize_dataset(240, seed=7)


# ---- storage ----------------------------------------------------------------

def test_jsonl_round_trip(tmp_path):
    data = core.synthesize_dataset(5, seed=3)
    path = tmp_path / "out" / "data.jsonl"
    storage.write_jsonl(str(path), data)
    examples, problems = storage.read_dataset(str(path))
    assert examples == data and problems == []


def test_csv_dataset_with_a_byte_order_mark(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("\ufeffname,specs,description\n"
                    'Lamp,"Power: 8 W","A reliable lamp that suits every desk and every room."\n', encoding="utf-8")
    examples, problems = storage.read_dataset(str(path))
    assert len(examples) == 1 and examples[0]["name"] == "Lamp" and problems == []


def test_missing_dataset_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        storage.read_dataset(str(tmp_path / "nope.jsonl"))


def test_adapter_listing_and_config(tmp_path):
    (tmp_path / "good").mkdir()
    (tmp_path / "good" / storage.ADAPTER_CONFIG).write_text(json.dumps({"base_model": "gpt2", "rank": 8}))
    (tmp_path / "empty").mkdir()
    assert storage.list_adapters(str(tmp_path)) == ["good"]
    assert storage.list_adapters(str(tmp_path / "missing")) == []
    assert storage.read_adapter_config(str(tmp_path / "good"))["base_model"] == "gpt2"
    with pytest.raises(FileNotFoundError):
        storage.read_adapter_config(str(tmp_path / "empty"))
    assert storage.adapter_stamp("") == 0.0


def test_adapter_names_are_made_safe():
    assert storage.safe_name(" My adapter! ") == "My-adapter"
    assert storage.safe_name("../etc") == "etc"
    with pytest.raises(ValueError):
        storage.safe_name("***")


# ---- command line (no model needed) ------------------------------------------

def test_data_command_writes_a_dataset(tmp_path, capsys):
    out = tmp_path / "products.jsonl"
    code, stdout, _ = run(capsys, "data", "--n", "30", "--out", str(out))
    assert code == 0 and "wrote 30 examples" in stdout
    examples, problems = storage.read_dataset(str(out))
    assert len(examples) == 30 and problems == []


def test_assess_command_scores_a_description(capsys):
    good = ("Up to 12 hours of battery life and IPX5 protection make these wireless earbuds ideal for the gym, "
            "the commute and everything in between, with USB-C charging when you need a quick top-up.")
    code, out, _ = run(capsys, "assess", "--spec", "Battery life: 12 hours", "--spec", "Water resistance: IPX5",
                       "--spec", "Charging: USB-C", "--keywords", "wireless earbuds", "--text", good)
    assert code == 0 and "specs mentioned : 100%" in out and "flags           : none" in out

    code, out, _ = run(capsys, "assess", "--spec", "Battery life: 12 hours", "--text", "Lasts 40 hours.")
    assert "numbers not in the input: 40" in out


def test_train_reports_a_missing_or_tiny_dataset(tmp_path, capsys):
    code, _, err = run(capsys, "train", "--data", str(tmp_path / "nope.jsonl"))
    assert code == 1 and "dataset not found" in err
    tiny = tmp_path / "tiny.jsonl"
    storage.write_jsonl(str(tiny), core.synthesize_dataset(3))
    code, _, err = run(capsys, "train", "--data", str(tiny))
    assert code == 1 and "at least about 10" in err


def test_generate_needs_a_product(capsys):
    code, _, err = run(capsys, "generate", "--adapter", "")
    assert code == 1 and "--name" in err


def test_generate_merge_and_review_report_a_missing_adapter(tmp_path, capsys):
    for command in (["generate", "--name", "Lamp", "--spec", "Power: 8 W"], ["merge"], ["review"]):
        code, _, err = run(capsys, *command, "--adapters", str(tmp_path), "--adapter", "ghost")
        assert code == 1 and "no adapter" in err


# ---- the Streamlit app --------------------------------------------------------

def test_the_app_starts_and_validates_the_specs():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP_SCRIPT), default_timeout=60).run()
    assert not app.exception
    assert app.title[0].value == "Product Description Enhancer"

    specs = next(t for t in app.text_area if t.label.startswith("Specs"))
    specs.set_value("this line has no value").run()
    button = next(b for b in app.button if b.label == "Generate description")
    button.click().run()
    assert not app.exception
    assert any("needs a name and a value" in error.value for error in app.error)
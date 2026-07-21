"""Pure-Python unit tests for eval_capability helpers (no lm-eval, no GPU)."""

import importlib.util
import sys
from pathlib import Path

# eval_capability.py is a script (not in the package); load it directly.
_spec = importlib.util.spec_from_file_location(
    "eval_capability", Path(__file__).resolve().parent.parent / "scripts" / "eval_capability.py"
)
ec = importlib.util.module_from_spec(_spec)
sys.modules["eval_capability"] = ec
_spec.loader.exec_module(ec)


def test_parse_limit():
    assert ec.parse_limit(None) is None
    assert ec.parse_limit("50") == 50
    assert ec.parse_limit(50) == 50
    assert ec.parse_limit("mmlu=15,gsm8k_cot=200") == {"mmlu": 15, "gsm8k_cot": 200}
    # 'full'/'none'/'all' opt back into the whole test set over a config default
    assert ec.parse_limit("full") is None
    assert ec.parse_limit("NONE") is None
    assert ec.parse_limit("all") is None


def test_limit_for():
    assert ec.limit_for(None, "mmlu") is None
    assert ec.limit_for(50, "mmlu") == 50
    assert ec.limit_for({"mmlu": 15, "gsm8k_cot": 200}, "mmlu") == 15
    assert ec.limit_for({"mmlu": 15}, "gsm8k_cot") is None


def test_capability_dir_default():
    assert ec.capability_dir({"results_dir": "results/qwen3b"}).name == "capability"
    assert "qwen3b" in str(ec.capability_dir({"results_dir": "results/qwen3b"}))
    assert ec.capability_dir({"results_dir": "x", "capability_dir": "custom/cap"}).name == "cap"


def test_model_args_base():
    cfg = {"model_id": "Qwen/Qwen2.5-3B-Instruct", "adapter_dir": "results/qwen3b/m1_resist_adapter"}
    a = ec.model_args(cfg, "m0")
    assert "pretrained=Qwen/Qwen2.5-3B-Instruct" in a and "peft=" not in a


def test_read_metric_missing(tmp_path):
    assert ec.read_metric(tmp_path, "mmlu") == (None, None)


def test_read_metric_parses(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    (d / "results_x.json").write_text('{"results": {"mmlu": {"acc,none": 0.765, "acc_stderr,none": 0.01}}}')
    val, key = ec.read_metric(tmp_path, "mmlu")
    assert key == "acc,none" and abs(val - 0.765) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    import tempfile
    for fn in fns:
        fn(Path(tempfile.mkdtemp())) if "tmp_path" in fn.__code__.co_varnames else fn()
        print("PASS", fn.__name__)
    print(f"\n{len(fns)} passed")

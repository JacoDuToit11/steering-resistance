"""Hugging Face Hub backup: adapter + provenance + auto-generated model card (PRINCIPLES §9).

push_run() uploads the trained adapter to the repo root (so
`PeftModel.from_pretrained(repo_id)` just works) and mirrors the entire run
directory — run_meta.json, config, code.patch, eval jsonl, summaries — under
run/. The model card is generated from the same manifest, so base model,
dataset, hyperparameters, git commit, W&B link and eval results are attached
to the weights themselves, not to someone's memory of the run.

Enabled by cfg['hub_repo_id'] (auto-push in-pipeline) or scripts/push_to_hub.py
(manual re-push). Private repos by default. Auth: `hf auth login` or HF_TOKEN.
"""

import json
from pathlib import Path

TRAIN_PARAM_KEYS = [
    "layer", "train_alphas", "eval_alphas", "efficacy_alpha", "efficacy_min_rate",
    "steered_frac", "relevant_frac", "repeats_per_question", "lora_r", "lora_alpha",
    "lora_dropout", "lora_targets", "lr", "epochs", "effective_batch_size",
    "train_batch_size", "max_seq_len", "seed",
]  # fmt: skip


def build_model_card(cfg: dict, meta: dict, summary_md: str | None) -> str:
    """Model card (Hub-schema frontmatter + provenance/params/results) from the run manifest."""
    git, env = meta.get("git", {}), meta.get("env", {})
    commit, remote = git.get("commit"), git.get("remote") or ""
    commit_link = f"[`{commit[:12]}`]({remote.removesuffix('.git')}/tree/{commit})" if commit and remote else (commit or "unknown")

    front = [
        "---",
        f"base_model: {cfg['model_id']}",
        "library_name: peft",
        "tags:",
        "- lora",
        "- activation-steering",
        "- steering-resistance",
    ]
    if cfg.get("qa_source") == "popqa":
        front += ["datasets:", "- akariasai/PopQA"]
    front.append("---")

    prov = [
        ("run", meta.get("run")),
        ("trained", meta.get("started_at")),
        ("code", commit_link + (" **+ uncommitted changes** (see `run/code.patch`)" if git.get("dirty") else "")),
        ("config", f"`{meta.get('config_path')}` (snapshot: `run/config.yaml`)"),
        ("wandb", meta.get("wandb_url") or "—"),
        ("hardware", env.get("accelerator")),
        ("stack", ", ".join(f"{k} {v}" for k, v in (env.get("packages") or {}).items() if v)),
    ]
    for key, fp in (meta.get("data") or {}).items():
        prov.append((f"data: {key}", f"`{fp['sha256'][:16]}…` ({fp['path']})" if "sha256" in fp else json.dumps(fp)))

    lines = front + [
        "",
        f"# {meta.get('run', 'run')} — steering-resistance LoRA adapter",
        "",
        f"LoRA adapter for **{cfg['model_id']}** trained to resist adversarial activation",
        f"steering: fine-tuned with CAA vectors injected live at decoder layer {cfg['layer']},",
        "rewarded for reproducing its own clean answers. Full method:",
        f"{remote.removesuffix('.git') or 'the steering-resistance repo'}.",
        "",
        "## Provenance",
        "",
        "| | |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in prov],
        "",
        "## Training parameters",
        "",
        "| param | value |",
        "|---|---|",
        *[f"| {k} | `{json.dumps(cfg[k])}` |" for k in TRAIN_PARAM_KEYS if k in cfg],
        "",
        "## Eval results",
        "",
        summary_md.strip() if summary_md else "_No eval summary at push time — see `run/` for raw jsonl, or re-push after eval_m1._",
        "",
        "## Reproduce",
        "",
        "```bash",
        f"git clone {remote or '<repo>'} && cd {Path(remote).stem or '<repo>'}",
        f"git checkout {commit or '<commit>'}",
        f"python scripts/run.py {meta.get('config_path') or 'configs/<config>.yaml'}",
        "```",
        "",
        "`run/` mirrors the full experiment directory: `run_meta.json` (manifest with",
        "artifact hashes), append-only eval jsonl, summaries, and the exact config.",
        "",
    ]
    return "\n".join(lines)


def push_run(cfg: dict, meta: dict | None = None, repo_id: str | None = None) -> str:
    """Create/update the Hub repo: adapter at root, run dir under run/, model card. Returns repo URL."""
    from huggingface_hub import HfApi

    repo_id = repo_id or cfg.get("hub_repo_id")
    if not repo_id:
        raise ValueError("no hub_repo_id in config and no --repo-id given")
    results_dir = Path(cfg["results_dir"])
    adapter_dir = Path(cfg["adapter_dir"])
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"no trained adapter at {adapter_dir}")
    if meta is None:
        meta_path = results_dir / "run_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    summary_md_path = results_dir / "summary.md"
    card = build_model_card(cfg, meta, summary_md_path.read_text() if summary_md_path.exists() else None)

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=bool(cfg.get("hub_private", True)), exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(adapter_dir),
        commit_message=f"adapter ({(meta.get('git', {}).get('commit') or 'no-commit')[:12]})",
        ignore_patterns=["README.md"],  # PEFT's stub card; ours goes up below
    )
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(results_dir),
        path_in_repo="run",
        commit_message="run directory (provenance + evals)",
        ignore_patterns=["wandb/*", f"{adapter_dir.name}/*"],
    )
    api.upload_file(
        repo_id=repo_id,
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        commit_message="model card",
    )
    url = f"https://huggingface.co/{repo_id}"
    print(f"pushed adapter + run dir -> {url}")
    return url

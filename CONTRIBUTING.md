# Development workflow

`main` is protected and always green: it requires a pull request and a passing
CI run (`test`: ruff lint + pytest) before merge. Never commit to `main`
directly — every change goes through a branch and a PR.

## The loop

```bash
git switch main && git pull                 # start from latest main
git switch -c <type>/<short-name>           # branch (see naming below)

# ... make changes; commit in small, reviewable steps ...
git add -p && git commit -m "clear message"

# before pushing: run the same gates CI runs, so red CI is rare
ruff check src scripts tests
python -m pytest tests/ -q

git push -u origin HEAD                      # push the branch
gh pr create --fill                          # open the PR (CI starts automatically)
gh pr checks --watch                         # wait for CI green
gh pr merge --squash --delete-branch         # merge once green, tidy up
git switch main && git pull                  # sync local main
```

## Branch naming

| prefix | for | example |
|---|---|---|
| `exp/` | an experiment / research iteration | `exp/geometric-sweep` |
| `feat/` | new capability in the pipeline | `feat/llama-support` |
| `fix/` | bug fix | `fix/eval-resume-key` |
| `chore/` | tooling, docs, config | `chore/dev-workflow` |

## Conventions

- **Small PRs.** One idea per branch; easier to review and revert.
- **Squash-merge** so `main` history stays one-commit-per-change and linear.
- **CI must pass** — it runs `ruff` and `pytest` (no GPU, no network). If you
  touch scoring / pairing / eval logic, add or update a test in `tests/`.
- **Provenance stays intact.** Real experiment runs should be from a clean,
  committed tree (a dirty tree is recorded in `run_meta.json` + `code.patch`
  with a warning — fine for scratch, not for results you'll cite).
- Experiment results (`results/<run>/` jsonl + summaries) may be committed on
  their branch; adapters/checkpoints never are (they go to the HF Hub).

## Bypass (owner, emergencies only)

Admin enforcement is off, so the repo owner *can* push to `main` in a genuine
pinch — but don't make it a habit; the point of the gate is that `main` is
always installable and green.

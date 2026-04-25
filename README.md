---
title: Word Ladder BERT API
emoji: 🪜
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Word Ladder BERT API

FastAPI service for the Word Ladder thesis project: given **current** word, **target** word, and **mode** (`en_4`, `en_5`, `hr_4`, `hr_5`), returns the **best one-letter neighbor** by lowest **predicted graph distance** (BERT regression head), matching `scripts/play_wordladder.py` in the main repo.

## Layout

| Path | Purpose |
|------|--------|
| `app.py` | `POST /predict`, `GET /health` |
| `data/words_*.txt` | Largest-island word lists (same graph as the main game) |
| `models/en_4` … `models/hr_5` | Fine-tuned checkpoints (not in git — add locally or download) |

## Add model weights

Each of `models/en_4`, `models/en_5`, `models/hr_4`, `models/hr_5` must contain a HuggingFace-style folder (`config.json` + `tokenizer` files + `model.safetensors` or `pytorch_model.bin`). See `models/README.md` for the mapping to the main repo’s `bert_wordladder_*` folders.

Optional: set `PRELOAD_ALL_MODELS=1` to load every checkpoint at startup (needs enough RAM; otherwise first request per mode loads lazily).

## Local run (CPU)

```bash
cd word-ladder-api
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# Place checkpoints under models/… then:
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Request example

`POST /predict` with JSON body:

```json
{
  "current": "cold",
  "target": "warm",
  "mode": "en_4",
  "full_ranking": false
}
```

Response includes `best_neighbor` and `predicted_distance` (lower = better).

## GitHub (code only — no BERT files)

Do **not** commit weights; `.gitignore` already ignores `*.bin` / `*.safetensors` / `*.pt`.

1. Create a new empty repo on GitHub (e.g. `word-ladder-api`).
2. From this folder:

```bash
cd word-ladder-api
git init
git add .
git status   # confirm no huge model files
git commit -m "Add Word Ladder BERT API"
git branch -M main
git remote add origin https://github.com/<YOU>/<REPO>.git
git push -u origin main
```

## Put the BERT checkpoints on Hugging Face (Model Hub)

**You host weights as separate “Model” repos, not in GitHub.** four options, pick one.

### A) Four small Model repos (clearest)

1. On [huggingface.co](https://huggingface.co) → **New model** (e.g. `yourname/bert-wl-en5`, repeat for en4, hr4, hr5).
2. On your machine, in each trained export folder (with `config.json` + weights + tokenizer files), run after `huggingface-cli login`:

```bash
huggingface-cli upload yourname/bert-wl-en5 . --repo-type model
# run from the folder that contains config.json, model.safetensors, etc.
```

Or use the model page → **Files** → **Upload files** (OK for a few hundred MB if the UI allows).

3. In your **Space** → **Settings** → **Repository secrets**: add a **Read** token as `HUGGING_FACE_HUB_TOKEN` (or `HF_TOKEN`) if any model repo is **private**; public models work without a token.
4. In the same place, add **Variables** (or secrets) mapping Hub repos to modes:

- `HF_HUB_EN_4` = `yourname/bert-wl-en4`
- `HF_HUB_EN_5` = `yourname/bert-wl-en5`
- `HF_HUB_HR_4` = `yourname/bert-wl-hr4`
- `HF_HUB_HR_5` = `yourname/bert-wl-hr5`

The Docker **entrypoint** runs `download_models_from_hub.py` on startup, downloads into `models/en_4` … `models/hr_5`, then starts `uvicorn`. If a mode already has files in `models/`, that mode is skipped.

### B) Keep weights only on the Space (manual)

In the Space **Files** tab you can add files, but very large bins are painful; the Hub + env vars in **A** is more reliable.

### C) Baked into the image (not recommended)

You could `COPY` weights in `Dockerfile`, but the image gets huge; prefer **A**.

## Connect the Space to GitHub

1. **New Space** on Hugging Face → **Docker** SDK → “Import from GitHub” and select this repo, or create an empty Space and set **Repository** to your GitHub URL.
2. **app_port** for Spaces is `7860` (already in this README’s frontmatter and `Dockerfile`).

## Move this folder to its own GitHub repository

This folder is self-contained. You can also copy it out of the monorepo with `git subtree split` and push the same way as in “GitHub” above.

## Environment

- `MODELS_DIR` — override path to the `models` root (default: `./models`).

---

This folder lives inside the [word-ladder](https://github.com/) monorepo for development; you can publish only this subtree to HF.

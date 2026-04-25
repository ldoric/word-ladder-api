"""
Word Ladder BERT hint API: score one-letter neighbors by predicted graph distance
to the target (same idea as ../scripts/play_wordladder.py).
"""
from __future__ import annotations

import os
import string
import threading
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Paths relative to this file (Space / Docker working dir = repo root)
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = Path(os.environ.get("MODELS_DIR", str(ROOT / "models")))

MAX_LENGTH = 32

# Word lists: copies of main repo data/islands/*_largest_island.txt
MODE_CONFIG: dict[str, dict] = {
    "en_4": {"word_file": "words_en_4.txt", "length": 4, "locale": "en"},
    "en_5": {"word_file": "words_en_5.txt", "length": 5, "locale": "en"},
    "hr_4": {"word_file": "words_hr_4.txt", "length": 4, "locale": "hr"},
    "hr_5": {"word_file": "words_hr_5.txt", "length": 5, "locale": "hr"},
}

app = FastAPI(title="Word Ladder BERT API", version="1.0.0")

# Lazy-loaded per mode: { "en_4": { "model", "tokenizer", "vocab", "replacement_chars" } }
_state: dict[str, dict] = {}
_lock = threading.Lock()
_device: torch.device | None = None


def get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def load_vocab(path: Path, length: int) -> set[str]:
    return {
        w.strip().lower()
        for w in path.read_text(encoding="utf-8").splitlines()
        if w.strip() and len(w.strip()) == length
    }


def charset_for(words: set[str]) -> list[str]:
    """Distinct characters in the lexicon (Croatian diacritics)."""
    chars: set[str] = set()
    for w in words:
        chars.update(w)
    return sorted(chars)


def one_letter_neighbors(
    w: str,
    vocab: set[str],
    replacement_chars: list[str],
) -> set[str]:
    out: set[str] = set()
    n = len(w)
    for i in range(n):
        for c in replacement_chars:
            if c == w[i]:
                continue
            nw = w[:i] + c + w[i + 1 :]
            if nw in vocab:
                out.add(nw)
    return out


def score_candidates(
    model,
    tokenizer,
    target: str,
    candidates: list[str],
    device: torch.device,
) -> list[tuple[str, float]]:
    """Predict distance from each candidate to target; lower = better."""
    if not candidates:
        return []

    enc = tokenizer(
        list(candidates),
        [target] * len(candidates),
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    model.eval()
    with torch.no_grad():
        out = model(**enc)
        preds = out.logits.squeeze(-1).cpu().tolist()

    if isinstance(preds, float):
        preds = [preds]
    return sorted(zip(candidates, preds), key=lambda x: x[1])


def _model_dir_ready(model_dir: Path) -> bool:
    if (model_dir / "config.json").is_file():
        if any(model_dir.glob("*.safetensors")) or (model_dir / "pytorch_model.bin").is_file():
            return True
    return False


def _load_mode_state(mode: str) -> None:
    cfg = MODE_CONFIG[mode]
    wpath = DATA_DIR / cfg["word_file"]
    if not wpath.is_file():
        raise HTTPException(
            500,
            detail=f"Word list missing: {wpath}",
        )
    length = cfg["length"]
    vocab = load_vocab(wpath, length)
    if cfg["locale"] == "en":
        replacement = list(string.ascii_lowercase)
    else:
        replacement = charset_for(vocab)

    model_dir = MODELS_DIR / mode
    if not _model_dir_ready(model_dir):
        raise HTTPException(
            500,
            detail=f"No model in {model_dir}. Add config.json + model weights "
            "(model.safetensors or pytorch_model.bin).",
        )

    device = get_device()
    mdl = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    mdl = mdl.to(device)
    mdl.eval()

    _state[mode] = {
        "model": mdl,
        "tokenizer": tok,
        "vocab": vocab,
        "replacement_chars": replacement,
    }


def ensure_mode(mode: str) -> dict:
    if mode not in MODE_CONFIG:
        raise HTTPException(
            400,
            detail=f"Invalid mode: {mode!r}. Expected one of {list(MODE_CONFIG)}.",
        )
    with _lock:
        if mode not in _state:
            _load_mode_state(mode)
        return _state[mode]


@app.on_event("startup")
def startup_preload() -> None:
    if os.environ.get("PRELOAD_ALL_MODELS", "0") != "1":
        return
    with _lock:
        for mode in MODE_CONFIG:
            if mode not in _state:
                try:
                    _load_mode_state(mode)
                except HTTPException:
                    # Missing weights: skip so /health and docs still work
                    pass


class PredictIn(BaseModel):
    current: str = Field(..., min_length=1, description="Current word in the ladder")
    target: str = Field(..., min_length=1, description="Target / goal word")
    mode: Literal["en_4", "en_5", "hr_4", "hr_5"]
    full_ranking: bool = Field(
        False, description="If true, return all neighbors sorted by predicted distance"
    )


class NeighborScore(BaseModel):
    word: str
    predicted_distance: float


class PredictOut(BaseModel):
    best_neighbor: str | None
    predicted_distance: float | None
    message: str | None = None
    all_ranked: list[NeighborScore] | None = None


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "device": str(get_device()),
        "loaded_modes": list(_state.keys()),
    }


@app.post("/predict", response_model=PredictOut)
def predict(body: PredictIn) -> PredictOut:
    st = ensure_mode(body.mode)
    length = MODE_CONFIG[body.mode]["length"]

    current = body.current.strip().lower()
    target = body.target.strip().lower()

    if len(current) != length or len(target) != length:
        raise HTTPException(
            400,
            detail=f"Words must be exactly {length} characters for mode {body.mode!r}.",
        )

    vocab: set[str] = st["vocab"]
    if current not in vocab:
        raise HTTPException(400, detail="current is not in the dictionary for this mode")
    if target not in vocab:
        raise HTTPException(400, detail="target is not in the dictionary for this mode")

    neighbors = sorted(
        one_letter_neighbors(current, vocab, st["replacement_chars"])
    )
    if not neighbors:
        return PredictOut(
            best_neighbor=None,
            predicted_distance=None,
            message="no legal one-letter moves from current word in this dictionary",
        )

    device = get_device()
    ranked = score_candidates(
        st["model"], st["tokenizer"], target, neighbors, device
    )
    best_word, best_dist = ranked[0]
    all_ranked = None
    if body.full_ranking:
        all_ranked = [
            NeighborScore(word=w, predicted_distance=float(d)) for w, d in ranked
        ]

    return PredictOut(
        best_neighbor=best_word,
        predicted_distance=float(best_dist),
        all_ranked=all_ranked,
    )


@app.get("/")
def root() -> dict:
    return {
        "service": "word-ladder-bert-api",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }

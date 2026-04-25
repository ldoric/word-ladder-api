"""
Optional: download checkpoints from Hugging Face Model repos at container start.

Set one or more env vars to repo IDs, e.g.:
  HF_HUB_EN_5=yourname/bert-wordladder-en5
Public repos work without a token. Private repos need HF_TOKEN (or HUGGING_FACE_HUB_TOKEN).

If a models/<mode> folder already has config + weights, that mode is skipped.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"

PAIRS: list[tuple[str, str | None]] = [
    ("en_4", os.environ.get("HF_HUB_EN_4") or os.environ.get("HUB_EN_4")),
    ("en_5", os.environ.get("HF_HUB_EN_5") or os.environ.get("HUB_EN_5")),
    ("hr_4", os.environ.get("HF_HUB_HR_4") or os.environ.get("HUB_HR_4")),
    ("hr_5", os.environ.get("HF_HUB_HR_5") or os.environ.get("HUB_HR_5")),
]


def _has_weights(d: Path) -> bool:
    if not (d / "config.json").is_file():
        return False
    return any(d.glob("*.safetensors")) or (d / "pytorch_model.bin").is_file()


def main() -> None:
    to_fetch = [(s, r) for s, r in PAIRS if r and str(r).strip()]
    if not to_fetch:
        print("[download] no HF_HUB_* env set — using only files under models/ (if any)")
        return

    from huggingface_hub import snapshot_download  # type: ignore[import-untyped]

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    for sub, repo_id in to_fetch:
        dest = MODELS / sub
        dest.mkdir(parents=True, exist_ok=True)
        if _has_weights(dest):
            print(f"[download] skip {sub} (already has weights in {dest})")
            continue
        rid = str(repo_id).strip()
        print(f"[download] {rid} -> {dest}")
        snapshot_download(
            repo_id=rid,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
            token=token,
        )
        print(f"[download] done {sub}")


if __name__ == "__main__":
    main()

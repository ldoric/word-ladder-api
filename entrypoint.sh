#!/bin/sh
set -e
cd /app
# Pull checkpoints from the Hub if env vars are set (see README)
python download_models_from_hub.py
exec uvicorn app:app --host 0.0.0.0 --port 7860

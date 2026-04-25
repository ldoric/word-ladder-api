# BERT distance-regression checkpoints

Copy each fine-tuned folder from the main `word-ladder` repo (after training on Colab) so each directory contains at least `config.json`, a tokenizer, and weights (`model.safetensors` and/or `pytorch_model.bin`).

| `models/` subfolder | Main repo source (typical) |
|----------------------|----------------------------|
| `en_4` | `models/bert_wordladder_4letter/` |
| `en_5` | `models/bert_wordladder_5letter/` |
| `hr_4` | `models/bert_wordladder_croatian4/` |
| `hr_5` | `models/bert_wordladder_croatian5/` |

You can also push to a private Hugging Face **Model** repo and download at build time (see top-level `README.md`).

# TinyTale

A small GPT-style transformer trained from scratch on public-domain fairy tales (Grimm, Andersen, Aesop, English Fairy Tales). Character-level tokenizer, roughly 57M parameters, AdamW training, temperature + top-k generation.

## what's here

- `config.py` — hyperparameters: VOCAB_SIZE=93, BLOCK_SIZE=256, N_EMBD=256, N_HEAD=8, N_LAYER=72, DROPOUT=0.1, LEARNING_RATE=3e-4, MAX_STEPS=50000.
- `tokenizer/` — character-level tokenizer; `train_tokenizer.py` builds `vocab.json` from the training text.
- `model/transformer.py` — `TaleWizard` (aliased `GPT`): token + positional embedding, 72 `MagicBlock` layers (causal attention + 4x-fan GELU MLP), final LayerNorm, linear output head.
- `prepare_data.py` — reads the four raw books, strips Gutenberg markers, writes `data/processed/combined.txt`.
- `split_data.py` — 90/10 split of `combined.txt` into `train.txt` and `val.txt`.
- `train.py` — AdamW training, evaluates every `EVAL_INTERVAL` steps, saves checkpoints, resumes from `checkpoints/model.pt`.
- `generate.py` — samples with temperature and top-k.

## how it goes

1. `prepare_data.py` -> `data/processed/combined.txt`
2. `split_data.py` -> `train.txt` + `val.txt`
3. `tokenizer/train_tokenizer.py` -> `tokenizer/vocab.json`
4. `train.py` -> `checkpoints/model.pt` (50K steps)
5. `generate.py "prompt" 500 0.8`



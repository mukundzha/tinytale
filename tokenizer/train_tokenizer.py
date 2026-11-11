from pathlib import Path

from tokenizer import Tokenizer

TRAIN_FILE = Path("data/processed/train.txt")
VAL_FILE = Path("data/processed/val.txt")
VOCAB_FILE = Path("tokenizer/vocab.json")


if not TRAIN_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {TRAIN_FILE}")

train_text = TRAIN_FILE.read_text(encoding="utf-8")
val_text = VAL_FILE.read_text(encoding="utf-8")

tok = Tokenizer()
tok.build(train_text)
tok.save(VOCAB_FILE)

print("Tokenizer built!")
print("-" * 35)
print(f"Vocab size: {len(tok)}")
print(f"Saved to:   {VOCAB_FILE}")

train_ids = tok.encode(train_text)
val_ids = tok.encode(val_text)

print(f"Train chars: {len(train_text):,} -> ids: {len(train_ids):,}")
print(f"Val chars:   {len(val_text):,} -> ids: {len(val_ids):,}")
print(f"Train round trip: {tok.decode(train_ids) == train_text}")
print(f"Val round trip:   {tok.decode(val_ids) == val_text}")

loaded = Tokenizer.load(VOCAB_FILE)
print(f"Load check: {loaded.encode('Once upon a time') == tok.encode('Once upon a time')}")

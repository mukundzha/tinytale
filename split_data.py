from pathlib import Path

SOURCE_FILE = Path("data/processed/combined.txt")
OUTPUT_DIR = Path("data/processed")

needed_ratio = 0.90

if not SOURCE_FILE.exists():
    raise FileNotFoundError(
            f"Dataset not found: {SOURCE_FILE}"
    )


text = SOURCE_FILE.read_text(encoding="utf-8")

if not text.strip():
    raise ValueError("Dataset is Empty")


split_index = int(len(text) * needed_ratio)

train_text = text[:split_index]
val_text = text[split_index:]

train_path = OUTPUT_DIR / "train.txt"
val_path = OUTPUT_DIR / "val.txt"

train_path.write_text(train_text, encoding="utf-8")
val_path.write_text(val_text, encoding="utf-8")

print("Dataset splitting completed!")
print("-" * 35)

print(f"Total characters:      {len(text):,}")
print(f"Training characters:   {len(train_text):,}")
print(f"Validation characters: {len(val_text):,}")

print(f"Training percentage:   {len(train_text) / len(text) * 100:.2f}%")
print(f"Validation percentage: {len(val_text) / len(text) * 100:.2f}%")

print("\nSaved files:")
print(f"  {train_path}")
print(f"  {val_path}")

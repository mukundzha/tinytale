from pathlib import Path
import re

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

BOOKS = {
    "grimm.txt": "Grimm's Fairy Tales",
    "andersen.txt": "Andersen's Fairy Tales",
    "aesop.txt": "Aesop's Fables",
    "english_fairy_tales.txt": "English Fairy Tales",
}


def clean_gutenberg_text(text):
    # Remove UTF-8 BOM and normalize line endings
    text = text.lstrip("\ufeff")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Find Gutenberg start and end markers
    start = re.search(
        r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        text,
        re.IGNORECASE
    )

    end = re.search(
        r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        text,
        re.IGNORECASE
    )

    if not start or not end or end.start() <= start.end():
        raise ValueError("Could not find valid Gutenberg boundaries.")

    # Keep only the actual book content
    text = text[start.end():end.start()]

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    combined_text = []
    total_words = 0

    for filename, title in BOOKS.items():
        file_path = RAW_DIR / filename

        if not file_path.exists():
            print(f"[SKIP] {filename} not found")
            continue

        raw_text = file_path.read_text(encoding="utf-8")
        cleaned_text = clean_gutenberg_text(raw_text)

        word_count = len(cleaned_text.split())
        total_words += word_count

        combined_text.append(
            f"\n\n{title}\n\n{cleaned_text}"
        )

        print(f"[OK] {title}: {word_count:,} words")

    if not combined_text:
        raise RuntimeError("No books were successfully processed.")

    output = "\n\n".join(combined_text).strip()

    output_path = PROCESSED_DIR / "combined.txt"
    output_path.write_text(output, encoding="utf-8")

    print("\n----------------------------")
    print("DATASET PREPARATION COMPLETE")
    print("----------------------------")
    print(f"Total words: {total_words:,}")
    print(f"Output size: {len(output.encode('utf-8')) / 1024:.2f} KB")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()


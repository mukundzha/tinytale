import json
from pathlib import Path

UNK = "<unk>"


class Tokenizer:
    def __init__(self):
        self.chars = []
        self.char_to_id = {}

    def build(self, text):
        self.chars = [UNK] + sorted(set(text))
        self.char_to_id = {}
        for i, ch in enumerate(self.chars):
            self.char_to_id[ch] = i

    def encode(self, text):
        ids = []
        for ch in text:
            if ch in self.char_to_id:
                ids.append(self.char_to_id[ch])
            else:
                ids.append(0)
        return ids

    def decode(self, ids):
        text = ""
        for i in ids:
            text += self.chars[i]
        return text

    def save(self, path):
        data = json.dumps(self.chars, ensure_ascii=False)
        Path(path).write_text(data, encoding="utf-8")

    @classmethod
    def load(cls, path):
        tok = cls()
        tok.chars = json.loads(Path(path).read_text(encoding="utf-8"))
        tok.char_to_id = {}
        for i, ch in enumerate(tok.chars):
            tok.char_to_id[ch] = i
        return tok

    def __len__(self):
        return len(self.chars)

import sys
from pathlib import Path

import torch

import config
from model.transformer import GPT
from tokenizer import Tokenizer

CHECKPOINT = Path("checkpoints/model.pt")

if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Model not found: {CHECKPOINT}")

prompt = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"
max_new_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 500
temperature = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8

device = "cuda" if torch.cuda.is_available() else "cpu"

tok = Tokenizer.load("tokenizer/vocab.json")
state = torch.load(CHECKPOINT, map_location=device)

if "model" in state:
    state = state["model"]

model = GPT(
    config.VOCAB_SIZE,
    state["spot_pot.weight"].shape[0],
    config.N_EMBD,
    config.N_HEAD,
    config.N_LAYER,
    config.DROPOUT,
).to(device)
model.load_state_dict(state)
model.eval()

ids = tok.encode(prompt)
x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
out = model.generate(x, max_new_tokens, temperature=temperature)
print(tok.decode(out[0].tolist()))

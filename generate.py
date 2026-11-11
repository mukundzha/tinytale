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

# Extract model state dict (handle both formats)
if isinstance(state, dict) and "model" in state:
    state = state["model"]
elif isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]

# Get model dimensions from checkpoint
embedding_dim = state["word_pot.weight"].shape[1]
n_layer = len([key for key in state.keys() if key.startswith("blocks.")])

model = GPT(
    config.VOCAB_SIZE,
    config.BLOCK_SIZE,
    embedding_dim,
    config.N_HEAD,
    n_layer,
    config.DROPOUT,
).to(device)
model.load_state_dict(state)
model.eval()

ids = tok.encode(prompt)
x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
out = model.generate(x, max_new_tokens, temperature=temperature)
print(tok.decode(out[0].tolist()))

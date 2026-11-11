from pathlib import Path

import torch
import torch.nn.functional as F

import config
from model.transformer import GPT
from tokenizer import Tokenizer

torch.manual_seed(config.SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

train_text = Path("data/processed/train.txt").read_text(encoding="utf-8")
val_text = Path("data/processed/val.txt").read_text(encoding="utf-8")

tok = Tokenizer.load("tokenizer/vocab.json")

train_data = torch.tensor(tok.encode(train_text), dtype=torch.long)
val_data = torch.tensor(tok.encode(val_text), dtype=torch.long)

model = GPT(
    config.VOCAB_SIZE,
    config.BLOCK_SIZE,
    config.N_EMBD,
    config.N_HEAD,
    config.N_LAYER,
    config.DROPOUT,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

CHECKPOINT = Path(config.CHECKPOINT_DIR) / "model.pt"

start_step = 0
if CHECKPOINT.exists():
    saved = torch.load(CHECKPOINT, map_location=device)
    if "model" in saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_step = saved["step"]
    else:
        model.load_state_dict(saved)
    print(f"Resumed from step {start_step}")


def save_checkpoint(step):
    Path(config.CHECKPOINT_DIR).mkdir(exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        CHECKPOINT,
    )
    print(f"  saved checkpoint at step {step}")


def get_batch(data):
    starts = torch.randint(len(data) - config.BLOCK_SIZE, (config.BATCH_SIZE,))
    x = torch.stack([data[i:i + config.BLOCK_SIZE] for i in starts])
    y = torch.stack([data[i + 1:i + 1 + config.BLOCK_SIZE] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {}
    for name, data in (("train", train_data), ("val", val_data)):
        total = 0.0
        for _ in range(config.EVAL_ITERS):
            x, y = get_batch(data)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total += loss.item()
        losses[name] = total / config.EVAL_ITERS
    model.train()
    return losses


for step in range(start_step, config.MAX_STEPS):
    if step % config.EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"step {step}: train {losses['train']:.4f} | val {losses['val']:.4f}")
        save_checkpoint(step)

    x, y = get_batch(train_data)
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

losses = estimate_loss()
print(f"step {config.MAX_STEPS}: train {losses['train']:.4f} | val {losses['val']:.4f}")

save_checkpoint(config.MAX_STEPS)
print("Model saved to checkpoints/model.pt")

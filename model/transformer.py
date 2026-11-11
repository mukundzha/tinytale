import torch
import torch.nn as nn


class MagicBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(
            n_embd, n_head, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.ln1(x)
        t = x.size(1)
        mask = torch.triu(
            torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1
        )
        attn_out, _ = self.attn(
            h, h, h, attn_mask=mask, need_weights=False
        )
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class TaleWizard(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.block_size = block_size
        self.word_pot = nn.Embedding(vocab_size, n_embd)
        self.spot_pot = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[MagicBlock(n_embd, n_head, dropout) for _ in range(n_layer)]
        )
        self.final_soup = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        b, t = idx.shape
        spots = torch.arange(t, device=idx.device)
        x = self.word_pot(idx) + self.spot_pot(spots)
        x = self.drop(x)
        x = self.blocks(x)
        x = self.final_soup(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            short = idx[:, -self.block_size:]
            logits = self(short)[:, -1, :] / temperature
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                cut = torch.topk(logits, k).values[:, -1:]
                logits = logits.masked_fill(logits < cut, -float("inf"))
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


GPT = TaleWizard

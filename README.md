# TinyTale 👾

A tiny language model that reads itself into existence out of public-domain fairy tales. We're training a GPT-style transformer from scratch on fairy tales (Grimm, Andersen, Aesop, English Fairy Tales) — character by character — and watching it learn to write its own stories. It's small enough to download, understand, and retrain on a budget, and honestly the whole point is the journey rather than the final score.

## what's actually in here

Here's the current state of the project, as it stands today:

- **Data**: four public-domain books, ~1.3 million characters total. These are split into a training set (~1.2M characters) and a validation set (~132K characters).
- **Vocabulary**: a character-level tokenizer with 93 entries covering the printable characters we see in the training text, plus an `` token for anything that doesn't fit.
- **Model**: a GPT-style transformer called `TaleWizard` (aliased as `GPT`), built from 72 stacked `MagicBlock` layers. Each block does causal multi-head attention with a GELU MLP, and the whole thing has roughly 57 million parameters: a 256-dimension embedding, 8 attention heads, and a 256-character context window.
- **Training**: AdamW at a 3e-4 learning rate, with periodic checkpoints and the ability to resume from exactly where you left off.
- **Generation**: temperature and top-k sampling, so you can dial the chaos up or keep it conservative.
- **Environment**: everything here runs on a plain CPU locally, but training is meant to be done on a GPU (Colab or otherwise) — that's where the speed lives.

## how it flows

The pipeline is a straight line, and it only needs to run in order the very first time:

1. **prepare_data.py** reads the four raw books, strips out the Gutenberg front and back matter, and stitches them into one big file (`data/processed/combined.txt`).
2. **split_data.py** cuts that file into a 90/10 training/validation split.
3. **tokenizer/train_tokenizer.py** builds the character vocabulary from the training text and saves it to `tokenizer/vocab.json`.
4. **tokenizer/tokenizer.py** is the actual tokenizer — it maps text to integer IDs and back again.
5. **model/transformer.py** holds the model architecture (see below).
6. **train.py** runs the training loop, printing loss and saving checkpoints along the way.
7. **generate.py** lets you feed in a prompt and have it keep writing.

The first three setup steps produce `train.txt`, `val.txt`, and `vocab.json`. After the first pass, you can skip straight to training. That said, the files are all here already, so the whole pipeline is one command away on a machine that can run it.

## the model, roughly drawn

The model has three jobs, stacked on top of each other:

```
input text (character IDs)
  -> word_pot: token embedding (vocab x 256)
  -> spot_pot: positional embedding (position x 256)
  -> 72 MagicBlock layers (attention + MLP)
  -> final LayerNorm
  -> head: linear (256 -> 93) -> softmax over characters
```

Each `MagicBlock` takes the hidden state, normalizes it, runs a causal multi-head attention so it only ever looks at past characters, adds that back, normalizes again, runs a 4x-fan MLP with a GELU in the middle, and adds that back too. Position is carried by a learned vector that knows which slot in the 256-character window each character sits in, so the model can tell "early word" from "late word" without any literal position numbers.

## how to run it

### locally (proof it works, but slow)

```bash
python prepare_data.py
python split_data.py
python tokenizer/train_tokenizer.py
python train.py
python generate.py "Once upon a time" 500 0.8
```

The tokenizer builds and the model instantiates cleanly on CPU, but at 50K steps the local machine will drag. That's exactly why the next setup is Colab.

### on Google Colab (the real deal)

1. Drop the project into a Colab notebook — either upload the zip I packaged for you, or upload the files into a folder and run them from there.
2. Switch the runtime to **GPU** (Runtime → Change runtime type → GPU).
3. Make sure torch is the CUDA build: `pip install torch --index-url https://download.pytorch.org/whl/cu121` (skipping this just defaults to a CPU wheel, which defeats the point).
4. Run the pipeline the same way as local (`prepare_data.py`, `split_data.py`, `train_tokenizer.py`), then run `train.py` in a notebook cell. Checkpoints land in `checkpoints/model.pt`.
5. When training is done, run `generate.py "Once upon a time" 500 0.8` to see what it wrote.

## what training actually looks like

The training loop does a few useful things:

- It evaluates on both train and validation every 1000 steps, prints both losses, and saves a checkpoint.
- Every checkpoint saves the model weights, the optimizer state, and the current step, so you can pick up exactly where you left off.
- At the end it saves one final checkpoint at step 50000.

At the current size (roughly 57M params, 72 layers), expect around 40–60 minutes on a decent GPU. On CPU it's a stretch of hours, so if you want to play around cheaply, drop `N_LAYER` in `config.py` (try 4–8) and trim `MAX_STEPS` down a notch — the pipeline behaves exactly the same, just faster. The full-size 50K-step run is the "real" one.

## notes

Two things worth knowing while you're tinkering. First, the vocabulary is built from the training split alone, so a handful of characters in the validation split (é, ñ, ö, à, &, ô) don't get their own token and render as `` during evaluation — a small, intentional side effect of the setup. Second, `generate.py` doesn't care how the checkpoint was saved; it looks for `state["model"]` or `state["state_dict"]` and then reads the layer count straight off the tensors, so it's tolerant of both the old and new save formats. `requirements.txt` is still empty, so there's nothing pinned locally — anything you add is worth remembering here.


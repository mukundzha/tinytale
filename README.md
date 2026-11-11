# TinyTale

A small language model experiment training a GPT-style transformer from scratch using public-domain fairy tales.

## Features

- Train a GPT-style transformer from scratch using fairy tales (Grimm, Andersen, Aesop, English Fairy Tales)
- Character-level tokenizer with configurable vocabulary
- Generate stories with temperature and top-k sampling
- Checkpoint saving and resuming

## Configuration

Edit `config.py` to change model parameters.

## Google Colab Setup

1. Open Google Colab
2. Run setup cells as shown in the notebook examples
3. Train and generate text

## Local Training

```bash
# Prepare data
python prepare_data.py
python tokenizer/train_tokenizer.py

# Train
python train.py

# Generate
python generate.py "Once upon a time" 500 0.8
```

## Training Time

- **GPU**: ~40-60 minutes for 50K steps
- **CPU**: Several hours (reduce MAX_STEPS for faster training)

## License

Public-domain fairy tale texts.

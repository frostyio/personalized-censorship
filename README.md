# Personalized Censorship with Prototypical Networks

This repository implements a Prototypical Network for personalized censorship, allowing classification based on user-specific definitions of toxicity. With only a small number of examples per user, the model can adapt to varying definitions of offensive or sensitive content.

[BERT](https://research.google/pubs/bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding/)

[Jigsaw Toxicity dataset](https://huggingface.co/datasets/google/jigsaw_toxicity_pred)

Embeddings are generated using a fine-tuned [BERT](https://research.google/pubs/bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding/) model trained on the [Jigsaw Toxicity dataset](https://huggingface.co/datasets/google/jigsaw_toxicity_pred).

## Architecture

### Text Encoder

Embed query with fine-tuned BERT

- Uses BERT (distilbert-base-uncased) fine-tuned for binary toxicity classification
- Outputs 768-dimensional embeddings per input

### Projection Head

A small MLP projects BERT embeddings into a space more suitable for few-shot classification

[768] → [hidden_dim] → ReLU → Dropout → [768] → LayerNorm

### Prototypical Network

- Learns class prototypes from support examples.
- Computes distances between query samples and class prototypes using:
    - Euclidean distance (default)
    - Cosine distance (optional)
- Returns log-probabilities over `n_way` classes.

## Dataset

The Jigsaw Toxic Comment Classification Challenge dataset from Kaggle:

- Categories: toxic, severe_toxic, obscene, threat, insult, identity_hate
- Binary label: toxic (1) vs non-toxic (0)
- Supports filtering by toxicity type or threshold

Data loading, tokenization, and label balancing are handled in `jigsaw.py`

## Quick-start

There are two Jupyter notebooks provided,

- `notebooks/project.ipynb` for fine-tuning BERT and training the projection head
- `notebooks/project.ipynb` for easily testing individual queries

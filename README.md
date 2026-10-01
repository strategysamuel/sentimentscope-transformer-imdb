# SentimentScope — Transformer Sentiment Analysis

A portfolio-ready PyTorch project implementing a custom Transformer model for binary sentiment classification on the IMDB movie-review dataset.

## Results

| Metric | Result |
|---|---:|
| Validation accuracy | **77.24%** |
| Test accuracy | **76.13%** |
| Training epochs | 3 |
| Batch size | 32 |
| Maximum sequence length | 128 |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Tokenizer | `bert-base-uncased` |

The recorded test accuracy of **76.13%** exceeds the project's stated target of greater than 75%.

## Architecture

The notebook implements:

- Token embeddings
- Positional embeddings
- Causal self-attention
- Multi-head attention
- Feed-forward networks with GELU
- Residual connections
- Layer normalization
- Mean pooling
- Linear classification head

The Transformer itself is implemented in PyTorch; `bert-base-uncased` is used as the tokenizer rather than as the pretrained classifier.

## Dataset

The project uses the standard IMDB dataset structure:

```text
aclImdb/
├── train/
│   ├── pos/
│   └── neg/
└── test/
    ├── pos/
    └── neg/
```

The notebook creates:

- 22,500 training examples
- 2,500 validation examples
- 25,000 test examples

The dataset is intentionally not included in this repository because of its size and licensing/distribution considerations.

## Run the notebook

### 1. Clone

```bash
git clone https://github.com/<YOUR-USERNAME>/sentimentscope-transformer-imdb.git
cd sentimentscope-transformer-imdb
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Obtain the IMDB dataset

Download the IMDB dataset and place/extract `aclImdb/` in the project working directory expected by the notebook.

### 4. Open the notebook

```bash
jupyter notebook notebooks/SentimentScope_Transformer_IMDB.ipynb
```

A CUDA-capable GPU is recommended for training.

## Key takeaways

1. Transformer architectures can be adapted for binary text classification.
2. BERT subword tokenization can be paired with a custom Transformer architecture.
3. PyTorch `Dataset` and `DataLoader` provide the batching and preprocessing pipeline.
4. Mean pooling converts token-level Transformer representations into a sequence-level classification representation.
5. AdamW and cross-entropy loss were used for optimization.
6. Validation accuracy improved from 72.44% after epoch 1 to 77.24% after epoch 3.
7. The final recorded test accuracy was 76.13%.

## Project structure

```text
SentimentScope/
├── notebooks/
│   └── SentimentScope_Transformer_IMDB.ipynb
├── results/
│   └── README.md
├── .gitignore
├── README.md
└── requirements.txt
```

## Next experiments

Possible extensions documented by the project include training for more epochs, increasing embedding size, and increasing the number of Transformer blocks.

## Author

**A. Samuel Arun Kumar**

This project was completed as part of the Udacity Future AWS AI Programmer learning track.

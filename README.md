# myTokenizer

My implementation of Byte Pair Encoding (BPE) tokenization algorithm, built from scratch. This tokenizer follows byte-level BPE with regex-based pretokenization to prevent cross-word merges.

## Overview

This project implements a complete BPE tokenizer training and inference pipeline. Unlike wrapper libraries around existing tokenizers, this is a ground-up implementation demonstrating:

- **Byte-level BPE**: Handles any Unicode text by operating at the byte level
- **Regex pretokenization**: GPT-2/GPT-4 style splitting to constrain merges within semantic boundaries
- **Parallelized training**: Multiprocessing for efficient pair counting on large corpora
- **Deterministic encoding**: Merge operations follow training-time priority for consistent tokenization
- **Comprehensive validation**: Automated test suite covering edge cases, Unicode, and compression metrics

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Training Pipeline                     │
├─────────────────────────────────────────────────────────────┤
│  Corpus Text                                                │
│      ↓                                                      │
│  Pretokenization (regex) → Word/Number/Punctuation Chunks   │
│      ↓                                                      │
│  Byte Encoding → Tuple of Byte Integers                     │
│      ↓                                                      │
│  BPE Training Loop                                          │
│    - Count adjacent pairs (parallelized)                  │
│    - Merge most frequent pair                               │
│    - Update corpus                                          │
│    - Repeat until target vocab size                         │
│      ↓                                                      │
│  Save: vocab.json + merges.txt + config.json                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      Inference Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│  Input Text                                                 │
│      ↓                                                      │
│  Pretokenization → Chunks                                   │
│      ↓                                                      │
│  Byte → Vocab ID Mapping                                    │
│      ↓                                                      │
│  Apply Merges (in training order)                           │
│      ↓                                                      │
│  Token IDs                                                  │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Requirements

- Python 3.8+
- `regex` module (required for Unicode property matching)

```bash
pip install regex
```

### Clone Repository

```bash
git clone <repository-url>
cd myTokenizer
```

## Quick Start

### 1. Train a Tokenizer

```python
from train import train

# Train on your corpus
train(
    corpus_path="./training_data/raw/openwebtext.txt",
    vocab_size=10000,
    save_dir="saved",
    max_lines=50000  # Limit for faster iteration
)
```

Or run directly:

```bash
python train.py
```

### 2. Load and Use

```python
from tokenizer import Tokenizer

# Load trained tokenizer
tokenizer = Tokenizer.from_pretrained("saved/")

# Encode text
text = "Hello, world!"
ids = tokenizer.encode(text, add_special_tokens=True)
print(f"Token IDs: {ids}")
# Output: [2, 72, 101, 108, 108, 111, ... , 3]  # With BOS/EOS

# Decode back
decoded = tokenizer.decode(ids, skip_special_tokens=True)
print(f"Decoded: {decoded}")
# Output: "Hello, world!"
```

### 3. Validate

```bash
python validate.py
```

Runs comprehensive tests: round-trip encoding, compression metrics, special tokens, edge cases, Unicode handling, and consistency checks.

## API Reference

### `Tokenizer`

#### `Tokenizer.from_pretrained(model_dir: str) -> Tokenizer`

Load a trained tokenizer from disk.

**Parameters:**
- `model_dir`: Path to directory containing `vocab.json`, `merges.txt`, and `config.json`

**Returns:**
- Initialized `Tokenizer` instance

#### `encode(text: str, add_special_tokens: bool = False) -> list[int]`

Encode text into token IDs.

**Parameters:**
- `text`: Input text string
- `add_special_tokens`: If `True`, wraps output with `<BOS>` and `<EOS>` tokens

**Returns:**
- List of token IDs

#### `decode(ids: list[int], skip_special_tokens: bool = True) -> str`

Decode token IDs back to text.

**Parameters:**
- `ids`: List of token IDs
- `skip_special_tokens`: If `True`, removes `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>` from output

**Returns:**
- Decoded text string

#### `get_vocab_size() -> int`

Returns vocabulary size.

#### `token_to_id(token: str) -> int`

Get ID for a token string. Returns `<UNK>` ID if token not in vocabulary.

#### `id_to_token_str(id: int) -> str`

Get token string for an ID.

### `train`

#### `train(corpus_path: str, vocab_size: int = 500, save_dir: str = "saved", max_lines: int = None)`

Train a BPE tokenizer on a text corpus.

**Parameters:**
- `corpus_path`: Path to training text file
- `vocab_size`: Target vocabulary size (default: 500)
- `save_dir`: Directory to save trained tokenizer files
- `max_lines`: Optional limit on lines to read from corpus

**Output Files:**
- `vocab.json`: Token string to ID mapping
- `merges.txt`: Merge rules in format `id_a id_b new_id`
- `config.json`: Metadata including special token IDs

## Technical Details

### Pretokenization

The tokenizer uses GPT-2's regex pattern to split text before BPE processing:

```regex
'(?i:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
```

This ensures:
- Contractions stay together (don't, I'll, we've)
- Words aren't split across chunk boundaries
- Numbers and punctuation are handled correctly
- Multiple languages (Unicode) are supported via `\p{L}` and `\p{N}`

**Note:** Requires the `regex` module, not Python's built-in `re`.

### Base Vocabulary

The tokenizer starts with 260 tokens:

| Token ID | Description |
|----------|-------------|
| 0 | `<PAD>` - Padding token |
| 1 | `<UNK>` - Unknown token |
| 2 | `<BOS>` - Beginning of sequence |
| 3 | `<EOS>` - End of sequence |
| 4-259 | Single bytes (0x00-0xFF) encoded as latin-1 strings |

### BPE Training Algorithm

1. **Initialize**: Start with base vocabulary (260 tokens)
2. **Count Pairs**: Find all adjacent token pairs in corpus (parallelized)
3. **Select Merge**: Choose most frequent pair
4. **Create Token**: Concatenate pair strings, assign new ID
5. **Apply Merge**: Replace all occurrences in corpus
6. **Repeat**: Until reaching target vocabulary size

### Encoding Strategy

When encoding new text:

1. Pretokenize into chunks
2. Convert each chunk to byte IDs
3. Apply merges **in training order** (earliest learned merges have priority)
4. Continue until no more merges possible

This deterministic approach ensures consistent tokenization.

### Byte Handling

Since Python strings are Unicode, the tokenizer uses a latin-1 encoding bridge:

- **Training**: Bytes → latin-1 decode → vocabulary tokens
- **Encoding**: Text → UTF-8 bytes → vocab IDs → merges
- **Decoding**: Token IDs → latin-1 bytes → UTF-8 decode

This allows representation of any byte sequence (0-255) as a valid Python string.

## Performance

### Training Optimizations

- **Parallel pair counting**: Uses `multiprocessing.Pool` for large corpora (>1000 chunks)
- **Set membership tests**: `O(1)` lookup to skip chunks without target pair
- **Early termination**: Stops when no more pairs can be merged

### Benchmarks

Typical performance on OpenWebText sample:

| Metric | Value |
|--------|-------|
| Training (10k vocab, 50k lines) | ~3-5 hours for complete training |
| Encoding | ~10k-50k chars/second |
| Compression | 2.5-4.0 chars/token (depends on vocab size) |

## Testing

The `validate.py` module provides comprehensive testing:

### Test Categories

1. **Round-Trip**: Verify `encode(decode(x)) == x`
2. **Compression**: Measure chars/token and bytes/token ratios
3. **Special Tokens**: Test `<BOS>`, `<EOS>`, `<PAD>`, `<UNK>` handling
4. **Edge Cases**: Empty strings, whitespace, single characters, control chars
5. **Unicode**: Multi-script text (Chinese, Japanese, Korean, Arabic, emojis)
6. **Consistency**: Deterministic encoding of same input

### Running Tests

```bash
python validate.py
```

Output includes pass/fail status and compression metrics.

## Project Structure

```
myTokenizer/
├── pretokenize.py      # Regex-based text splitting
├── train.py            # BPE training pipeline
├── tokenizer.py        # Inference Tokenizer class
├── validate.py         # Comprehensive test suite
├── test.py             # Development tests (deprecated)
├── test2.py            # Additional development tests (deprecated)
├── training_data/
│   └── raw/
│       └── openwebtext.txt    # Training corpus
└── saved/
    ├── vocab.json      # Token → ID mapping
    ├── merges.txt      # Merge rules
    └── config.json     # Metadata
```

## References

- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) - Original BPE paper
- [GPT-2 Tokenizer](https://github.com/openai/gpt-2) - Reference implementation
- [tiktoken](https://github.com/openai/tiktoken) - Reference implementation
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/) - Production tokenizer library
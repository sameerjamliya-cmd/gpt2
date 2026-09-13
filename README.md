# GPT From Scratch

A GPT style language model I built from the ground up, following Karpathy's
["Let's build GPT: from scratch, in code, spelled out"](https://youtu.be/kCc8FmEb1nY).
I wanted to actually understand every piece of a transformer, not just wire
together library calls.

No `transformers`, no pretrained weights, no shortcuts. Just PyTorch, starting
from a bigram baseline and building up to a working multi head attention
transformer.

## Why I built this

Most people learn GPT by calling an API or fine tuning a checkpoint. I went
the other direction. I built the smallest possible model, one that has no
idea what "context" even means, and then added exactly one capability at a
time, watching what broke and why, until it became a real transformer.

My goal wasn't to reproduce GPT-2. It was to understand attention, residual
connections, and autoregressive generation well enough that I can explain
why each piece exists, not just that it exists.

## Model config

| Param | Value |
|---|---|
| Parameters | ~<!-- e.g. 10-14M --> M |
| `n_embd` | <!-- e.g. 384 --> |
| `n_head` | <!-- e.g. 6 --> |
| `n_layer` | <!-- e.g. 6 --> |
| `block_size` | <!-- e.g. 256 --> |
| `vocab_size` | <!-- e.g. 65 (char level) --> |
| Tokenizer | <!-- char level / BPE --> |
| Dataset | <!-- e.g. tiny Shakespeare --> |
| Hardware | Trained on Apple Silicon (MPS backend) |

## What's actually inside

I built this in stages, each one solving a problem the previous stage
couldn't handle.

**1. Bigram baseline**
A lookup table (`nn.Embedding(vocab_size, vocab_size)`) that predicts the
next token using only the current token, no context at all. This exists to
nail down the training loop, loss function, and generation loop before I
added any real modeling power.

**2. Self attention**
This replaces "no context" with a mechanism that lets each token look at
every token before it and decide, dynamically, how much attention each one
deserves. It's built from Query, Key, and Value projections:

**Query** is what this token is looking for.
**Key** is what each other token has to offer.
**Value** is what actually gets passed along once relevance is decided.

Causal masking (`torch.tril`) ensures a token can never attend to the
future, which is required for valid autoregressive generation.

**3. Multi head attention**
This runs several attention "specialists" in parallel instead of one, each
with its own learned Q/K/V weights, then combines their outputs. It lets the
model track multiple kinds of relationships between tokens at once.

**4. Feedforward**
Attention gathers context across tokens. Feedforward processes it, per
token, independently. It's a small 2 layer MLP (expand 4x, ReLU, compress)
that gives the model nonlinear capacity. Without it, stacking layers would
just collapse into one linear transformation.

**5. Transformer block**
This wraps attention and feedforward with residual connections and pre
LayerNorm:
```
x = x + attention(norm(x))
x = x + feedforward(norm(x))
```
The residual (`x + ...` instead of `x = ...`) is what makes deep stacks
trainable. Each block only has to learn a correction to add, not a full
reconstruction.

**6. Full model**
Token embeddings plus positional embeddings (attention has no innate sense
of order, so this adds it), then N stacked transformer blocks, then a final
LayerNorm, then a linear projection to vocabulary logits.

## What this is not

This is not GPT-2 at scale (124M+ params). I kept it small on purpose so I
could train and iterate on a single machine. It also doesn't use a
pretrained tokenizer or pretrained weights, and there's no KV caching, flash
attention, or other inference time optimizations. Those are a separate
concern from actually understanding the architecture.

## Sample output

```
<!-- paste a generation sample here once you have one -->
```

## Running it

```bash
pip install -r requirements.txt
python train.py
```

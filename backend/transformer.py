"""
transformer.py
==============
Core Transformer math implemented from scratch using NumPy.
Every function is annotated line-by-line so you can trace EXACTLY
what happens to the sentence "The dog chased the cat because it was scared".

Architecture:
  Input text
    → Tokenisation (text → integer IDs)
    → Embedding lookup  (IDs → dense vectors, shape [seq_len, d_model])
    → Positional Encoding  (add sine/cosine position signal)
    → Multi-Head Self-Attention  (N heads, each computing Q·K^T / √d_k)
    → Softmax  (convert raw scores → probabilities)
    → Weighted sum of Values  (produce context-aware representation)
    → Feed-Forward Network  (per-token MLP: Linear → GELU → Linear)
    → Layer Norm  (stabilise activations)
    → Output logits → next-token probabilities
"""

import numpy as np          # All our tensor math lives here
import math                 # For sqrt, pi used in GELU

# ─────────────────────────────────────────────
# CONSTANTS  – kept small so we can visualise
# ─────────────────────────────────────────────
D_MODEL   = 64   # Embedding dimension (GPT-2 uses 768, GPT-4 ~12288; we use 64 for clarity)
N_HEADS   = 4    # Number of attention heads (GPT-2 uses 12)
D_HEAD    = D_MODEL // N_HEADS   # Dimension per head = 64 / 4 = 16
D_FF      = 128  # Feed-forward inner dimension (usually 4 × d_model)
SEQ_LEN   = 9    # Our sentence has 9 tokens

# ─────────────────────────────────────────────
# VOCABULARY  – a tiny vocabulary just for demo
# ─────────────────────────────────────────────
VOCAB = {
    # Special tokens
    "[PAD]": 0,  "[UNK]": 1,
    # Our sentence tokens
    "the":  2,  "dog":     3,  "chased": 4,
    "cat":  5,  "because": 6,  "it":     7,
    "was":  8,  "scared":  9,
    # A few extra words so the output vocabulary has interesting predictions
    "ran":  10, "jumped":  11, "slept":  12, "barked": 13, "hid":    14,
    "very": 15, "so":      16, "quite":  17,
}
# Reverse vocab so we can decode IDs back to words
ID_TO_WORD = {v: k for k, v in VOCAB.items()}
VOCAB_SIZE  = len(VOCAB)   # 18 total tokens in our mini vocabulary


# ═══════════════════════════════════════════════════
# STEP 1 – TOKENISATION
# ═══════════════════════════════════════════════════
def tokenize(text: str) -> tuple[list[str], list[int]]:
    """
    Split a raw string into a list of token strings and their integer IDs.

    Example:
        "The dog" → (["the", "dog"], [2, 3])

    In production models (GPT, BERT) this uses Byte-Pair Encoding (BPE)
    or WordPiece, which can split "chasing" → ["chas", "##ing"].
    Here we use simple whitespace splitting + lower-case lookup.
    """
    # Lower-case and split on whitespace / punctuation
    words = text.lower().replace(",", "").replace(".", "").strip().split()

    tokens  = []  # human-readable token strings
    ids     = []  # integer IDs from VOCAB dictionary

    for word in words:
        # Look up the word; fall back to [UNK] (ID=1) if not in vocab
        token_id = VOCAB.get(word, VOCAB["[UNK]"])
        tokens.append(word)
        ids.append(token_id)

    return tokens, ids   # e.g. (["the","dog",...], [2,3,...])


# ═══════════════════════════════════════════════════
# STEP 2 – EMBEDDING LOOKUP
# ═══════════════════════════════════════════════════
def get_embedding_matrix(seed: int = 42) -> np.ndarray:
    """
    Return the token embedding matrix E of shape [VOCAB_SIZE, D_MODEL].

    Each row E[i] is the embedding vector for token ID i.
    In real transformers, this matrix is learned during training on
    billions of text examples via gradient descent.
    Here we generate it once with a fixed random seed so results are
    deterministic and reproducible.

    Shape: [18, 64]  → 18 vocabulary entries, each a 64-dim vector.
    """
    rng = np.random.RandomState(seed)   # seeded for reproducibility
    # Sample from Normal(0, 0.02) — same initialisation as GPT-2
    E = rng.randn(VOCAB_SIZE, D_MODEL).astype(np.float32) * 0.02
    return E   # shape: [VOCAB_SIZE, D_MODEL]


def embed_tokens(token_ids: list[int], E: np.ndarray) -> np.ndarray:
    """
    Look up every token ID in the embedding matrix.

    Args:
        token_ids : list of int, e.g. [2, 3, 4, 2, 5, 6, 7, 8, 9]
        E         : embedding matrix, shape [VOCAB_SIZE, D_MODEL]

    Returns:
        x  : np.ndarray, shape [seq_len, D_MODEL]
             Each row is the embedding vector for that position's token.

    Example (conceptual):
        token_ids[0] = 2  ("the")
        x[0]         = E[2]   ← 64 floats representing "the"
    """
    # NumPy fancy indexing: E[token_ids] stacks the rows we want
    x = E[token_ids]   # shape: [seq_len, D_MODEL]
    return x


# ═══════════════════════════════════════════════════
# STEP 3 – POSITIONAL ENCODING
# ═══════════════════════════════════════════════════
def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    Compute sinusoidal positional encodings (Vaswani et al., 2017).

    Because self-attention is permutation-invariant (it doesn't inherently
    know word order), we inject explicit position information by adding a
    unique signal to each position.

    Formula:
        PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
        PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))

    Why sin/cos?
      • Different frequencies mean every position has a unique fingerprint.
      • The model can attend to *relative* positions because sin(a−b) can
        be expressed as a linear function of sin(a) and sin(b).

    Returns:
        PE : np.ndarray, shape [seq_len, d_model]
    """
    PE = np.zeros((seq_len, d_model), dtype=np.float32)   # blank canvas

    # pos = token position (0 to seq_len-1)
    # i   = dimension index (0 to d_model//2 - 1)
    for pos in range(seq_len):
        for i in range(d_model // 2):
            # Denominator grows exponentially: 10000^(2i/d_model)
            denom = 10000 ** (2 * i / d_model)

            PE[pos, 2 * i]     = math.sin(pos / denom)   # even dims ← sin
            PE[pos, 2 * i + 1] = math.cos(pos / denom)   # odd  dims ← cos

    return PE   # shape: [seq_len, d_model]


# ═══════════════════════════════════════════════════
# STEP 4 – WEIGHT MATRICES  (W_Q, W_K, W_V, W_O)
# ═══════════════════════════════════════════════════
def get_attention_weights(seed: int = 99) -> dict:
    """
    Generate the learnable projection matrices for Multi-Head Attention.

    In a real transformer these are trained; here we use a fixed seed.

    Each attention head h has three weight matrices:
        W_Q[h] : [D_MODEL, D_HEAD]  — projects input into Query space
        W_K[h] : [D_MODEL, D_HEAD]  — projects input into Key space
        W_V[h] : [D_MODEL, D_HEAD]  — projects input into Value space

    Plus one output projection:
        W_O    : [N_HEADS * D_HEAD, D_MODEL]  — projects concat heads back

    Returns a dict with all weight matrices.
    """
    rng = np.random.RandomState(seed)
    scale = 1.0 / math.sqrt(D_HEAD)   # Xavier-like initialisation scaling

    weights = {
        # One set of Q/K/V matrices per head
        "W_Q": rng.randn(N_HEADS, D_MODEL, D_HEAD).astype(np.float32) * scale,
        "W_K": rng.randn(N_HEADS, D_MODEL, D_HEAD).astype(np.float32) * scale,
        "W_V": rng.randn(N_HEADS, D_MODEL, D_HEAD).astype(np.float32) * scale,
        # Output projection: collapses all heads back to d_model
        "W_O": rng.randn(N_HEADS * D_HEAD, D_MODEL).astype(np.float32) * scale,
    }
    return weights


# ═══════════════════════════════════════════════════
# STEP 5 – SCALED DOT-PRODUCT ATTENTION  (single head)
# ═══════════════════════════════════════════════════
def scaled_dot_product_attention(
    Q: np.ndarray,   # shape: [seq_len, d_head]
    K: np.ndarray,   # shape: [seq_len, d_head]
    V: np.ndarray,   # shape: [seq_len, d_head]
    mask: np.ndarray | None = None,  # causal mask for decoder (optional)
) -> tuple[np.ndarray, np.ndarray]:
    """
    The fundamental attention operation:

        Attention(Q, K, V) = softmax(Q · Kᵀ / √d_k) · V

    Step-by-step:
      1. Compute raw scores: S = Q · Kᵀ         shape [seq, seq]
      2. Scale:              S = S / √d_k        prevents vanishing gradients
      3. (Optional) mask:    S[future] = -∞      for autoregressive decoding
      4. Softmax:            A = softmax(S, dim=-1)  probabilities sum to 1
      5. Weighted sum:       out = A · V         aggregate value vectors

    Returns:
        out    : np.ndarray, shape [seq_len, d_head]  — context-aware output
        attn_w : np.ndarray, shape [seq_len, seq_len] — attention weights (probabilities)
    """
    d_k = Q.shape[-1]   # dimensionality of each key/query vector (= D_HEAD = 16)

    # ── 1. Raw dot-product scores ────────────────────────────────────────────
    # For each query position i, compute similarity with every key position j
    # Q shape: [seq, d_k],  K.T shape: [d_k, seq]  →  S shape: [seq, seq]
    scores = Q @ K.T                                # matrix multiply

    # ── 2. Scale ─────────────────────────────────────────────────────────────
    # Without scaling, for large d_k the dot products grow large → softmax
    # saturates → gradients vanish.  Divide by √d_k stabilises training.
    scores = scores / math.sqrt(d_k)               # element-wise division

    # ── 3. Causal mask (decoder-only models) ─────────────────────────────────
    # In decoder-only models (GPT, Claude), token i cannot see token j > i
    # because we generate left-to-right.  We set future scores to -1e9
    # so softmax maps them to ≈ 0.
    if mask is not None:
        scores = scores + mask * (-1e9)             # apply causal mask

    # ── 4. Softmax → attention probabilities ─────────────────────────────────
    # softmax(x_i) = exp(x_i) / Σ exp(x_j)
    # Subtract row max for numerical stability (equivalent result, no overflow)
    scores_stable = scores - scores.max(axis=-1, keepdims=True)   # shift
    exp_scores    = np.exp(scores_stable)                          # exponentiate
    attn_weights  = exp_scores / exp_scores.sum(axis=-1, keepdims=True)  # normalise
    # attn_weights shape: [seq_len, seq_len], each row sums to 1.0

    # ── 5. Weighted sum of Values ─────────────────────────────────────────────
    # For each query position, blend the Value vectors weighted by attention
    # attn_weights: [seq, seq],  V: [seq, d_head]  →  out: [seq, d_head]
    out = attn_weights @ V                          # context-aware representation

    return out, attn_weights   # return both output AND weights for visualisation


# ═══════════════════════════════════════════════════
# STEP 6 – MULTI-HEAD ATTENTION
# ═══════════════════════════════════════════════════
def multi_head_attention(
    x: np.ndarray,          # shape: [seq_len, D_MODEL]  — input embeddings
    weights: dict,          # Q/K/V/O weight matrices
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Run N_HEADS independent attention heads in parallel, then concatenate.

    Each head learns to attend to *different* relationships:
        Head 1: pronoun resolution  ("it" → "cat")
        Head 2: subject-verb links  ("dog" ↔ "chased")
        Head 3: sentiment flow      ("scared" context)
        Head 4: positional proximity (nearby word bias)

    The heads use *different* W_Q / W_K / W_V matrices, so they project
    the input into different subspaces and can capture different patterns.

    Returns:
        mha_out   : shape [seq_len, D_MODEL]   — output after projection
        all_attn  : list of N_HEADS attention weight matrices [seq, seq]
    """
    all_heads = []    # will hold each head's output vector [seq, d_head]
    all_attn  = []    # will hold each head's attention weights [seq, seq]

    # ── Run each head independently ──────────────────────────────────────────
    for h in range(N_HEADS):
        # Project input x into this head's Q, K, V spaces
        # x: [seq, D_MODEL]  @  W_Q[h]: [D_MODEL, D_HEAD]  →  Q: [seq, D_HEAD]
        Q_h = x @ weights["W_Q"][h]   # "What am I looking for?"
        K_h = x @ weights["W_K"][h]   # "What do I offer?"
        V_h = x @ weights["W_V"][h]   # "What info do I share?"

        # Run scaled dot-product attention for this head
        head_out, attn_w = scaled_dot_product_attention(Q_h, K_h, V_h, mask)
        # head_out: [seq, D_HEAD],  attn_w: [seq, seq]

        all_heads.append(head_out)   # collect head output
        all_attn.append(attn_w)      # collect attention weights for viz

    # ── Concatenate all heads along the feature dimension ────────────────────
    # each head_out: [seq, D_HEAD=16],  after concat: [seq, N_HEADS*D_HEAD=64]
    concat = np.concatenate(all_heads, axis=-1)   # shape: [seq, N_HEADS*D_HEAD]

    # ── Final linear projection: collapse back to D_MODEL ────────────────────
    # concat: [seq, N_HEADS*D_HEAD]  @  W_O: [N_HEADS*D_HEAD, D_MODEL]
    mha_out = concat @ weights["W_O"]   # shape: [seq, D_MODEL]

    return mha_out, all_attn   # return output + per-head attention for viz


# ═══════════════════════════════════════════════════
# STEP 7 – LAYER NORMALISATION
# ═══════════════════════════════════════════════════
def layer_norm(
    x: np.ndarray,      # shape: [seq_len, D_MODEL]
    eps: float = 1e-6,  # small constant to avoid division by zero
) -> np.ndarray:
    """
    Normalise each token's vector to zero mean and unit variance,
    then apply learned scale (γ) and shift (β) — both set to 1, 0 here.

    Formula:
        LN(x) = γ · (x − μ) / √(σ² + ε) + β

    Why?
      Without normalisation, activations can blow up or vanish as they
      pass through many layers, making training unstable.

    Returns:
        out : np.ndarray, shape [seq_len, D_MODEL]
    """
    # Compute mean and variance per token (across D_MODEL dimensions)
    mean = x.mean(axis=-1, keepdims=True)           # shape: [seq, 1]
    var  = x.var(axis=-1,  keepdims=True)            # shape: [seq, 1]

    # Normalise: subtract mean, divide by std
    x_norm = (x - mean) / np.sqrt(var + eps)         # shape: [seq, D_MODEL]

    # γ=1, β=0 (identity transform) — in real models these are learned
    return x_norm                                     # shape: [seq, D_MODEL]


# ═══════════════════════════════════════════════════
# STEP 8 – FEED-FORWARD NETWORK (per token)
# ═══════════════════════════════════════════════════
def gelu(x: np.ndarray) -> np.ndarray:
    """
    Gaussian Error Linear Unit activation function.
    Smoother than ReLU; used by GPT-2 and later transformers.

    Approximation: GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
    """
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


def feed_forward_network(
    x: np.ndarray,   # shape: [seq_len, D_MODEL]
    seed: int = 77,
) -> np.ndarray:
    """
    A two-layer MLP applied *identically and independently* to each token.

    Architecture:
        x  →  Linear(D_MODEL → D_FF)  →  GELU  →  Linear(D_FF → D_MODEL)

    D_FF = 4 × D_MODEL in canonical transformers (64 × 4 = 256; we use 128).
    This expansion-then-contraction acts like a "thinking layer" where
    the model can store and recall factual associations (e.g., "cat → animal").

    Returns:
        out : np.ndarray, shape [seq_len, D_MODEL]
    """
    rng = np.random.RandomState(seed)
    scale = 1.0 / math.sqrt(D_MODEL)

    # Weight matrices for the two linear layers (fixed / deterministic)
    W1 = rng.randn(D_MODEL, D_FF).astype(np.float32) * scale   # [D_MODEL, D_FF]
    b1 = np.zeros(D_FF, dtype=np.float32)                       # bias, shape [D_FF]

    W2 = rng.randn(D_FF, D_MODEL).astype(np.float32) * scale   # [D_FF, D_MODEL]
    b2 = np.zeros(D_MODEL, dtype=np.float32)                    # bias, shape [D_MODEL]

    # ── Forward pass ─────────────────────────────────────────────────────────
    # Layer 1: project up from D_MODEL → D_FF
    hidden = x @ W1 + b1        # shape: [seq, D_FF]
    # GELU activation introduces non-linearity so the model can learn
    # complex (non-linear) relationships
    hidden = gelu(hidden)       # shape: [seq, D_FF]
    # Layer 2: project back down D_FF → D_MODEL
    out    = hidden @ W2 + b2   # shape: [seq, D_MODEL]

    return out   # shape: [seq, D_MODEL]


# ═══════════════════════════════════════════════════
# STEP 9 – ONE TRANSFORMER BLOCK
# ═══════════════════════════════════════════════════
def transformer_block(
    x: np.ndarray,          # shape: [seq_len, D_MODEL]
    weights: dict,          # attention weight matrices
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    A single Transformer layer:

        x  →  MultiHeadAttn  →  Add & LayerNorm
                             →  FeedForward    →  Add & LayerNorm  →  out

    The "Add" is a *residual connection*: we add the original input x
    to the sublayer output.  This means:
      • Gradients flow unchanged through the skip path during training
      • The layer only needs to learn a *residual correction*, not the
        full mapping → easier to optimise

    Real GPT-2 small has 12 such blocks stacked; GPT-4 has ~96.
    We run 1 block for clarity.
    """
    # ── Sub-layer 1: Multi-Head Self-Attention ────────────────────────────────
    attn_out, all_attn = multi_head_attention(x, weights, mask)
    # Residual connection + Layer Norm
    x = layer_norm(x + attn_out)   # shape: [seq, D_MODEL]

    # ── Sub-layer 2: Feed-Forward Network ────────────────────────────────────
    ff_out = feed_forward_network(x)
    # Residual connection + Layer Norm
    x = layer_norm(x + ff_out)     # shape: [seq, D_MODEL]

    return x, all_attn   # (processed embeddings, per-head attention weights)


# ═══════════════════════════════════════════════════
# STEP 10 – OUTPUT HEAD (next-token prediction)
# ═══════════════════════════════════════════════════
def output_projection(
    x: np.ndarray,   # shape: [seq_len, D_MODEL]  — last token's representation
    E: np.ndarray,   # embedding matrix [VOCAB_SIZE, D_MODEL]  — weight tying
) -> np.ndarray:
    """
    Project the final hidden state to logits over the vocabulary.

    We use "weight tying": the output projection matrix is the *transpose*
    of the input embedding matrix E.  This was shown to improve perplexity
    and halves the parameter count.

        logits = h · Eᵀ   where h = last token's hidden state

    Shape:
        h      : [D_MODEL]      (just the LAST token, which "predicts" next)
        E.T    : [D_MODEL, V]
        logits : [V]            one score per vocabulary item

    Then softmax converts logits → probabilities.
    """
    h = x[-1]           # take the LAST token's hidden state for next-token pred
    logits = h @ E.T    # project to vocabulary space;  shape: [VOCAB_SIZE]

    # Softmax to get probabilities
    logits_stable = logits - logits.max()          # numerical stability
    probs = np.exp(logits_stable) / np.exp(logits_stable).sum()

    return probs   # shape: [VOCAB_SIZE]


# ═══════════════════════════════════════════════════
# MASTER FUNCTION – full forward pass
# ═══════════════════════════════════════════════════
def full_forward_pass(text: str, use_causal_mask: bool = False) -> dict:
    """
    Run the complete transformer forward pass and return ALL intermediate
    values needed for the interactive visualisation.

    Returns a dict with:
      tokens         : list of token strings
      token_ids      : list of int
      embeddings     : [seq, D_MODEL]
      pos_encoding   : [seq, D_MODEL]
      input_vectors  : [seq, D_MODEL]  (embeddings + pos_encoding)
      attention_heads: list of N_HEADS arrays, each [seq, seq]
      output_hidden  : [seq, D_MODEL]  (after transformer block)
      next_token_probs: dict {word: float}  top-k probabilities
    """
    # ── 1. Tokenise ──────────────────────────────────────────────────────────
    tokens, token_ids = tokenize(text)
    seq_len = len(token_ids)

    # ── 2. Build the fixed weight matrices ───────────────────────────────────
    E       = get_embedding_matrix(seed=42)           # embedding table
    weights = get_attention_weights(seed=99)          # Q/K/V/O weights

    # ── 3. Token embeddings ───────────────────────────────────────────────────
    embeddings = embed_tokens(token_ids, E)           # [seq, D_MODEL]

    # ── 4. Positional encoding ────────────────────────────────────────────────
    pos_enc = positional_encoding(seq_len, D_MODEL)   # [seq, D_MODEL]

    # ── 5. Input = embeddings + positional encoding ───────────────────────────
    x = embeddings + pos_enc                          # [seq, D_MODEL]

    # ── 6. Optional causal mask (for decoder-only behaviour) ─────────────────
    mask = None
    if use_causal_mask:
        # Upper-triangular mask: position i cannot see position j > i
        # Shape: [seq, seq], with 1s where we want to block attention
        mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float32), k=1)

    # ── 7. Transformer block (attention + FFN) ────────────────────────────────
    output_hidden, all_attn = transformer_block(x, weights, mask)
    # output_hidden : [seq, D_MODEL]
    # all_attn      : list of N_HEADS arrays, each [seq, seq]

    # ── 8. Next-token prediction ───────────────────────────────────────────────
    probs = output_projection(output_hidden, E)       # [VOCAB_SIZE]

    # Convert probabilities to a readable dict {word: prob}
    prob_dict = {
        ID_TO_WORD[i]: float(round(p, 4))
        for i, p in enumerate(probs)
    }
    # Sort by probability, return top-k
    top_probs = dict(
        sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)[:10]
    )

    # ── 9. Package everything for the API ────────────────────────────────────
    return {
        "tokens":          tokens,
        "token_ids":       token_ids,
        # Round embeddings to 4 decimals for clean JSON
        "embeddings":      embeddings.round(4).tolist(),
        "pos_encoding":    pos_enc.round(4).tolist(),
        "input_vectors":   x.round(4).tolist(),
        # Convert each head's attention matrix [seq,seq] → list of lists
        "attention_heads": [a.round(4).tolist() for a in all_attn],
        "output_hidden":   output_hidden.round(4).tolist(),
        "next_token_probs": top_probs,
        # Metadata for the UI
        "d_model":  D_MODEL,
        "n_heads":  N_HEADS,
        "d_head":   D_HEAD,
        "vocab":    VOCAB,
    }

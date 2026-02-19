"""
main.py  –  FastAPI Server
==========================
This is the web server that sits between the React frontend and our
transformer mathematics.  It exposes HTTP endpoints that the browser
calls via fetch() / axios, receives JSON, and returns JSON.

Key FastAPI concepts used here:
  • @app.get / @app.post  – route decorators (like Flask's @route)
  • BaseModel             – Pydantic schema for automatic request validation
  • JSONResponse          – send back arbitrary JSON
  • CORSMiddleware        – allow the React dev server (port 5173) to call us
  • uvicorn               – the ASGI server that runs this app

Run with:
    uvicorn main:app --reload --port 8000
"""

# ─── Standard library ────────────────────────────────────────────────────────
import time          # for measuring computation duration

# ─── Third-party ─────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException   # web framework + error raising
from fastapi.middleware.cors import CORSMiddleware  # allow cross-origin requests
from fastapi.responses import JSONResponse  # explicit JSON responses
from pydantic import BaseModel, Field      # request body schemas with validation

# ─── Our transformer logic ───────────────────────────────────────────────────
from transformer import (
    full_forward_pass,   # runs the entire transformer pipeline
    tokenize,            # text → tokens + IDs
    positional_encoding, # sin/cos position signals
    VOCAB,               # our mini vocabulary dict
    D_MODEL,             # embedding dimension (64)
    N_HEADS,             # number of attention heads (4)
)


# ════════════════════════════════════════════════════════════════════════════
# APPLICATION SETUP
# ════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title       = "Transformer Explainer API",
    description = "Interactive Transformer Architecture Visualiser — Backend",
    version     = "1.0.0",
    # These endpoints will be shown in the auto-generated docs at /docs
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── CORS Middleware ───────────────────────────────────────────────────────────
# The React frontend runs on http://localhost:5173 (Vite dev server).
# Browsers block cross-origin requests by default (CORS policy).
# We explicitly allow our frontend origins here.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite (React) dev server
        "http://localhost:3000",   # Create-React-App dev server (alt)
        "http://127.0.0.1:5173",
    ],
    allow_credentials = True,
    allow_methods     = ["*"],    # GET, POST, OPTIONS, etc.
    allow_headers     = ["*"],    # Content-Type, Authorization, etc.
)


# ════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS  (Pydantic validates these automatically)
# ════════════════════════════════════════════════════════════════════════════

class TextInput(BaseModel):
    """
    Schema for any endpoint that accepts a text string.
    Pydantic will reject the request automatically if 'text' is missing
    or not a string.
    """
    text: str = Field(
        default = "The dog chased the cat because it was scared",
        min_length = 1,
        max_length = 200,
        description = "Input sentence to run through the transformer",
    )
    use_causal_mask: bool = Field(
        default = False,
        description = "If True, apply decoder-style causal mask (can't see future tokens)",
    )


class AttentionHeadInput(BaseModel):
    """Schema for requesting a specific attention head's weights."""
    text:      str = "The dog chased the cat because it was scared"
    head_idx:  int = Field(0, ge=0, lt=N_HEADS,
                           description="Which attention head (0 to N_HEADS-1)")


# ════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════

# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    """
    GET /
    Simple health-check endpoint.
    The React app calls this on startup to verify the backend is running.
    """
    return {
        "status":  "ok",
        "message": "Transformer Explainer API is running",
        "docs":    "Visit /docs for the interactive API explorer",
    }


@app.get("/health")
async def health():
    """GET /health — quick liveness probe (Docker, k8s, load balancers use this)."""
    return {"status": "healthy", "timestamp": time.time()}


# ── Tokenisation endpoint ─────────────────────────────────────────────────────
@app.post("/api/tokenize")
async def api_tokenize(body: TextInput):
    """
    POST /api/tokenize
    ──────────────────
    Takes raw text and returns:
      • tokens    : list of lowercase word strings
      • token_ids : integer ID for each token (from our mini vocab)
      • vocab_size: total vocabulary size

    The React frontend calls this first to display the "Tokenisation" panel.

    Example request body:
        { "text": "The dog chased the cat because it was scared" }

    Example response:
        {
          "tokens":     ["the", "dog", "chased", ...],
          "token_ids":  [2, 3, 4, ...],
          "vocab_size": 18
        }
    """
    # Run our tokenise function (see transformer.py, STEP 1)
    tokens, token_ids = tokenize(body.text)

    return {
        "tokens":    tokens,       # ["the", "dog", "chased", ...]
        "token_ids": token_ids,    # [2, 3, 4, ...]
        "vocab":     VOCAB,        # the full vocabulary dict {word: id}
        "vocab_size": len(VOCAB),  # 18
    }


# ── Positional encoding endpoint ─────────────────────────────────────────────
@app.post("/api/positional-encoding")
async def api_positional_encoding(body: TextInput):
    """
    POST /api/positional-encoding
    ──────────────────────────────
    Returns the sinusoidal positional encoding matrix for the input sentence.
    Shape: [seq_len, d_model]

    The frontend uses this to draw the "Position Encoding" heatmap showing
    how each dimension varies across positions.
    """
    tokens, _ = tokenize(body.text)    # need seq_len
    seq_len   = len(tokens)

    pe = positional_encoding(seq_len, D_MODEL)   # numpy [seq, D_MODEL]

    return {
        "tokens":          tokens,
        "seq_len":         seq_len,
        "d_model":         D_MODEL,
        "pos_encoding":    pe.tolist(),   # converts numpy → plain Python list
    }


# ── Full forward pass endpoint ────────────────────────────────────────────────
@app.post("/api/forward")
async def api_forward(body: TextInput):
    """
    POST /api/forward
    ──────────────────
    Runs the COMPLETE transformer forward pass and returns every intermediate
    tensor needed to drive all visualisation panels:

      • tokens           : token strings
      • token_ids        : integer IDs
      • embeddings       : [seq, d_model]  — raw lookup vectors
      • pos_encoding     : [seq, d_model]  — sinusoidal signals
      • input_vectors    : [seq, d_model]  — embeddings + pos_encoding
      • attention_heads  : list of N_HEADS arrays, each [seq, seq]
      • output_hidden    : [seq, d_model]  — after transformer block
      • next_token_probs : top-10 {word: probability} for next-token prediction

    This is the primary endpoint; the React App calls it on every input change.
    """
    t0 = time.perf_counter()   # start timer to measure inference time

    try:
        # Delegate to transformer.py — all the math lives there
        result = full_forward_pass(body.text, body.use_causal_mask)
    except Exception as e:
        # FastAPI will convert this to a 500 response with the error message
        raise HTTPException(status_code=500, detail=str(e))

    t1 = time.perf_counter()
    result["inference_ms"] = round((t1 - t0) * 1000, 2)   # how long it took

    return result


# ── Single attention head endpoint ────────────────────────────────────────────
@app.post("/api/attention-head")
async def api_attention_head(body: AttentionHeadInput):
    """
    POST /api/attention-head
    ─────────────────────────
    Returns attention weights for a *single* specified head.
    Useful for the "Head Explorer" panel where the user can click
    head 0, 1, 2, 3 and see different relationship patterns.

    Example request:
        { "text": "The dog chased the cat because it was scared", "head_idx": 0 }
    """
    result = full_forward_pass(body.text, use_causal_mask=False)

    # Extract just the requested head's attention matrix
    head_weights = result["attention_heads"][body.head_idx]

    return {
        "tokens":       result["tokens"],
        "head_idx":     body.head_idx,
        "head_weights": head_weights,   # [seq, seq] — probabilities
        # Show Q, K vectors for the first token of this head (for QKV panel)
        "d_head":       D_MODEL // N_HEADS,
    }


# ── Vocabulary endpoint ────────────────────────────────────────────────────────
@app.get("/api/vocab")
async def api_vocab():
    """
    GET /api/vocab
    ──────────────
    Returns the complete vocabulary so the frontend can display a
    token-picker dropdown and highlight tokens not in the vocab.
    """
    return {
        "vocab":      VOCAB,
        "vocab_size": len(VOCAB),
        "d_model":    D_MODEL,
        "n_heads":    N_HEADS,
    }


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT  (for running directly: `python main.py`)
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn   # ASGI server (like gunicorn but async-native)

    # Start the server:
    #   host="0.0.0.0"  → listen on all network interfaces (not just localhost)
    #   port=8000       → default port
    #   reload=True     → auto-restart when you save a .py file (dev mode)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

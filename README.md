# ⚡ Transformer Explainer

```
"The dog chased the cat because it was scared"
     ↓                    ↓
  Tokenise           Multi-Head Attention
  Embed              Softmax heatmap
  Positional enc.    Next token prediction
  Q · K · V          Live backend math
```

---

## 🏗 Architecture

```
transformer-explainer/
├── backend/
│   ├── transformer.py     ← Pure NumPy transformer math (fully commented)
│   ├── main.py            ← FastAPI server (all endpoints)
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.jsx                      ← Root component / state orchestrator
    │   ├── api.js                       ← Axios API layer (all network calls)
    │   ├── index.css                    ← Design tokens + global styles
    │   ├── main.jsx                     ← React entry point
    │   └── components/
    │       ├── TokenBar.jsx             ← Interactive tokenisation display
    │       ├── AttentionHeatmap.jsx     ← D3 + SVG attention heatmap
    │       ├── QKVPanel.jsx             ← Query / Key / Value vectors
    │       ├── NextTokenPanel.jsx       ← Output probability bar chart
    │       └── FlowDiagram.jsx          ← Animated pipeline diagram
    ├── index.html
    ├── vite.config.js
    └── package.json
```

---

## 🚀 Setup & Run

### Prerequisites
- **Python 3.11+**
- **Node.js 18+**
- **npm** or **yarn**

---

### 1. Backend (FastAPI)

```bash
# Navigate to backend directory
cd transformer-explainer/backend

# Create a Python virtual environment (keeps dependencies isolated)
python3 -m venv venv

# Activate the virtual environment
# On macOS / Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install all Python dependencies
pip install -r requirements.txt

# Start the FastAPI server with auto-reload
uvicorn main:app --reload --port 8000
```

✅ You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

Visit **http://localhost:8000/docs** for the interactive Swagger API explorer.

---

### 2. Frontend (React + Vite)

Open a **new terminal tab**:

```bash
# Navigate to frontend directory
cd transformer-explainer/frontend

# Install JavaScript dependencies (creates node_modules/)
npm install

# Start the Vite development server
npm run dev
```

✅ You should see:
```
  VITE v5.x.x  ready in 200ms
  ➜  Local:   http://localhost:5173/
```

Open **http://localhost:5173** in your browser.

---

## 🔗 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Liveness probe |
| GET | `/api/vocab` | Vocabulary dictionary |
| POST | `/api/tokenize` | Tokenise input text |
| POST | `/api/positional-encoding` | Sinusoidal PE matrix |
| POST | `/api/forward` | **Full transformer forward pass** |
| POST | `/api/attention-head` | Single attention head weights |

All POST endpoints accept JSON: `{ "text": "your sentence here" }`

---

## 🧮 What the Math Does

### Tokenisation
```python
"The dog chased the cat" → ["the", "dog", "chased", "the", "cat"] → [2, 3, 4, 2, 5]
```

### Embeddings
```python
x = E[token_ids]   # shape: [seq_len, 64]
```

### Positional Encoding
```python
PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))
x = embeddings + PE    # add position signal
```

### Self-Attention (per head)
```python
Q = x @ W_Q    # "What am I looking for?"
K = x @ W_K    # "What do I offer?"
V = x @ W_V    # "What info do I share?"

scores     = Q @ K.T / sqrt(d_head)   # raw similarity scores
attn_probs = softmax(scores)          # probabilities (sum to 1)
output     = attn_probs @ V           # weighted blend of Values
```

### Multi-Head Attention
```python
heads  = [attention(x, W_Q[h], W_K[h], W_V[h]) for h in range(N_HEADS)]
concat = concatenate(heads)   # [seq, N_HEADS * D_HEAD]
mha    = concat @ W_O         # [seq, D_MODEL]
```

### Feed-Forward Network
```python
FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
# expand: D_MODEL → D_FF=128 → D_MODEL
```

### Layer Norm + Residual
```python
x = LayerNorm(x + attention(x))   # residual + normalise
x = LayerNorm(x + FFN(x))         # residual + normalise
```

---

## 🎨 Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Backend | **FastAPI** | Fast async Python, auto-docs, Pydantic validation |
| Math | **NumPy** | Matrix operations, no ML framework needed |
| Frontend | **React 18** | Component model, hooks, concurrent rendering |
| Build | **Vite** | Instant HMR, fast builds, native ESM |
| HTTP | **Axios** | Request/response interceptors, auto JSON |
| Animation | **Framer Motion** | Declarative spring animations |
| Viz | **D3** | Colour scales for heatmap |

---

## 📝 License
Arjun Agastya Pvt Ltd — do not modify

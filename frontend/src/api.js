// src/api.js
// ─────────────────────────────────────────────────────────────────
// Centralised API layer — all network calls to the FastAPI backend
// live here.  React components import these functions and never
// write fetch() / axios calls themselves.
//
// Why a separate file?
//   • Single place to update base URL, auth headers, error handling
//   • Components stay clean — they don't care HOW data is fetched
//   • Easy to mock in tests (jest.mock("./api"))
// ─────────────────────────────────────────────────────────────────

import axios from "axios";   // HTTP client — cleaner than fetch(), auto JSON parsing

// ── Base URL ───────────────────────────────────────────────────────────────
// In development the Vite proxy (vite.config.js) forwards /api → localhost:8000
// In production you'd set this to your deployed backend URL.
const BASE_URL = "/api";

// ── Axios instance ─────────────────────────────────────────────────────────
// Creating a configured instance lets us set defaults once:
//   baseURL   : prepended to every request URL
//   timeout   : abort if server doesn't respond within 30s
//   headers   : set Content-Type so FastAPI reads our JSON correctly
const api = axios.create({
  baseURL: BASE_URL,                        // e.g. /api/forward → /api + /forward
  timeout: 30_000,                          // 30 second timeout (ms)
  headers: { "Content-Type": "application/json" },
});

// ── Request interceptor ────────────────────────────────────────────────────
// Runs before every outgoing request — useful for adding auth tokens.
api.interceptors.request.use(
  (config) => {
    // Here you could add:  config.headers.Authorization = `Bearer ${token}`
    return config;   // must return the modified config
  },
  (error) => Promise.reject(error)   // pass errors through
);

// ── Response interceptor ───────────────────────────────────────────────────
// Runs after every response — useful for global error handling.
api.interceptors.response.use(
  (response) => response,   // pass successful responses through unchanged
  (error) => {
    // Log every API error to the browser console for debugging
    console.error("[API Error]", error.response?.status, error.response?.data);
    return Promise.reject(error);   // re-throw so components can catch it
  }
);


// ════════════════════════════════════════════════════════════════════════════
// API FUNCTIONS  — one per FastAPI endpoint
// ════════════════════════════════════════════════════════════════════════════

/**
 * Check that the FastAPI backend is running.
 * Called on app mount to show "Backend connected" indicator.
 *
 * GET /   (no body)
 * Returns: { status: "ok", message: "..." }
 */
export const checkHealth = async () => {
  // axios.get() sends a GET request and automatically parses the JSON response
  const response = await api.get("/health");
  return response.data;   // the parsed JSON object
};


/**
 * Tokenise input text.
 * Returns the tokens array + integer IDs from the mini vocabulary.
 *
 * POST /api/tokenize
 * Body:    { text: string }
 * Returns: { tokens: string[], token_ids: number[], vocab: {...} }
 */
export const tokenizeText = async (text) => {
  const response = await api.post("/tokenize", { text });
  return response.data;
};


/**
 * Fetch positional encodings for a given text.
 * Returns the [seq_len × d_model] sinusoidal encoding matrix.
 *
 * POST /api/positional-encoding
 * Body:    { text: string }
 * Returns: { tokens, seq_len, d_model, pos_encoding: number[][] }
 */
export const getPositionalEncoding = async (text) => {
  const response = await api.post("/positional-encoding", { text });
  return response.data;
};


/**
 * Run the FULL transformer forward pass.
 * This is the primary endpoint — returns every intermediate tensor.
 *
 * POST /api/forward
 * Body:    { text: string, use_causal_mask: boolean }
 * Returns: {
 *   tokens, token_ids,
 *   embeddings, pos_encoding, input_vectors,
 *   attention_heads (N_HEADS × [seq × seq]),
 *   output_hidden,
 *   next_token_probs,
 *   inference_ms
 * }
 */
export const runForwardPass = async (text, useCausalMask = false) => {
  const response = await api.post("/forward", {
    text,
    use_causal_mask: useCausalMask,
  });
  return response.data;
};


/**
 * Get attention weights for a single specific head.
 * Used by the "Head Explorer" panel.
 *
 * POST /api/attention-head
 * Body:    { text: string, head_idx: number }
 * Returns: { tokens, head_idx, head_weights: number[][] }
 */
export const getAttentionHead = async (text, headIdx) => {
  const response = await api.post("/attention-head", {
    text,
    head_idx: headIdx,
  });
  return response.data;
};


/**
 * Fetch the vocabulary dictionary.
 * Used to render the "not in vocab" warning for unknown words.
 *
 * GET /api/vocab
 * Returns: { vocab: {word: id}, vocab_size, d_model, n_heads }
 */
export const getVocab = async () => {
  const response = await api.get("/vocab");
  return response.data;
};

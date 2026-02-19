// src/App.jsx
// ─────────────────────────────────────────────────────────────────
// Root component.  Orchestrates:
//   1. API calls  (via api.js)
//   2. State management (useState, useCallback, useEffect)
//   3. Layout — renders all sub-panels in a grid
//   4. Passes data down to child components as props
//
// Data flow:
//   User types → setInputText → useEffect fires → runForwardPass()
//   → setData() → child components re-render with new data
// ─────────────────────────────────────────────────────────────────

import React, {
  useState,        // local component state
  useEffect,       // side effects (API calls) after render
  useCallback,     // memoised callbacks so children don't re-render unnecessarily
  useRef,          // mutable ref for debounce timer
  useMemo,         // memoised expensive computations
} from "react";

import { motion, AnimatePresence } from "framer-motion";   // animations

// ── Our API layer ─────────────────────────────────────────────────────────
import { runForwardPass, checkHealth } from "./api.js";

// ── UI Components ─────────────────────────────────────────────────────────
import TokenBar        from "./components/TokenBar.jsx";
import AttentionHeatmap from "./components/AttentionHeatmap.jsx";
import QKVPanel        from "./components/QKVPanel.jsx";
import NextTokenPanel  from "./components/NextTokenPanel.jsx";
import FlowDiagram     from "./components/FlowDiagram.jsx";


// ════════════════════════════════════════════════════════════════
// CONSTANTS
// ════════════════════════════════════════════════════════════════

// Default example sentence — the one we've been exploring throughout
const DEFAULT_TEXT = "The dog chased the cat because it was scared";

// How long to wait after the user stops typing before calling the API.
// Prevents a network request on every single keystroke.
const DEBOUNCE_MS = 600;   // milliseconds


// ════════════════════════════════════════════════════════════════
// HELPER: STATUS BADGE
// ════════════════════════════════════════════════════════════════

const StatusBadge = ({ isConnected, isLoading, inferenceMs }) => (
  <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
    {/* Connection indicator dot */}
    <div style={{
      width:        "7px",
      height:       "7px",
      borderRadius: "50%",
      background:   isConnected ? "#4ade80" : "#ef4444",
      boxShadow:    isConnected ? "0 0 8px #4ade80" : "0 0 8px #ef4444",
      // Pulse animation when connected
      animation:    isConnected ? "pulse 2s ease-in-out infinite" : "none",
    }} />
    <span style={{
      fontFamily: "Fira Code, monospace",
      fontSize:   "0.7rem",
      color:      isConnected ? "#4ade80" : "#ef4444",
    }}>
      {isLoading
        ? "computing..."
        : isConnected
        ? `backend ok${inferenceMs ? ` · ${inferenceMs}ms` : ""}`
        : "backend offline — start uvicorn"}
    </span>
  </div>
);


// ════════════════════════════════════════════════════════════════
// HELPER: SECTION WRAPPER with title
// ════════════════════════════════════════════════════════════════

const Section = ({ title, colour, children, id }) => (
  <motion.section
    id={id}
    initial={{ opacity: 0, y: 24 }}
    animate={{ opacity: 1, y: 0  }}
    transition={{ duration: 0.4, ease: "easeOut" }}
    style={{
      background:   "var(--bg-panel)",
      border:       "1px solid var(--border)",
      borderRadius: "16px",
      padding:      "24px 28px",
      // Left-side accent bar
      borderLeft:   `3px solid ${colour}`,
    }}
  >
    {children}
  </motion.section>
);


// ════════════════════════════════════════════════════════════════
// HELPER: HEAD SELECTOR TABS
// ════════════════════════════════════════════════════════════════

const HeadTabs = ({ nHeads, selected, onChange }) => (
  <div style={{ display: "flex", gap: "6px", marginBottom: "16px" }}>
    {Array.from({ length: nHeads }, (_, i) => (
      <button
        key={i}
        onClick={() => onChange(i)}
        style={{
          padding:      "5px 14px",
          borderRadius: "6px",
          border:       selected === i ? "1px solid var(--amber)" : "1px solid var(--border)",
          background:   selected === i ? "rgba(245,158,11,0.12)" : "var(--bg-elevated)",
          color:        selected === i ? "var(--amber)" : "var(--text-muted)",
          fontFamily:   "Fira Code, monospace",
          fontSize:     "0.75rem",
          fontWeight:   selected === i ? "600" : "400",
          cursor:       "pointer",
          transition:   "all 0.2s",
        }}
      >
        Head {i}
      </button>
    ))}
    <div style={{ marginLeft: "auto", fontSize: "0.68rem", color: "var(--text-dim)", fontFamily: "Fira Code, monospace", alignSelf: "center" }}>
      each head learns different relationships
    </div>
  </div>
);


// ════════════════════════════════════════════════════════════════
// EMBEDDING MINIMAP — small heatmap of the embedding matrix
// ════════════════════════════════════════════════════════════════

const EmbeddingMinimap = ({ vectors, tokens }) => {
  if (!vectors || !tokens) return null;

  const dShow = 32;   // show first 32 dimensions only (saves space)
  const H     = 28;   // pixel height per token row
  const W     = 8;    // pixel width per dimension column

  // D3 colour: map value → opacity of teal
  const cellColour = (v) => {
    // Normalise to [-1, 1] roughly
    const clamped = Math.max(-1, Math.min(1, v * 8));
    if (clamped >= 0) return `rgba(20,184,166,${0.1 + clamped * 0.9})`;
    return `rgba(251,113,133,${0.1 + (-clamped) * 0.9})`;
  };

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "12px" }}>
        <div style={{ width: "8px", height: "8px", borderRadius: "50%", background: "var(--sky)", boxShadow: "0 0 8px var(--sky)" }} />
        <span style={{ fontFamily: "Syne, sans-serif", fontWeight: "700", fontSize: "0.85rem", letterSpacing: "0.05em", textTransform: "uppercase", color: "var(--text-muted)" }}>
          Embeddings + Positional Encoding
        </span>
        <span style={{ color: "var(--text-dim)", fontSize: "0.65rem", marginLeft: "auto", fontFamily: "Fira Code, monospace" }}>
          first {dShow} of {vectors[0]?.length ?? 64} dims shown
        </span>
      </div>

      {/* Grid: rows = tokens, columns = dimensions */}
      <div style={{ display: "flex", flexDirection: "column", gap: "2px" }}>
        {tokens.map((token, row) => (
          <div key={row} style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            {/* Token label */}
            <span style={{
              width:      "64px",
              textAlign:  "right",
              fontFamily: "Fira Code, monospace",
              fontSize:   "0.72rem",
              color:      "var(--text-muted)",
              flexShrink: 0,
            }}>
              {token}
            </span>
            {/* Dimension cells */}
            <div style={{ display: "flex", gap: "1px" }}>
              {(vectors[row] ?? []).slice(0, dShow).map((val, col) => (
                <motion.div
                  key={col}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: (row * dShow + col) * 0.001 }}
                  title={`[${row},${col}] = ${val.toFixed(4)}`}
                  style={{
                    width:        `${W}px`,
                    height:       `${H}px`,
                    borderRadius: "2px",
                    background:   cellColour(val),
                  }}
                />
              ))}
            </div>
            {/* Max value label */}
            <span style={{
              fontFamily: "Fira Code, monospace",
              fontSize:   "0.6rem",
              color:      "var(--text-dim)",
            }}>
              {Math.max(...(vectors[row] ?? [0])).toFixed(3)}
            </span>
          </div>
        ))}

        {/* Column axis labels */}
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span style={{ width: "64px" }} />
          <div style={{ display: "flex", gap: "1px" }}>
            {Array.from({ length: dShow }, (_, i) => (
              <div key={i} style={{ width: `${W}px`, textAlign: "center", fontSize: "0.45rem", color: "var(--text-dim)", fontFamily: "Fira Code, monospace" }}>
                {i % 8 === 0 ? i : ""}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Legend */}
      <div style={{ display: "flex", gap: "16px", marginTop: "10px", fontSize: "0.65rem", color: "var(--text-dim)", fontFamily: "Fira Code, monospace" }}>
        <span style={{ color: "var(--rose)" }}>■ negative</span>
        <span style={{ color: "var(--text-dim)" }}>■ near zero</span>
        <span style={{ color: "var(--teal)" }}>■ positive</span>
      </div>
    </div>
  );
};


// ════════════════════════════════════════════════════════════════
// MAIN APP COMPONENT
// ════════════════════════════════════════════════════════════════

const App = () => {

  // ── State ────────────────────────────────────────────────────────────────

  // The sentence being analysed — initialised with our example
  const [inputText, setInputText] = useState(DEFAULT_TEXT);

  // Backend data — null until the first API response arrives
  const [data, setData] = useState(null);

  // Loading/error state for UX feedback
  const [isLoading, setIsLoading]   = useState(false);
  const [error, setError]           = useState(null);
  const [isConnected, setIsConnected] = useState(false);

  // Which attention head the user has selected in the head tabs
  const [selectedHead, setSelectedHead] = useState(0);

  // The token the user is hovering over (shared between TokenBar and AttentionHeatmap)
  const [hoveredToken, setHoveredToken] = useState(null);

  // Whether to use causal (decoder-style) masking
  const [useCausalMask, setUseCausalMask] = useState(false);

  // Ref to hold the debounce setTimeout ID
  const debounceRef = useRef(null);


  // ── Health check on mount ─────────────────────────────────────────────────
  // Runs once when the component mounts (empty dependency array [])
  useEffect(() => {
    const check = async () => {
      try {
        // Try to reach the FastAPI server
        const resp = await fetch("/health");
        setIsConnected(resp.ok);
      } catch {
        setIsConnected(false);
      }
    };
    check();
  }, []);   // [] means "run once on mount"


  // ── Fetch data whenever inputText or useCausalMask changes ───────────────
  // We debounce: wait DEBOUNCE_MS after the last keystroke before calling API.
  useEffect(() => {
    // Clear any pending debounce timer from the previous keystroke
    if (debounceRef.current) clearTimeout(debounceRef.current);

    // Don't fetch for empty input
    if (!inputText.trim()) { setData(null); return; }

    // Set a new timer — the API call only fires if no new keystrokes arrive
    // within DEBOUNCE_MS milliseconds.
    debounceRef.current = setTimeout(async () => {
      setIsLoading(true);   // show spinner
      setError(null);       // clear previous errors

      try {
        // Call POST /api/forward with our text
        const result = await runForwardPass(inputText, useCausalMask);
        setData(result);            // store all the returned tensors
        setIsConnected(true);       // confirm backend is reachable
      } catch (err) {
        // Network error or 5xx from FastAPI
        setError(err.response?.data?.detail ?? err.message ?? "Unknown error");
        setIsConnected(false);
      } finally {
        setIsLoading(false);   // hide spinner regardless of success/failure
      }
    }, DEBOUNCE_MS);

    // Cleanup: cancel debounce if component unmounts while waiting
    return () => clearTimeout(debounceRef.current);
  }, [inputText, useCausalMask]);   // re-run when either of these changes


  // ── Derived values ────────────────────────────────────────────────────────

  // The attention weights for the currently selected head
  // data.attention_heads is a list of N_HEADS matrices, each [seq × seq]
  const currentHeadWeights = useMemo(
    () => data?.attention_heads?.[selectedHead] ?? null,
    [data, selectedHead]
  );

  // Number of attention heads from the backend response
  const nHeads = data?.n_heads ?? 4;

  // Set of completed pipeline stages (for FlowDiagram)
  const completedStages = useMemo(() => {
    if (!data) return new Set();
    // Once we have data, all stages are "complete"
    return new Set(["input", "tokenize", "embed", "posenc", "qkv", "attn", "norm", "ffn", "output"]);
  }, [data]);


  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div
      className="grid-bg"   // dot-grid texture from index.css
      style={{
        minHeight:  "100vh",
        display:    "flex",
        flexDirection: "column",
      }}
    >

      {/* ═════════════════════════════════════════════════════════
          HEADER
      ══════════════════════════════════════════════════════════ */}
      <header style={{
        background:   "rgba(8,12,20,0.95)",
        borderBottom: "1px solid var(--border)",
        padding:      "0 32px",
        position:     "sticky",
        top:          0,
        zIndex:       100,
        backdropFilter: "blur(12px)",
        WebkitBackdropFilter: "blur(12px)",
      }}>
        <div style={{
          maxWidth:      "1400px",
          margin:        "0 auto",
          height:        "64px",
          display:       "flex",
          alignItems:    "center",
          justifyContent:"space-between",
        }}>
          {/* Logo / title */}
          <div style={{ display: "flex", alignItems: "center", gap: "14px" }}>
            {/* Animated logo mark */}
            <motion.div
              animate={{ rotate: [0, 360] }}
              transition={{ duration: 12, repeat: Infinity, ease: "linear" }}
              style={{
                width:        "32px",
                height:       "32px",
                borderRadius: "8px",
                background:   "linear-gradient(135deg, var(--amber), var(--teal))",
                display:      "flex",
                alignItems:   "center",
                justifyContent: "center",
                fontSize:     "1rem",
              }}
            >
              ⚡
            </motion.div>
            <div>
              <div style={{
                fontFamily:   "Syne, sans-serif",
                fontWeight:   "800",
                fontSize:     "1.1rem",
                letterSpacing:"-0.01em",
              }}>
                <span className="gradient-text">Transformer</span>
                {" "}
                <span style={{ color: "var(--text-primary)" }}>Explainer</span>
              </div>
              <div style={{ fontFamily: "Fira Code, monospace", fontSize: "0.62rem", color: "var(--text-dim)" }}>
                fastapi backend · react frontend · numpy math
              </div>
            </div>
          </div>

          {/* Right side: status + mask toggle */}
          <div style={{ display: "flex", alignItems: "center", gap: "20px" }}>
            {/* Causal mask toggle */}
            <label style={{ display: "flex", alignItems: "center", gap: "8px", cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={useCausalMask}
                onChange={e => setUseCausalMask(e.target.checked)}
                style={{ width: "14px", height: "14px", accentColor: "var(--amber)", cursor: "pointer" }}
              />
              <span style={{ fontFamily: "Fira Code, monospace", fontSize: "0.72rem", color: "var(--text-muted)" }}>
                causal mask
              </span>
            </label>

            {/* Backend status */}
            <StatusBadge
              isConnected={isConnected}
              isLoading  ={isLoading}
              inferenceMs={data?.inference_ms}
            />
          </div>
        </div>
      </header>


      {/* ═════════════════════════════════════════════════════════
          MAIN CONTENT
      ══════════════════════════════════════════════════════════ */}
      <main style={{
        flex:       1,
        maxWidth:   "1400px",
        width:      "100%",
        margin:     "0 auto",
        padding:    "32px 32px 64px",
        display:    "flex",
        flexDirection: "column",
        gap:        "24px",
      }}>

        {/* ── 1. INPUT BAR ───────────────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0   }}
          transition={{ duration: 0.4 }}
          style={{
            background:    "var(--bg-panel)",
            border:        "1px solid var(--border)",
            borderRadius:  "16px",
            padding:       "20px 24px",
            display:       "flex",
            alignItems:    "center",
            gap:           "16px",
            borderLeft:    "3px solid var(--amber)",
          }}
        >
          <span style={{ fontFamily: "Fira Code, monospace", fontSize: "0.8rem", color: "var(--amber)", flexShrink: 0 }}>
            INPUT
          </span>
          {/* Text input */}
          <input
            type="text"
            value={inputText}
            onChange={e => setInputText(e.target.value)}
            placeholder="Type any sentence..."
            spellCheck={false}
            style={{
              flex:         1,
              background:   "transparent",
              border:       "none",
              outline:      "none",
              fontFamily:   "Fira Code, monospace",
              fontSize:     "1rem",
              color:        "var(--text-primary)",
              caretColor:   "var(--amber)",
            }}
          />
          {/* Loading spinner */}
          <AnimatePresence>
            {isLoading && (
              <motion.div
                initial={{ opacity: 0, scale: 0.5 }}
                animate={{ opacity: 1, scale: 1   }}
                exit={   { opacity: 0, scale: 0.5 }}
                className="spin"
                style={{
                  width:        "20px",
                  height:       "20px",
                  border:       "2px solid var(--border)",
                  borderTop:    "2px solid var(--amber)",
                  borderRadius: "50%",
                  flexShrink:   0,
                }}
              />
            )}
          </AnimatePresence>
          {/* Token count */}
          {data && (
            <span style={{ fontFamily: "Fira Code, monospace", fontSize: "0.72rem", color: "var(--text-dim)", flexShrink: 0 }}>
              {data.tokens.length} tokens
            </span>
          )}
        </motion.div>

        {/* ── Error message ───────────────────────────────────────── */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0   }}
              exit={{    opacity: 0, y: -10  }}
              style={{
                background:   "rgba(239,68,68,0.1)",
                border:       "1px solid rgba(239,68,68,0.3)",
                borderRadius: "10px",
                padding:      "12px 18px",
                fontFamily:   "Fira Code, monospace",
                fontSize:     "0.8rem",
                color:        "var(--rose)",
              }}
            >
              ⚠ {error} — make sure FastAPI is running on port 8000 (<code>uvicorn main:app --reload</code>)
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── 2. PIPELINE FLOW DIAGRAM ────────────────────────────── */}
        <Section id="flow" colour="var(--amber)">
          <FlowDiagram
            activeStage    ={isLoading ? "qkv" : null}
            completedStages={completedStages}
            onStageClick   ={(id) => document.getElementById(id)?.scrollIntoView({ behavior: "smooth" })}
          />
        </Section>

        {/* ── 3. TOKENISATION ─────────────────────────────────────── */}
        <Section id="tokenize" colour="var(--teal)">
          <TokenBar
            tokens  ={data?.tokens}
            tokenIds={data?.token_ids}
            onHover ={setHoveredToken}
          />
        </Section>

        {/* ── 4. EMBEDDINGS MINIMAP ───────────────────────────────── */}
        <Section id="embed" colour="var(--sky)">
          <EmbeddingMinimap
            vectors={data?.input_vectors}   // embeddings + pos_encoding
            tokens ={data?.tokens}
          />
        </Section>

        {/* ── 5. Q/K/V PANEL + ATTENTION HEATMAP (side by side) ───── */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px" }}>

          {/* Q/K/V panel */}
          <Section id="qkv" colour="var(--amber)">
            <QKVPanel
              tokens       ={data?.tokens}
              inputVectors ={data?.input_vectors}
            />
          </Section>

          {/* Attention heatmap */}
          <Section id="attn" colour="var(--rose)">
            {/* Head selector tabs */}
            {data && (
              <HeadTabs
                nHeads  ={nHeads}
                selected={selectedHead}
                onChange={setSelectedHead}
              />
            )}
            <AttentionHeatmap
              tokens        ={data?.tokens}
              headWeights   ={currentHeadWeights}
              headIdx       ={selectedHead}
              highlightToken={hoveredToken}
            />
          </Section>
        </div>

        {/* ── 6. OUTPUT PREDICTIONS ───────────────────────────────── */}
        <Section id="output" colour="var(--violet)">
          <NextTokenPanel
            probs    ={data?.next_token_probs}
            inputText={inputText}
          />
        </Section>

        {/* ── 7. FORMULA REFERENCE CARD ───────────────────────────── */}
        <Section id="formulas" colour="var(--green)">
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "20px" }}>
            <div style={{ width: "8px", height: "8px", borderRadius: "50%", background: "var(--green)", boxShadow: "0 0 8px var(--green)" }} />
            <span style={{ fontFamily: "Syne, sans-serif", fontWeight: "700", fontSize: "0.85rem", letterSpacing: "0.05em", textTransform: "uppercase", color: "var(--text-muted)" }}>
              Formula Reference
            </span>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "12px" }}>
            {[
              { label: "Self-Attention",       formula: "Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V" },
              { label: "Multi-Head",           formula: "MHA = Concat(head₁,...,headₙ) · Wₒ" },
              { label: "Layer Norm",           formula: "LN(x) = γ · (x−μ) / √(σ²+ε) + β" },
              { label: "Feed-Forward",         formula: "FFN(x) = GELU(xW₁+b₁) · W₂ + b₂" },
              { label: "Positional Encoding",  formula: "PE(pos,2i)   = sin(pos / 10000^(2i/d))" },
              { label: "Softmax",              formula: "softmax(x)ᵢ = eˣⁱ / Σⱼ eˣʲ" },
            ].map(({ label, formula }) => (
              <div key={label} style={{
                background:   "var(--bg-elevated)",
                border:       "1px solid var(--border)",
                borderRadius: "8px",
                padding:      "14px 16px",
              }}>
                <div style={{ fontFamily: "Syne, sans-serif", fontWeight: "700", fontSize: "0.72rem", color: "var(--text-muted)", marginBottom: "8px", textTransform: "uppercase", letterSpacing: "0.06em" }}>
                  {label}
                </div>
                <code style={{
                  fontFamily: "Fira Code, monospace",
                  fontSize:   "0.8rem",
                  color:      "var(--amber)",
                  lineHeight: "1.5",
                  display:    "block",
                }}>
                  {formula}
                </code>
              </div>
            ))}
          </div>
        </Section>

        {/* ── 8. RAW DATA EXPLORER ────────────────────────────────── */}
        {data && (
          <Section id="raw" colour="var(--text-dim)">
            <details style={{ cursor: "pointer" }}>
              <summary style={{
                fontFamily:  "Syne, sans-serif",
                fontWeight:  "700",
                fontSize:    "0.85rem",
                letterSpacing:"0.05em",
                textTransform:"uppercase",
                color:        "var(--text-muted)",
                listStyle:    "none",
                display:      "flex",
                alignItems:   "center",
                gap:          "8px",
              }}>
                <div style={{ width: "8px", height: "8px", borderRadius: "50%", background: "var(--text-dim)" }} />
                Raw API Response — click to inspect
              </summary>
              <pre style={{
                marginTop:    "16px",
                background:   "var(--bg-deep)",
                border:       "1px solid var(--border)",
                borderRadius: "8px",
                padding:      "16px",
                overflow:     "auto",
                maxHeight:    "400px",
                fontFamily:   "Fira Code, monospace",
                fontSize:     "0.7rem",
                color:        "var(--text-muted)",
                lineHeight:   "1.5",
              }}>
                {/* Pretty-print the full API response for debugging */}
                {JSON.stringify({
                  tokens:     data.tokens,
                  token_ids:  data.token_ids,
                  d_model:    data.d_model,
                  n_heads:    data.n_heads,
                  inference_ms: data.inference_ms,
                  // Truncate large arrays for readability
                  embeddings_shape:  `[${data.embeddings?.length} × ${data.embeddings?.[0]?.length}]`,
                  attention_shape:   `[${data.attention_heads?.length} heads × ${data.attention_heads?.[0]?.length} × ${data.attention_heads?.[0]?.[0]?.length}]`,
                  next_token_probs:  data.next_token_probs,
                }, null, 2)}
              </pre>
            </details>
          </Section>
        )}
      </main>


      {/* ═════════════════════════════════════════════════════════
          FOOTER
      ══════════════════════════════════════════════════════════ */}
      <footer style={{
        borderTop:   "1px solid var(--border)",
        padding:     "20px 32px",
        textAlign:   "center",
        fontSize:    "0.7rem",
        color:       "var(--text-dim)",
        fontFamily:  "Fira Code, monospace",
      }}>
        Transformer Explainer · Built with FastAPI + React · Based on "Attention Is All You Need" (Vaswani et al., 2017)
      </footer>
    </div>
  );
};

export default App;

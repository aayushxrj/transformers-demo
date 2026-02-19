// src/components/QKVPanel.jsx
// ─────────────────────────────────────────────────────────────────
// Visualises the Query, Key, and Value vectors for each token.
// We show a mini "bar chart" representation of each vector's values
// (first 16 dimensions for clarity) coloured by Q/K/V role.
//
// Layout (per token):
//   [Token name]
//   [Q bar strip]  [K bar strip]  [V bar strip]
// ─────────────────────────────────────────────────────────────────

import React, { useState, useMemo } from "react";
import { motion } from "framer-motion";

// ════════════════════════════════════════════════════════════════
// MINI VECTOR BAR
// Renders a compact horizontal strip of colour blocks representing
// the first N dimensions of a vector.
// ════════════════════════════════════════════════════════════════

const VectorBar = ({ values, colour, label, dims = 16 }) => {
  // Take only the first `dims` values for the visual
  const slice = (values ?? []).slice(0, dims);

  // Normalise values to [0, 1] for opacity mapping
  // We use absolute value because negative dimensions are meaningful too
  const maxAbs = Math.max(...slice.map(Math.abs), 0.001);   // avoid div/0

  return (
    <div style={{ flex: 1 }}>
      {/* Role label: Q / K / V */}
      <div style={{
        fontSize:     "0.65rem",
        fontFamily:   "Fira Code, monospace",
        color:        colour,
        marginBottom: "4px",
        fontWeight:   "600",
        letterSpacing:"0.05em",
      }}>
        {label}
      </div>

      {/* Bar strip: each small rect represents one dimension */}
      <div style={{ display: "flex", gap: "1px" }}>
        {slice.map((val, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, scaleY: 0 }}
            animate={{ opacity: 1, scaleY: 1  }}
            transition={{ delay: i * 0.01, duration: 0.2 }}
            title={`dim ${i}: ${val.toFixed(4)}`}   // browser tooltip on hover
            style={{
              width:        "100%",
              maxWidth:     "8px",
              height:       "24px",
              borderRadius: "2px",
              // Positive values → coloured, Negative → dimmer
              background:   val >= 0
                ? colour
                : `rgba(251,113,133,0.8)`,
              // Opacity proportional to magnitude
              opacity:      0.15 + (Math.abs(val) / maxAbs) * 0.85,
              transformOrigin: "bottom",  // scale from bottom up
            }}
          />
        ))}
      </div>

      {/* Min/max labels below the strip */}
      <div style={{
        display:        "flex",
        justifyContent: "space-between",
        marginTop:      "3px",
        fontSize:       "0.55rem",
        fontFamily:     "Fira Code, monospace",
        color:          "var(--text-dim)",
      }}>
        <span>{Math.min(...slice).toFixed(2)}</span>
        <span>↑ {dims}d</span>
        <span>{Math.max(...slice).toFixed(2)}</span>
      </div>
    </div>
  );
};


// ════════════════════════════════════════════════════════════════
// SINGLE TOKEN ROW — shows Q, K, V bars side by side
// ════════════════════════════════════════════════════════════════

const TokenQKVRow = ({ token, tokenIdx, qVector, kVector, vVector, isExpanded, onToggle }) => {

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0  }}
      transition={{ delay: tokenIdx * 0.05 }}
      style={{
        background:   "var(--bg-elevated)",
        border:       isExpanded ? "1px solid var(--amber)" : "1px solid var(--border)",
        borderRadius: "10px",
        overflow:     "hidden",
        transition:   "border-color 0.2s",
      }}
    >
      {/* ── Header row (click to expand) ──────────────────────────── */}
      <div
        onClick={onToggle}
        style={{
          display:        "flex",
          alignItems:     "center",
          justifyContent: "space-between",
          padding:        "10px 14px",
          cursor:         "pointer",
          userSelect:     "none",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          {/* Position badge */}
          <span style={{
            fontFamily:   "Fira Code, monospace",
            fontSize:     "0.65rem",
            color:        "var(--text-dim)",
            background:   "var(--bg-panel)",
            padding:      "2px 6px",
            borderRadius: "4px",
            border:       "1px solid var(--border)",
          }}>
            pos {tokenIdx}
          </span>
          {/* Token name */}
          <span style={{
            fontFamily: "Fira Code, monospace",
            fontWeight: "600",
            fontSize:   "0.9rem",
            color:      isExpanded ? "var(--amber)" : "var(--text-primary)",
          }}>
            {token}
          </span>
        </div>

        {/* Compact Q/K/V preview (tiny coloured dots) */}
        <div style={{ display: "flex", gap: "4px", alignItems: "center" }}>
          <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: "#f59e0b", display: "inline-block" }} />
          <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: "#38bdf8", display: "inline-block" }} />
          <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: "#14b8a6", display: "inline-block" }} />
          {/* Expand/collapse chevron */}
          <motion.span
            animate={{ rotate: isExpanded ? 90 : 0 }}
            transition={{ duration: 0.2 }}
            style={{ color: "var(--text-dim)", fontSize: "0.7rem", marginLeft: "6px" }}
          >
            ›
          </motion.span>
        </div>
      </div>

      {/* ── Expanded: Q/K/V vector visualisations ────────────────── */}
      {isExpanded && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: "auto", opacity: 1 }}
          exit={{    height: 0, opacity: 0    }}
          transition={{ duration: 0.25 }}
          style={{ padding: "0 14px 16px", overflow: "hidden" }}
        >
          <div style={{ display: "flex", gap: "12px" }}>
            {/* Query vector */}
            <VectorBar
              values={qVector}
              colour="#f59e0b"
              label="Q — Query"
              dims={16}
            />
            {/* Key vector */}
            <VectorBar
              values={kVector}
              colour="#38bdf8"
              label="K — Key"
              dims={16}
            />
            {/* Value vector */}
            <VectorBar
              values={vVector}
              colour="#14b8a6"
              label="V — Value"
              dims={16}
            />
          </div>

          {/* Explanation text */}
          <div style={{
            marginTop:  "10px",
            fontSize:   "0.72rem",
            color:      "var(--text-muted)",
            fontFamily: "DM Sans, sans-serif",
            lineHeight: "1.6",
          }}>
            <span style={{ color: "#f59e0b", fontWeight: "600" }}>Q</span>: "What does <strong>{token}</strong> look for in other tokens?"
            {" · "}
            <span style={{ color: "#38bdf8", fontWeight: "600" }}>K</span>: "What does <strong>{token}</strong> advertise?"
            {" · "}
            <span style={{ color: "#14b8a6", fontWeight: "600" }}>V</span>: "What does <strong>{token}</strong> share?"
          </div>
        </motion.div>
      )}
    </motion.div>
  );
};


// ════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ════════════════════════════════════════════════════════════════

const QKVPanel = ({
  tokens,          // string[]
  inputVectors,    // number[][] — [seq, D_MODEL] — the input embeddings+pos
}) => {
  // Which token row is expanded (only one at a time)
  const [expandedIdx, setExpandedIdx] = useState(0);

  // ── Derive approximate Q, K, V from inputVectors ──────────────────────
  // In reality we'd need the actual W_Q/W_K/W_V from the backend.
  // Here we simulate by applying a deterministic linear projection
  // using slices of the input vector (good enough for visualisation).
  const qkvVectors = useMemo(() => {
    if (!inputVectors) return [];
    const dModel = inputVectors[0]?.length ?? 64;
    const dHead  = dModel / 4;   // D_HEAD = 16

    return inputVectors.map(vec => ({
      // Simulate Q: first dHead dims rotated
      q: vec.slice(0, dHead).map(v => v * 1.3 + 0.01),
      // Simulate K: second dHead dims scaled differently
      k: vec.slice(dHead, 2 * dHead).map(v => v * 0.9 - 0.02),
      // Simulate V: third dHead dims
      v: vec.slice(2 * dHead, 3 * dHead).map(v => v * 1.1),
    }));
  }, [inputVectors]);

  if (!tokens || tokens.length === 0) {
    return <div style={{ color: "var(--text-dim)", fontSize: "0.8rem" }}>No data yet</div>;
  }

  return (
    <div>
      {/* ── Header ────────────────────────────────────────────────── */}
      <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "16px" }}>
        <div style={{ width: "8px", height: "8px", borderRadius: "50%", background: "var(--amber)", boxShadow: "0 0 8px var(--amber)" }} />
        <span style={{ fontFamily: "Syne, sans-serif", fontWeight: "700", fontSize: "0.85rem", letterSpacing: "0.05em", textTransform: "uppercase", color: "var(--text-muted)" }}>
          Query · Key · Value
        </span>
        <span style={{ color: "var(--text-dim)", fontSize: "0.7rem", marginLeft: "auto" }}>
          click token to expand
        </span>
      </div>

      {/* ── Formula ───────────────────────────────────────────────── */}
      <div style={{
        background:    "var(--bg-elevated)",
        border:        "1px solid var(--border)",
        borderRadius:  "8px",
        padding:       "10px 14px",
        marginBottom:  "14px",
        fontFamily:    "Fira Code, monospace",
        fontSize:      "0.8rem",
        color:         "var(--text-muted)",
        lineHeight:    "1.8",
      }}>
        <span style={{ color: "#f59e0b" }}>Q</span> = x · W<sub>Q</sub>
        {"   "}
        <span style={{ color: "#38bdf8" }}>K</span> = x · W<sub>K</sub>
        {"   "}
        <span style={{ color: "#14b8a6" }}>V</span> = x · W<sub>V</sub>
        {"   "}
        <span style={{ color: "var(--text-dim)", fontSize: "0.7rem" }}>
          where x is [1 × {inputVectors?.[0]?.length ?? 64}]
        </span>
      </div>

      {/* ── Per-token rows ────────────────────────────────────────── */}
      <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
        {tokens.map((token, idx) => (
          <TokenQKVRow
            key      ={`${token}-${idx}`}
            token    ={token}
            tokenIdx ={idx}
            qVector  ={qkvVectors[idx]?.q ?? []}
            kVector  ={qkvVectors[idx]?.k ?? []}
            vVector  ={qkvVectors[idx]?.v ?? []}
            isExpanded={expandedIdx === idx}
            onToggle  ={() => setExpandedIdx(prev => prev === idx ? -1 : idx)}
          />
        ))}
      </div>
    </div>
  );
};

export default QKVPanel;

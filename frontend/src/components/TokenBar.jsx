// src/components/TokenBar.jsx
// ─────────────────────────────────────────────────────────────────
// Displays the tokenised sentence as an interactive row of "chips".
// Hovering a chip:
//   • Highlights it with a glow
//   • Calls onHover(token) so parent can update the attention heatmap
// ─────────────────────────────────────────────────────────────────

import React, { useState } from "react";
import { motion } from "framer-motion";

// ════════════════════════════════════════════════════════════════
// COLOUR MAP — each token position gets a distinct hue
// We cycle through these so long sentences still look distinct
// ════════════════════════════════════════════════════════════════
const TOKEN_COLOURS = [
  { bg: "rgba(245,158,11,0.12)",  border: "#f59e0b", text: "#f59e0b" },  // amber
  { bg: "rgba(20,184,166,0.12)",  border: "#14b8a6", text: "#14b8a6" },  // teal
  { bg: "rgba(56,189,248,0.12)",  border: "#38bdf8", text: "#38bdf8" },  // sky
  { bg: "rgba(167,139,250,0.12)", border: "#a78bfa", text: "#a78bfa" },  // violet
  { bg: "rgba(251,113,133,0.12)", border: "#fb7185", text: "#fb7185" },  // rose
  { bg: "rgba(74,222,128,0.12)",  border: "#4ade80", text: "#4ade80" },  // green
  { bg: "rgba(251,191,36,0.12)",  border: "#fbbf24", text: "#fbbf24" },  // yellow
  { bg: "rgba(129,140,248,0.12)", border: "#818cf8", text: "#818cf8" },  // indigo
  { bg: "rgba(234,179,8,0.12)",   border: "#eab308", text: "#eab308" },  // lime-yellow
];

// ════════════════════════════════════════════════════════════════
// SINGLE TOKEN CHIP
// ════════════════════════════════════════════════════════════════
const TokenChip = ({ token, tokenId, index, isHovered, onEnter, onLeave }) => {
  // Pick colour based on position (cycles if > 9 tokens)
  const colour = TOKEN_COLOURS[index % TOKEN_COLOURS.length];

  return (
    <motion.div
      // Animate chip appearance on mount — staggered delay by index
      initial={{ opacity: 0, y: 16, scale: 0.85 }}
      animate={{ opacity: 1, y: 0,  scale: 1    }}
      transition={{ delay: index * 0.06, duration: 0.35, ease: "easeOut" }}

      onMouseEnter={onEnter}   // tell parent which token is hovered
      onMouseLeave={onLeave}   // tell parent no token is hovered

      style={{
        display:       "flex",
        flexDirection: "column",
        alignItems:    "center",
        gap:           "4px",
        cursor:        "pointer",
      }}
    >
      {/* ── Position badge (0, 1, 2...) ─────────────────────────────────── */}
      <motion.span
        animate={{
          // Float up slightly when hovered
          y: isHovered ? -4 : 0,
          opacity: isHovered ? 1 : 0.5,
        }}
        transition={{ duration: 0.2 }}
        style={{
          fontSize:   "0.62rem",
          fontFamily: "Fira Code, monospace",
          color:      colour.text,
          fontWeight: "500",
        }}
      >
        {index}   {/* position index: 0, 1, 2, ... */}
      </motion.span>

      {/* ── Main chip ────────────────────────────────────────────────────── */}
      <motion.div
        animate={{
          background:  isHovered ? colour.bg        : "rgba(255,255,255,0.04)",
          borderColor: isHovered ? colour.border    : "rgba(255,255,255,0.08)",
          boxShadow:   isHovered ? `0 0 16px ${colour.border}44` : "none",
          scale:       isHovered ? 1.08 : 1,
        }}
        transition={{ duration: 0.2 }}
        style={{
          padding:      "8px 14px",
          borderRadius: "8px",
          border:       "1.5px solid",
          fontFamily:   "Fira Code, monospace",
          fontWeight:   "600",
          fontSize:     "0.9rem",
          color:        isHovered ? colour.text : "var(--text-primary)",
          letterSpacing: "0.02em",
          // Transition handles color smoothly
          transition:   "color 0.2s",
        }}
      >
        {token}
      </motion.div>

      {/* ── Token ID badge ───────────────────────────────────────────────── */}
      <motion.span
        animate={{
          opacity: isHovered ? 1 : 0.35,
          y:       isHovered ? 2 : 0,
        }}
        transition={{ duration: 0.2 }}
        style={{
          fontSize:     "0.6rem",
          fontFamily:   "Fira Code, monospace",
          color:        "var(--text-dim)",
          background:   "var(--bg-elevated)",
          padding:      "1px 5px",
          borderRadius: "4px",
          border:       "1px solid var(--border)",
        }}
      >
        id:{tokenId}   {/* the integer ID from the vocabulary */}
      </motion.span>
    </motion.div>
  );
};


// ════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ════════════════════════════════════════════════════════════════
const TokenBar = ({
  tokens,       // string[]  — e.g. ["the", "dog", "chased", ...]
  tokenIds,     // number[]  — e.g. [2, 3, 4, ...]
  onHover,      // (token: string | null) => void  — notifies parent
}) => {
  // Track which token is currently being hovered (-1 = none)
  const [hoveredIdx, setHoveredIdx] = useState(-1);

  // Handler for entering a chip — updates local state + notifies parent
  const handleEnter = (idx, token) => {
    setHoveredIdx(idx);
    onHover?.(token);    // optional chaining: safe even if onHover is undefined
  };

  // Handler for leaving a chip
  const handleLeave = () => {
    setHoveredIdx(-1);
    onHover?.(null);
  };

  if (!tokens || tokens.length === 0) {
    return (
      <div style={{ color: "var(--text-dim)", fontSize: "0.8rem" }}>
        Enter a sentence above to see tokens
      </div>
    );
  }

  return (
    <div>
      {/* ── Section header ────────────────────────────────────────────── */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <div style={{ width: "8px", height: "8px", borderRadius: "50%", background: "var(--teal)", boxShadow: "0 0 8px var(--teal)" }} />
          <span style={{ fontFamily: "Syne, sans-serif", fontWeight: "700", fontSize: "0.85rem", letterSpacing: "0.05em", textTransform: "uppercase", color: "var(--text-muted)" }}>
            Tokenisation
          </span>
        </div>
        {/* Token count pill */}
        <span style={{ fontFamily: "Fira Code, monospace", fontSize: "0.72rem", color: "var(--teal)", background: "rgba(20,184,166,0.1)", border: "1px solid rgba(20,184,166,0.3)", padding: "2px 8px", borderRadius: "99px" }}>
          {tokens.length} tokens
        </span>
      </div>

      {/* ── Token chips row ───────────────────────────────────────────── */}
      <div style={{
        display:    "flex",
        flexWrap:   "wrap",
        gap:        "12px",
        alignItems: "flex-end",
      }}>
        {tokens.map((token, idx) => (
          <TokenChip
            key      ={`${token}-${idx}`}
            token    ={token}
            tokenId  ={tokenIds?.[idx] ?? "?"}
            index    ={idx}
            isHovered={hoveredIdx === idx}
            onEnter  ={() => handleEnter(idx, token)}
            onLeave  ={handleLeave}
          />
        ))}
      </div>

      {/* ── Arrows showing the "flow" ─────────────────────────────────── */}
      <div style={{
        display:    "flex",
        alignItems: "center",
        gap:        "8px",
        marginTop:  "16px",
        fontSize:   "0.72rem",
        color:      "var(--text-dim)",
        fontFamily: "Fira Code, monospace",
      }}>
        <span style={{ color: "var(--teal)" }}>→</span>
        <span>text split on whitespace</span>
        <span style={{ color: "var(--teal)" }}>→</span>
        <span>lower-cased</span>
        <span style={{ color: "var(--teal)" }}>→</span>
        <span>looked up in vocabulary table</span>
        <span style={{ color: "var(--teal)" }}>→</span>
        <span>integer ID assigned</span>
      </div>
    </div>
  );
};

export default TokenBar;

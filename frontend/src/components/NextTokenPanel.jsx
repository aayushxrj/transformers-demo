// src/components/NextTokenPanel.jsx
// ─────────────────────────────────────────────────────────────────
// Visualises the output probability distribution over the vocabulary.
// This is what the model "thinks" the next token should be after
// processing the entire input sentence.
//
// Shows a horizontal bar chart, sorted by probability,
// with animated bars that grow on data update.
// ─────────────────────────────────────────────────────────────────

import React, { useMemo } from "react";
import { motion } from "framer-motion";

// ════════════════════════════════════════════════════════════════
// PROBABILITY BAR  — one row in the chart
// ════════════════════════════════════════════════════════════════

const ProbBar = ({ word, prob, rank, maxProb }) => {
  // Bar width is proportional to the probability relative to the highest value
  // (rather than absolute, so the top token always hits 100% width)
  const widthPct = (prob / maxProb) * 100;

  // Colour gradient: top token = amber, rest = teal → sky scale
  const colour =
    rank === 0 ? "#f59e0b" :   // gold for top prediction
    rank <= 2  ? "#14b8a6" :   // teal for 2nd/3rd
    rank <= 4  ? "#38bdf8" :   // sky for 4th/5th
                 "#4a5568";    // grey for the rest

  return (
    <div style={{
      display:    "flex",
      alignItems: "center",
      gap:        "10px",
      marginBottom:"6px",
    }}>
      {/* Rank number */}
      <span style={{
        fontFamily:  "Fira Code, monospace",
        fontSize:    "0.65rem",
        color:       "var(--text-dim)",
        width:       "16px",
        textAlign:   "right",
        flexShrink:  0,
      }}>
        #{rank + 1}
      </span>

      {/* Token word label */}
      <span style={{
        fontFamily:  "Fira Code, monospace",
        fontSize:    "0.82rem",
        fontWeight:  "600",
        color:       rank === 0 ? colour : "var(--text-primary)",
        width:       "64px",
        flexShrink:  0,
        textAlign:   "right",
      }}>
        {word}
      </span>

      {/* Animated bar track */}
      <div style={{
        flex:         1,
        background:   "rgba(255,255,255,0.04)",
        borderRadius: "4px",
        height:       "22px",
        overflow:     "hidden",
        position:     "relative",
      }}>
        <motion.div
          // Start at 0 width, animate to the calculated width
          initial={{ width: "0%" }}
          animate={{ width: `${widthPct}%` }}
          transition={{ duration: 0.6, delay: rank * 0.04, ease: "easeOut" }}
          style={{
            height:       "100%",
            borderRadius: "4px",
            background:   colour,
            opacity:      rank === 0 ? 1 : 0.55,
          }}
        />
        {/* Percentage label overlaid on bar */}
        <span style={{
          position:   "absolute",
          left:       "8px",
          top:        "50%",
          transform:  "translateY(-50%)",
          fontFamily: "Fira Code, monospace",
          fontSize:   "0.72rem",
          color:      rank === 0 ? "#fff" : "var(--text-muted)",
          fontWeight: rank === 0 ? "600" : "400",
          mixBlendMode: "screen",   // makes text readable on any bar colour
        }}>
          {(prob * 100).toFixed(1)}%
        </span>
      </div>
    </div>
  );
};


// ════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ════════════════════════════════════════════════════════════════

const NextTokenPanel = ({
  probs,         // { [word: string]: number }  — {token: probability}
  inputText,     // string — the input sentence, for context display
}) => {
  // Sort entries by probability, highest first
  const sorted = useMemo(() => {
    if (!probs) return [];
    return Object.entries(probs)
      .sort(([, a], [, b]) => b - a)   // descending sort
      .slice(0, 8);                    // top 8 predictions
  }, [probs]);

  // The highest probability (used to normalise bar widths)
  const maxProb = sorted[0]?.[1] ?? 1;

  // Top predicted word
  const topWord = sorted[0]?.[0] ?? "—";

  return (
    <div>
      {/* ── Header ────────────────────────────────────────────────── */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "20px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <div style={{ width: "8px", height: "8px", borderRadius: "50%", background: "var(--violet)", boxShadow: "0 0 8px var(--violet)" }} />
          <span style={{ fontFamily: "Syne, sans-serif", fontWeight: "700", fontSize: "0.85rem", letterSpacing: "0.05em", textTransform: "uppercase", color: "var(--text-muted)" }}>
            Next Token Prediction
          </span>
        </div>
      </div>

      {/* ── Input context display ─────────────────────────────────── */}
      <div style={{
        background:   "var(--bg-elevated)",
        border:       "1px solid var(--border)",
        borderRadius: "8px",
        padding:      "10px 14px",
        marginBottom: "20px",
        fontSize:     "0.82rem",
        color:        "var(--text-muted)",
        fontFamily:   "Fira Code, monospace",
      }}>
        <span style={{ color: "var(--text-dim)", fontSize: "0.65rem" }}>INPUT: </span>
        <span style={{ color: "var(--text-primary)" }}>{inputText}</span>
        {/* Blinking cursor */}
        <motion.span
          animate={{ opacity: [1, 0, 1] }}
          transition={{ repeat: Infinity, duration: 1, ease: "steps(2)" }}
          style={{ color: "var(--amber)", fontWeight: "bold" }}
        >
          ▌
        </motion.span>
      </div>

      {/* ── Top prediction callout ────────────────────────────────── */}
      <div style={{
        background:    "rgba(245,158,11,0.08)",
        border:        "1px solid rgba(245,158,11,0.3)",
        borderRadius:  "10px",
        padding:       "14px 18px",
        marginBottom:  "20px",
        display:       "flex",
        alignItems:    "center",
        gap:           "14px",
      }}>
        <div style={{
          fontFamily:   "Syne, sans-serif",
          fontSize:     "2.5rem",
          fontWeight:   "800",
          color:        "#f59e0b",
        }}>
          {topWord}
        </div>
        <div>
          <div style={{ fontSize: "0.7rem", color: "var(--text-dim)", fontFamily: "Fira Code, monospace", letterSpacing: "0.05em", textTransform: "uppercase" }}>
            Most likely next token
          </div>
          <div style={{ fontSize: "1.2rem", fontFamily: "Fira Code, monospace", fontWeight: "600", color: "var(--text-primary)" }}>
            {(maxProb * 100).toFixed(1)}
            <span style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginLeft: "4px" }}>%</span>
          </div>
        </div>
      </div>

      {/* ── Probability bars ──────────────────────────────────────── */}
      <div>
        {sorted.map(([word, prob], rank) => (
          <ProbBar
            key    ={word}
            word   ={word}
            prob   ={prob}
            rank   ={rank}
            maxProb={maxProb}
          />
        ))}
      </div>

      {/* ── Footnote ─────────────────────────────────────────────── */}
      <div style={{
        marginTop: "16px",
        fontSize:  "0.7rem",
        color:     "var(--text-dim)",
        fontFamily:"Fira Code, monospace",
        lineHeight: "1.6",
      }}>
        softmax(logits) over {Object.keys(probs ?? {}).length || 18}-word vocabulary
        · highest prob token would be greedily sampled
        · real models use top-k / nucleus sampling for diversity
      </div>
    </div>
  );
};

export default NextTokenPanel;

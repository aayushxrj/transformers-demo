// src/components/FlowDiagram.jsx
// ─────────────────────────────────────────────────────────────────
// Renders the high-level transformer architecture as an animated
// flow diagram.  Each stage lights up when its data is available,
// and users can click a stage to scroll to that panel.
// ─────────────────────────────────────────────────────────────────

import React, { useState } from "react";
import { motion } from "framer-motion";

// ════════════════════════════════════════════════════════════════
// STAGE DEFINITIONS
// Each entry = one box in the pipeline diagram
// ════════════════════════════════════════════════════════════════

const STAGES = [
  {
    id:       "input",
    label:    "Input Text",
    sublabel: '"the dog chased..."',
    colour:   "#4a5568",
    icon:     "⌨",
  },
  {
    id:       "tokenize",
    label:    "Tokenisation",
    sublabel: "text → integer IDs",
    colour:   "#14b8a6",   // teal
    icon:     "✂",
  },
  {
    id:       "embed",
    label:    "Embeddings",
    sublabel: "IDs → 64-dim vectors",
    colour:   "#38bdf8",   // sky
    icon:     "📍",
  },
  {
    id:       "posenc",
    label:    "Positional\nEncoding",
    sublabel: "add sin/cos signals",
    colour:   "#a78bfa",   // violet
    icon:     "🎵",
  },
  {
    id:       "qkv",
    label:    "Q · K · V",
    sublabel: "project to 3 spaces",
    colour:   "#f59e0b",   // amber
    icon:     "⚡",
  },
  {
    id:       "attn",
    label:    "Self-Attention",
    sublabel: "softmax(QKᵀ/√d) · V",
    colour:   "#ef4444",   // red
    icon:     "👁",
  },
  {
    id:       "norm",
    label:    "Add & Norm",
    sublabel: "residual + LayerNorm",
    colour:   "#6366f1",   // indigo
    icon:     "⚖",
  },
  {
    id:       "ffn",
    label:    "Feed-Forward",
    sublabel: "Linear → GELU → Linear",
    colour:   "#4ade80",   // green
    icon:     "🧪",
  },
  {
    id:       "output",
    label:    "Output",
    sublabel: "next token probs",
    colour:   "#f59e0b",   // amber
    icon:     "🎯",
  },
];

// ════════════════════════════════════════════════════════════════
// SINGLE STAGE BOX
// ════════════════════════════════════════════════════════════════

const StageBox = ({ stage, index, isActive, isComplete, onClick }) => {
  const [hovered, setHovered] = useState(false);

  return (
    <motion.div
      // Stagger animation: each box fades in sequentially
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1  }}
      transition={{ delay: index * 0.08, duration: 0.35, ease: "easeOut" }}

      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}

      style={{
        display:       "flex",
        flexDirection: "column",
        alignItems:    "center",
        gap:           "6px",
        cursor:        "pointer",
        position:      "relative",
      }}
    >
      {/* ── Icon circle ──────────────────────────────────────────── */}
      <motion.div
        animate={{
          // Active: bright glow; Complete: dim glow; Inactive: none
          boxShadow: isActive
            ? `0 0 24px ${stage.colour}88`
            : isComplete
            ? `0 0 12px ${stage.colour}44`
            : "none",
          borderColor: isActive || isComplete || hovered
            ? stage.colour
            : "var(--border)",
          background: isActive
            ? `${stage.colour}22`
            : isComplete
            ? `${stage.colour}11`
            : "var(--bg-elevated)",
          scale: hovered ? 1.08 : 1,
        }}
        transition={{ duration: 0.3 }}
        style={{
          width:        "52px",
          height:       "52px",
          borderRadius: "12px",
          border:       "1.5px solid",
          display:      "flex",
          alignItems:   "center",
          justifyContent: "center",
          fontSize:     "1.3rem",
        }}
      >
        {/* Completion checkmark overlay */}
        {isComplete && !isActive && (
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            style={{
              position:  "absolute",
              top:       "-4px",
              right:     "-4px",
              width:     "16px",
              height:    "16px",
              borderRadius: "50%",
              background: stage.colour,
              display:   "flex",
              alignItems:"center",
              justifyContent: "center",
              fontSize:  "8px",
              color:     "#000",
              fontWeight:"bold",
            }}
          >
            ✓
          </motion.div>
        )}
        {stage.icon}
      </motion.div>

      {/* ── Stage label ───────────────────────────────────────────── */}
      <div style={{
        textAlign:   "center",
        maxWidth:    "72px",
      }}>
        <div style={{
          fontFamily:  "Syne, sans-serif",
          fontWeight:  "700",
          fontSize:    "0.65rem",
          color:       isActive ? stage.colour : isComplete ? "var(--text-primary)" : "var(--text-muted)",
          lineHeight:  "1.3",
          letterSpacing: "0.02em",
          whiteSpace:  "pre-line",   // honour \n in label strings
        }}>
          {stage.label}
        </div>
        <div style={{
          fontFamily:  "Fira Code, monospace",
          fontSize:    "0.52rem",
          color:       "var(--text-dim)",
          marginTop:   "2px",
          lineHeight:  "1.3",
        }}>
          {stage.sublabel}
        </div>
      </div>
    </motion.div>
  );
};

// ════════════════════════════════════════════════════════════════
// CONNECTOR ARROW between stages
// ════════════════════════════════════════════════════════════════

const Connector = ({ fromColour, toColour, animated }) => (
  <div style={{
    display:        "flex",
    alignItems:     "center",
    flexShrink:     0,
    paddingBottom:  "20px",   // aligns arrow with centres of boxes
  }}>
    {/* Animated flowing line */}
    <svg width="28" height="2" overflow="visible">
      <motion.line
        x1="0" y1="1" x2="28" y2="1"
        stroke={animated ? fromColour : "var(--border)"}
        strokeWidth="1.5"
        strokeDasharray="4 3"
        // Animate the dash offset to create a "flowing" effect
        animate={animated ? { strokeDashoffset: [8, 0] } : {}}
        transition={{ repeat: Infinity, duration: 0.8, ease: "linear" }}
      />
      {/* Arrowhead */}
      <polygon
        points="26,-4 32,1 26,6"
        fill={animated ? fromColour : "var(--border)"}
        opacity={animated ? 1 : 0.4}
      />
    </svg>
  </div>
);


// ════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ════════════════════════════════════════════════════════════════

const FlowDiagram = ({
  activeStage,     // string | null — which stage is currently being computed
  completedStages, // Set<string>  — stages whose data has been received
  onStageClick,    // (stageId: string) => void — navigate to that panel
}) => {
  return (
    <div style={{ overflowX: "auto", paddingBottom: "8px" }}>
      {/* ── Architecture title ────────────────────────────────────── */}
      <div style={{ marginBottom: "20px", display: "flex", alignItems: "center", gap: "8px" }}>
        <div style={{ width: "8px", height: "8px", borderRadius: "50%", background: "var(--amber)", boxShadow: "0 0 8px var(--amber)" }} />
        <span style={{ fontFamily: "Syne, sans-serif", fontWeight: "700", fontSize: "0.85rem", letterSpacing: "0.05em", textTransform: "uppercase", color: "var(--text-muted)" }}>
          Transformer Pipeline
        </span>
        <span style={{ color: "var(--text-dim)", fontSize: "0.68rem", marginLeft: "auto", fontFamily: "Fira Code, monospace" }}>
          click stage to jump
        </span>
      </div>

      {/* ── Stage boxes in a row with connectors ─────────────────── */}
      <div style={{
        display:    "flex",
        alignItems: "flex-start",
        gap:        "0",
        minWidth:   "max-content",
        padding:    "8px 0",
      }}>
        {STAGES.map((stage, idx) => (
          <React.Fragment key={stage.id}>
            <StageBox
              stage       ={stage}
              index       ={idx}
              isActive    ={activeStage === stage.id}
              isComplete  ={completedStages?.has(stage.id) ?? false}
              onClick     ={() => onStageClick?.(stage.id)}
            />
            {/* No connector after the last stage */}
            {idx < STAGES.length - 1 && (
              <Connector
                fromColour={stage.colour}
                toColour  ={STAGES[idx + 1].colour}
                animated  ={activeStage === stage.id || completedStages?.has(stage.id)}
              />
            )}
          </React.Fragment>
        ))}
      </div>

      {/* ── Scale labels ──────────────────────────────────────────── */}
      <div style={{
        display:       "flex",
        justifyContent:"space-between",
        marginTop:     "8px",
        fontSize:      "0.62rem",
        color:         "var(--text-dim)",
        fontFamily:    "Fira Code, monospace",
        paddingLeft:   "8px",
        paddingRight:  "8px",
      }}>
        <span>input</span>
        <span>←── transformer block ──→</span>
        <span>output</span>
      </div>
    </div>
  );
};

export default FlowDiagram;

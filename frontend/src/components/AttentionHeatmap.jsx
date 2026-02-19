// src/components/AttentionHeatmap.jsx
// ─────────────────────────────────────────────────────────────────
// Renders the [seq × seq] attention weight matrix as a colour heatmap.
// Each cell (row=i, col=j) shows how much token i attends to token j.
//
// We use D3 for the colour scale and SVG for the grid cells.
// Framer Motion animates cells on data change.
// ─────────────────────────────────────────────────────────────────

import React, { useRef, useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import * as d3 from "d3";   // data-driven documents — colour scales, transitions

// ════════════════════════════════════════════════════════════════
// COLOUR SCALE  — maps [0, 1] attention probability → CSS colour
// ════════════════════════════════════════════════════════════════

// d3.scaleSequential creates a function: number → colour string
// The domain [0, 1] maps to the interpolator (colour gradient).
// d3.interpolateYlOrRd: yellow → orange → red (classic heatmap)
const colourScale = d3.scaleSequential()
  .domain([0, 1])
  .interpolator(d3.interpolateRgbBasis([
    "#0f1521",   // 0.0 — deep navy (near-zero attention)
    "#134e4a",   // 0.2 — dark teal
    "#14b8a6",   // 0.4 — teal
    "#f59e0b",   // 0.7 — amber
    "#ef4444",   // 1.0 — red (maximum attention)
  ]));

// ════════════════════════════════════════════════════════════════
// TOOLTIP COMPONENT  — appears on hover
// ════════════════════════════════════════════════════════════════

const Tooltip = ({ x, y, fromToken, toToken, value }) => (
  // AnimatePresence lets us animate mount/unmount
  <motion.div
    // Unique key forces re-mount (and re-animation) when content changes
    key={`${fromToken}-${toToken}`}
    initial={{ opacity: 0, scale: 0.9, y: 8 }}
    animate={{ opacity: 1, scale: 1,   y: 0  }}
    exit={   { opacity: 0, scale: 0.9, y: 8  }}
    transition={{ duration: 0.15 }}
    style={{
      position:     "fixed",   // fixed to viewport so it follows the mouse
      left:         x + 16,    // 16px offset from cursor
      top:          y - 40,
      background:   "#0f1521",
      border:       "1px solid #2a3f5f",
      borderRadius: "8px",
      padding:      "8px 12px",
      fontSize:     "0.75rem",
      fontFamily:   "Fira Code, monospace",
      color:        "#e2e8f4",
      pointerEvents:"none",    // tooltip doesn't intercept mouse events
      zIndex:       1000,
      whiteSpace:   "nowrap",
      boxShadow:    "0 4px 20px rgba(0,0,0,0.5)",
    }}
  >
    {/* Row token (attending FROM) */}
    <span style={{ color: "#f59e0b" }}>{fromToken}</span>
    {" → "}
    {/* Column token (attending TO) */}
    <span style={{ color: "#14b8a6" }}>{toToken}</span>
    {": "}
    {/* Attention weight as percentage */}
    <span style={{ color: "#e2e8f4", fontWeight: 600 }}>
      {(value * 100).toFixed(1)}%
    </span>
  </motion.div>
);


// ════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ════════════════════════════════════════════════════════════════

const AttentionHeatmap = ({
  tokens,          // string[]  — ["the", "dog", "chased", ...]
  headWeights,     // number[][]  — [seq × seq] attention probabilities
  headIdx = 0,     // number — which head is being displayed (for title)
  highlightToken,  // string | null — token to highlight (from parent hover)
}) => {
  // ── State for tooltip ──────────────────────────────────────────────────
  const [tooltip, setTooltip] = useState(null);
  // tooltip = { x, y, fromToken, toToken, value } | null

  // ── Derived measurements ───────────────────────────────────────────────
  const seq = tokens?.length ?? 0;   // number of tokens

  // Cell size adapts to sequence length (more tokens → smaller cells)
  const cellSize = seq > 8 ? 44 : seq > 6 ? 52 : 60;
  const labelW   = 64;    // width of row-label column (token names)
  const labelH   = 60;    // height of column-label row
  const gridW    = seq * cellSize;   // total grid width
  const gridH    = seq * cellSize;   // total grid height

  // ── Event handlers ─────────────────────────────────────────────────────

  // Fires when mouse enters a heatmap cell
  const handleMouseEnter = (e, row, col, value) => {
    setTooltip({
      x:         e.clientX,          // cursor X in viewport
      y:         e.clientY,          // cursor Y in viewport
      fromToken: tokens[row],         // which token is attending (row)
      toToken:   tokens[col],         // which token being attended to (col)
      value,                          // attention weight [0, 1]
    });
  };

  // Move tooltip with mouse
  const handleMouseMove = (e) => {
    if (tooltip) {
      setTooltip(prev => ({ ...prev, x: e.clientX, y: e.clientY }));
    }
  };

  const handleMouseLeave = () => setTooltip(null);

  // ── Render ─────────────────────────────────────────────────────────────
  if (!headWeights || seq === 0) {
    return (
      <div style={{ color: "var(--text-dim)", fontSize: "0.8rem", padding: "20px" }}>
        No attention data yet — run the forward pass first.
      </div>
    );
  }

  return (
    <div style={{ position: "relative" }}>

      {/* ── Head label ──────────────────────────────────────────────── */}
      <div style={{
        display:       "flex",
        alignItems:    "center",
        gap:           "8px",
        marginBottom:  "16px",
      }}>
        <div style={{
          width:        "8px",
          height:       "8px",
          borderRadius: "50%",
          background:   "var(--amber)",
          boxShadow:    "0 0 8px var(--amber)",
        }} />
        <span style={{ fontFamily: "Fira Code, monospace", fontSize: "0.75rem", color: "var(--amber)" }}>
          ATTENTION HEAD {headIdx}
        </span>
        <span style={{ color: "var(--text-dim)", fontSize: "0.7rem", marginLeft: "auto" }}>
          hover to inspect
        </span>
      </div>

      {/* ── SVG heatmap grid ─────────────────────────────────────────── */}
      <div style={{ overflowX: "auto" }}>
        <svg
          width ={labelW + gridW + 8}
          height={labelH + gridH + 8}
          onMouseMove ={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        >
          {/* ── Column labels (token names, rotated 45°) ─────────────── */}
          {tokens.map((token, col) => (
            <text
              key={`col-${col}`}
              x={labelW + col * cellSize + cellSize / 2}   // centre of column
              y={labelH - 8}
              textAnchor="start"
              dominantBaseline="middle"
              // Rotate 45° around the label's position for diagonal layout
              transform={`rotate(-45, ${labelW + col * cellSize + cellSize / 2}, ${labelH - 8})`}
              fontSize="11"
              fontFamily="Fira Code, monospace"
              // Highlight the focused token in amber
              fill={highlightToken === token ? "#f59e0b" : "#8896b0"}
              fontWeight={highlightToken === token ? "600" : "400"}
            >
              {token}
            </text>
          ))}

          {/* ── Rows ─────────────────────────────────────────────────── */}
          {headWeights.map((rowWeights, row) => (
            <g key={`row-${row}`}>

              {/* Row label (left side) */}
              <text
                x={labelW - 8}        // just left of the grid
                y={labelH + row * cellSize + cellSize / 2}
                textAnchor="end"      // right-align the label
                dominantBaseline="middle"
                fontSize="11"
                fontFamily="Fira Code, monospace"
                fill={highlightToken === tokens[row] ? "#f59e0b" : "#8896b0"}
                fontWeight={highlightToken === tokens[row] ? "600" : "400"}
              >
                {tokens[row]}
              </text>

              {/* Cells for this row */}
              {rowWeights.map((value, col) => {
                // Determine if this cell involves the highlighted token
                const isHighlighted = (
                  highlightToken === tokens[row] ||
                  highlightToken === tokens[col]
                );

                return (
                  <motion.rect
                    key={`cell-${row}-${col}`}
                    x={labelW + col * cellSize}           // X position of cell
                    y={labelH + row * cellSize}           // Y position of cell
                    width ={cellSize - 2}                 // 2px gap between cells
                    height={cellSize - 2}
                    rx={3}                               // rounded corners (3px)
                    // Colour driven by attention value via D3 colour scale
                    fill={colourScale(value)}
                    // Opacity: dim non-highlighted cells when a token is focused
                    opacity={highlightToken && !isHighlighted ? 0.2 : 1}
                    // Animated transitions when data changes
                    animate={{ fill: colourScale(value), opacity: highlightToken && !isHighlighted ? 0.2 : 1 }}
                    transition={{ duration: 0.4, ease: "easeOut" }}
                    style={{ cursor: "crosshair" }}
                    onMouseEnter={(e) => handleMouseEnter(e, row, col, value)}
                  />
                );
              })}
            </g>
          ))}

          {/* ── Attention value labels (for cells with > 20% weight) ──── */}
          {headWeights.map((rowWeights, row) =>
            rowWeights.map((value, col) =>
              value > 0.2 ? (   // only label significant cells
                <text
                  key={`label-${row}-${col}`}
                  x={labelW + col * cellSize + cellSize / 2}
                  y={labelH + row * cellSize + cellSize / 2}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fontSize="9"
                  fontFamily="Fira Code, monospace"
                  // Dark text on light cells, light text on dark cells
                  fill={value > 0.6 ? "#ffffff" : "#e2e8f4"}
                  fontWeight="500"
                  pointerEvents="none"   // don't block mouse events on the rect
                >
                  {(value * 100).toFixed(0)}
                </text>
              ) : null
            )
          )}
        </svg>
      </div>

      {/* ── Colour legend bar ────────────────────────────────────────── */}
      <div style={{ display: "flex", alignItems: "center", gap: "8px", marginTop: "12px" }}>
        <span style={{ fontSize: "0.7rem", color: "var(--text-dim)", fontFamily: "Fira Code, monospace" }}>0%</span>
        {/* Gradient bar from the colour scale */}
        <div style={{
          flex:           1,
          height:         "6px",
          borderRadius:   "3px",
          background:     "linear-gradient(to right, #0f1521, #134e4a, #14b8a6, #f59e0b, #ef4444)",
        }} />
        <span style={{ fontSize: "0.7rem", color: "var(--text-dim)", fontFamily: "Fira Code, monospace" }}>100%</span>
        <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginLeft: "8px" }}>attention weight</span>
      </div>

      {/* ── Tooltip (portal-like, position fixed) ───────────────────── */}
      <AnimatePresence>
        {tooltip && (
          <Tooltip
            x         ={tooltip.x}
            y         ={tooltip.y}
            fromToken ={tooltip.fromToken}
            toToken   ={tooltip.toToken}
            value     ={tooltip.value}
          />
        )}
      </AnimatePresence>
    </div>
  );
};

export default AttentionHeatmap;

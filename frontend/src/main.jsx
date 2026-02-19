// src/main.jsx
// ─────────────────────────────────────────────────────────────────
// This is the very first JavaScript file that runs in the browser.
// It bootstraps the React application and mounts it to the DOM.
// ─────────────────────────────────────────────────────────────────

import React from "react";
// React 18 introduces the new concurrent rendering root API
import ReactDOM from "react-dom/client";

// Our root component — the entire app lives here
import App from "./App.jsx";

// Global CSS — custom properties, resets, typography
import "./index.css";

// ── Mount React to the DOM ─────────────────────────────────────────────────
// createRoot() creates a concurrent-mode React root.
// .render() kicks off the first render of our component tree.
ReactDOM.createRoot(
  document.getElementById("root")   // the <div id="root"> in index.html
).render(
  // StrictMode wraps the app in development-only checks:
  //   • Warns about deprecated APIs
  //   • Double-invokes render to detect side effects
  //   • Has NO effect in production builds
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

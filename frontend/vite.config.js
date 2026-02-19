// vite.config.js
// ─────────────────────────────────────────────────────────────────
// Vite is a next-generation frontend build tool.
// It uses native ES modules in development (instant startup)
// and Rollup for optimised production bundles.
// ─────────────────────────────────────────────────────────────────

import { defineConfig } from "vite";    // Vite's typed config helper
import react from "@vitejs/plugin-react"; // JSX transform + HMR for React

export default defineConfig({
  // ── Plugins ──────────────────────────────────────────────────────────────
  plugins: [
    react(),   // enables JSX syntax, React Fast Refresh (HMR), and automatic imports
  ],

  // ── Dev Server ───────────────────────────────────────────────────────────
  server: {
    port: 5173,   // default Vite port

    // Proxy API calls from the React dev server to FastAPI backend.
    // This avoids CORS issues in development:
    //   Browser calls http://localhost:5173/api/forward
    //   Vite internally forwards it to http://localhost:8000/api/forward
    proxy: {
      "/api": {
        target:      "http://localhost:8000",  // FastAPI server address
        changeOrigin: true,                    // rewrite 'Host' header
        secure:       false,                   // allow self-signed certs (dev)
      },
    },
  },

  // ── Build ─────────────────────────────────────────────────────────────────
  build: {
    outDir:         "dist",    // output folder for `npm run build`
    sourcemap:      true,      // generate .map files for debugging production
    rollupOptions: {
      output: {
        // Split vendor (node_modules) from app code for better caching
        manualChunks: {
          vendor: ["react", "react-dom"],
          motion: ["framer-motion"],
          d3:     ["d3"],
        },
      },
    },
  },
});

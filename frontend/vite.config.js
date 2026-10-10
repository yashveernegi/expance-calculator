import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxy /api calls to the FastAPI backend so the frontend can use a relative
// URL and avoid CORS issues during development.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});

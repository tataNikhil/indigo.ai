import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development, /api/* calls are forwarded to the FastAPI server,
// so the browser never has to deal with CORS.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000",
      "/health": "http://localhost:8000",
    },
  },
});

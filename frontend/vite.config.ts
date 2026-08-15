import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The bundle is served by the Python package, so it builds straight into it.
// `emptyOutDir` has to be explicit because the target sits outside the Vite root;
// without it Vite refuses to clean a directory it does not own.
export default defineConfig({
  plugins: [react()],
  base: "/",
  build: {
    outDir: "../qecgen/ui/static",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      // `qecgen ui --dev` serves the API here while Vite serves the pages.
      "/api": {
        target: "http://127.0.0.1:8765",
        changeOrigin: false,
      },
    },
  },
});

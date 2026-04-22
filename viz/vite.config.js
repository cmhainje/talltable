import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

export default defineConfig({
  plugins: [svelte()],
  server: {
    host: true,
    proxy: {
      "/ws": { target: "ws://localhost:8000", ws: true },
      "/chunks": { target: "http://localhost:8000" },
      "/map": { target: "http://localhost:8000" },
    },
  },
});

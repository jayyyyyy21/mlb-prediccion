import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// En desarrollo, /api se redirige al backend FastAPI (las claves nunca llegan al navegador).
export default defineConfig({
  plugins: [react()],
  // Modo instantánea: rutas relativas para servir la app como ficheros estáticos.
  base: process.env.VITE_STATIC === "1" ? "./" : "/",
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});

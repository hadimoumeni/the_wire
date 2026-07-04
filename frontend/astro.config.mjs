import { defineConfig } from 'astro/config';

// Static build (default). Output goes to dist/, which api.py serves.
export default defineConfig({
  server: { port: 4321 },
});

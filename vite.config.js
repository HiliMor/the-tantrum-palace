export default {
  // Relative base so the build works on GitHub Pages under /the-tantrum-palace/.
  base: './',
  build: { target: 'esnext' },
  esbuild: { target: 'esnext' },
  optimizeDeps: { esbuildOptions: { target: 'esnext' } },
};

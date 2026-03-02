# Contributing to accel-gpu

Thanks for your interest! Contributions are welcome.

## Development setup

1. **Clone and install**

   ```bash
   git clone https://github.com/Phantasm0009/accel-gpu.git
   cd accel-gpu
   npm install
   ```

2. **Run tests**

   ```bash
   npm test
   ```

   Tests run in Node with Vitest using the CPU backend (no browser required). Use `npm run test:watch` for watch mode.

3. **Lint and format**

   ```bash
   npm run lint        # ESLint
   npm run lint:fix    # Auto-fix lint issues
   npm run format      # Prettier
   npm run format:check # Check formatting
   ```

4. **Build**

   ```bash
   npm run build
   ```

5. **Build the site (demos)**

   ```bash
   npm run build:site
   ```

   Then serve the `deploy/` folder locally (e.g. `npx serve deploy`) to test the demo, benchmark, and playground pages.

## Project structure

- **`src/`** — TypeScript source
  - `index.ts` — Public API, `init()`, context
  - `array.ts` — `GPUArray` class, method chaining
  - `backend/` — WebGPU, WebGL2, and CPU backends
  - `ops/` — Math, linear algebra, ML primitives
- **`scripts/`** — Build and site scripts
- **`example/`, `benchmark/`, `playground/`** — Demo pages

## Architecture

- **Backend abstraction** — `init()` chooses WebGPU → WebGL2 → CPU based on availability.
- **Kernels** — WebGPU uses WGSL compute shaders; WebGL2 uses fragment shaders with RGBA8 float packing; CPU uses plain JavaScript.
- **Shape inference** — `matmul`, `softmax`, etc. infer M, N, K from array shapes when possible.

## Submitting changes

1. Open an [issue](https://github.com/Phantasm0009/accel-gpu/issues) to discuss larger changes.
2. Fork the repo, create a branch, make your changes.
3. Ensure `npm test` and `npm run lint` pass.
4. Open a pull request with a clear description of the change.

## Code style

- TypeScript, ES modules
- No `any` in public API
- Keep the bundle small; avoid heavy dependencies

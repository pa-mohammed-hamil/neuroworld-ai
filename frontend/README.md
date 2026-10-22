# NeuroWorld AI Frontend

39 self-contained HTML pages with embedded CSS and JavaScript. Open `index.html` directly in a browser or run `python -m http.server 5500` from this directory.

## Included working browser features
- Responsive 39-page navigation
- Interactive canvas simulation with 4 environments
- Pause, resume, step, reset, speed controls
- Experiment parameter controls, seeded initialization
- Save/load presets in browser localStorage
- Export session JSON and download canvas PNG
- Simulated telemetry chart, activity log, responsive layout
- Optional health check for `http://127.0.0.1:8000/health`

## Backend integration needed
Actual PyTorch training, checkpoint loading, GPU metrics, dataset uploads, neural inference, video understanding, model registry and benchmarks require FastAPI routes. The UI does not pretend to execute these operations.

## Notes
Every HTML file embeds CSS and JavaScript, as requested. The pages share a design system and simulation core; the displayed visualization is a client-side demo rather than a trained model.

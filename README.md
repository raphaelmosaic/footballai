# footballai

Soccer (Fußball) video analysis: detect and track all players and the ball from a
fixed-camera match video, project their positions into real pitch coordinates
(meters), and expose the results for a downstream prediction model and a dashboard.

## Subprojects

- **[`pipeline/`](pipeline/)** — the Python data pipeline (the deliverable). Ingests a
  video and emits a long/tidy table of per-frame player/ball positions in pitch meters,
  with persistent IDs, team labels, and occlusion provenance. See
  [`pipeline/docs/superpowers/`](pipeline/docs/superpowers/) for the design spec and
  implementation plan.
- **`dashboard/`** — _(planned)_ FastAPI + SvelteKit dashboard that reads the pipeline's
  artifacts and visualizes the video overlay, a 2D pitch minimap, and the coordinate table.

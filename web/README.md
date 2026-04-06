## Web

Standalone Tactics 3D viewer for FirecastRL.

Run locally:

```bash
npm install
npm start
```

Build packaged static assets into `firecastrl_env/web_dist`:

```bash
npm run build
```

Environment variables:

- `API_BASE_URL` default `http://localhost:6969`
- `WS_URL` default empty. When unset, the viewer connects to `ws(s)://<current-host>/ws`.
- `GENERATED_ASSETS_BASE_URL` default `/data`

Package runtime:

1. Build the static viewer once:

```bash
cd web
npm run build
```

2. In Python, create the env with `render_mode="3d"` and step it normally:

```python
import gymnasium as gym
import firecastrl_env

env = gym.make("firecastrl/Wildfire-env0", render_mode="3d")
obs, info = env.reset(seed=42)
env.render()
```

The env serves the built viewer itself and streams `terrain_init` / `snapshot` messages from the live Python env over WebSocket on the same port.

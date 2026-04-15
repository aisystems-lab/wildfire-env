## Web

This directory contains the packaged 3D viewer used by FirecastRL for `render_mode="3d"`.

The Python environment serves the built static assets from `firecastrl_env/web_dist` and streams live terrain and fire-state updates over WebSocket.

### Install

```bash
cd web
npm install
```

### Local development

Run the standalone webpack dev server:

```bash
npm start
```

This starts the viewer on port `8080` and is useful only when developing the frontend itself.

### Build packaged assets

Build the static viewer bundle into `../firecastrl_env/web_dist`:

```bash
npm run build
```

You should rebuild whenever files under `web/src` or `web/webpack.config.js` change.

### Runtime with Python

After building once, run the Python environment normally:

```python
import gymnasium as gym
import firecastrl_env

env = gym.make("firecastrl/Wildfire-env0", render_mode="3d")
obs, info = env.reset(seed=42)
env.render()
```

The environment serves the built viewer itself and streams `terrain_init` and `snapshot` messages from the live Python env over WebSocket on the same port.

### Environment variables

- `API_BASE_URL`: defaults to `http://localhost:6969`
- `WS_URL`: defaults to empty; when unset, the viewer connects to `ws(s)://<current-host>/ws`
- `GENERATED_ASSETS_BASE_URL`: defaults to `/data`

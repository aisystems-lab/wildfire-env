# FirecastRL: Wildfire Suppression Environment 

A **Gymnasium-compatible** wildfire simulation environment with physics-informed fire spread dynamics and helicopter-based firefighting. Designed for reinforcement learning research in wildfire management and suppression strategies.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-0.29+-green.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Features

- **Physics-Informed Fire Dynamics**: Realistic fire spread based on terrain, vegetation, wind, and elevation
- **Gymnasium API**: Fully compatible with modern RL frameworks (Stable-Baselines3, RLlib, etc.)
- **Real-Time Visualization**: Fast Pygame-based rendering at 60 FPS
- **Flexible Observation Space**: Dict-based observations with customizable wrappers
- **Helitack Firefighting**: Simulate aerial water/retardant drops
- **Custom Reward Functions**: Easy-to-extend reward system via wrappers
- **Real Terrain Data**: Includes landcover and elevation maps based on real geographic data

---

## Environment Screenshots

### Early Fire Spread
![Early Fire Spread](https://github.com/aisystems-lab/wildfire-env/blob/434305ce1b992c2cfd5f8c7818e5d475a1fb8642/docs/screenshot_early.png)

### Active Firefighting Operations
![Active Firefighting](https://github.com/aisystems-lab/wildfire-env/blob/434305ce1b992c2cfd5f8c7818e5d475a1fb8642/docs/screenshot_active.png)

**Legend:**
- **Dark Blue** = Water/non-burnable areas
- **Black** = Unburned terrain
- **Pink/Purple gradient** = Fire spread prediction (scheduled ignition times)
- **Orange** = Currently burning
- **Dark Gray** = Already burnt/extinguished
- **Yellow X** = Helicopter position
- **Blue circles** = Recent helitack drop locations (fade over time)

---

## Installation

### From TestPyPI (for testing)

```bash
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    firecastrl-env
```

### From Source

```bash
git clone https://github.com/aisystems-lab/firecast-rl.git
cd firecast-rl
pip install -e .
```

### Dependencies

- `gymnasium >= 0.29.0`
- `numpy >= 1.23`
- `pillow >= 9.0.0`
- `pygame >= 2.5.0`
- `requests >= 2.28.0`

---

## Quick Start

```python
import gymnasium as gym
import firecastrl_env  # Registers the environment

# Create environment
env = gym.make("firecastrl/Wildfire-env0", render_mode="human")

# Reset environment
obs, info = env.reset(seed=42)

# Run episode
done = False
truncated = False
total_reward = 0.0

while not (done or truncated):
    # Sample random action (or use your RL policy)
    action = env.action_space.sample()
    
    # Step environment
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    
    # Render (if render_mode="human")
    env.render()

env.close()
print(f"Episode reward: {total_reward:.2f}")
```

---

## Environment Details

### Observation Space

`Dict` with the following keys:

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `cells` | `Box(float32)` | `(160, 240)` | Ignition time for each cell (`inf` for unscheduled) |
| `helicopter_coord` | `Box(int32)` | `(2,)` | Helicopter position `[x, y]` |
| `quenched_cells` | `Box(float32)` | `(1,)` | Number of cells extinguished this step |

### Action Space

`Discrete(5)` - Five possible actions:

| Action | Value | Description |
|--------|-------|-------------|
| Move Down | `0` | Move helicopter down |
| Move Up | `1` | Move helicopter up |
| Move Left | `2` | Move helicopter left |
| Move Right | `3` | Move helicopter right |
| Helitack | `4` | Drop water/retardant in radius around helicopter |

### Reward Function

The default reward encourages fire suppression:
- **+5.0** per cell extinguished by helitack
- **-0.05** per cell currently burning (continuous penalty)
- **-0.01** per time step (encourages faster containment)
- **-1.0** for performing helitack on non-burnable/burnt cells

### Episode Termination

- **Terminated**: All fires extinguished (`cells_burning == 0`)
- **Truncated**: Maximum timesteps reached (default: 500)

### Info Dictionary

Each `step()` returns an `info` dict with:
- `cells_burning`: Current count of burning cells
- `cells_burnt`: Total cells that have burnt
- `simulation_time`: Elapsed simulation time

---

## Rendering Modes

### Human Mode (Interactive Window)
```python
env = gym.make("firecastrl/Wildfire-env0", render_mode="human")
```
Opens a Pygame window with real-time visualization at 60 FPS.

### RGB Array Mode (For Recording)
```python
env = gym.make("firecastrl/Wildfire-env0", render_mode="rgb_array")
rgb_frame = env.render()  # Returns numpy array (H, W, 3)
```
Returns RGB arrays suitable for video recording or analysis.

---

## Custom Wrappers

FirecastRL provides powerful wrappers to customize observations and rewards:

### 1. `CellObservationWrapper`

Adds detailed cell-level features to observations, replacing the basic `cells` array with a multi-channel tensor.

**Available Properties:**
- `ignition_time` - Time when cell will ignite
- `spread_rate` - Fire spread rate
- `burn_time` - Total burn duration
- `fire_state` - Current fire state (Unburnt/Burning/Burnt)
- `is_river` - Non-burnable water cells
- `is_unburnt_island` - Isolated unburnt regions
- `helitack_drops` - Number of times helitack performed on cell
- `elevation` - Terrain elevation
- `zone` - Vegetation/landcover zone
- `vegetation` - Vegetation type
- `drought` - Drought index
- `position` - Normalized X,Y coordinates

**Usage:**

```python
from firecastrl_env.wrappers import CellObservationWrapper

# Use default properties (most common features)
env = gym.make("firecastrl/Wildfire-env0")
env = CellObservationWrapper(env)

# Observation now includes 'detailed_cells' with shape (6, 160, 240)
# 6 channels: ignition_time, spread_rate, fire_state, is_river, helitack_drops, elevation

# Custom properties
env = CellObservationWrapper(
    env,
    properties=['ignition_time', 'fire_state', 'elevation', 'position'],
    remove_basic_cells=True  # Remove redundant 'cells' key
)

# All properties
env = CellObservationWrapper(env, properties='all')
```

**Benefits:**
- Richer state representation for deep RL
- CNN-friendly multi-channel format
- Normalized values in `[0, 1]` range
- Flexible feature selection

### 2. `CustomRewardWrapper`

Override the default reward function with your own logic.

**Usage:**

```python
from firecastrl_env.wrappers import CustomRewardWrapper

def my_reward_function(env, prev_state, curr_state):
    """
    Custom reward based on state changes.
    
    Args:
        env: The environment instance
        prev_state: Dict with keys: cells_burning, cells_burnt, helicopter_coord, quenched_cells
        curr_state: Dict with same keys as prev_state
    
    Returns:
        float: Custom reward value
    """
    # Example: Heavily penalize fire spread, reward containment
    delta_burning = curr_state['cells_burning'] - prev_state['cells_burning']
    reward = -2.0 * delta_burning  # Penalty for new ignitions
    reward += 10.0 * curr_state['quenched_cells']  # Bonus for extinguishing
    return float(reward)

env = gym.make("firecastrl/Wildfire-env0")
env = CustomRewardWrapper(env, reward_fn=my_reward_function)
```

**Default Reward Function:**
If no `reward_fn` is provided, uses a built-in default that balances:
- Extinguishing fires (+)
- Preventing spread (-)
- Time efficiency (-)

### 3. Combining Wrappers

Wrappers can be stacked:

```python
from firecastrl_env.wrappers import CellObservationWrapper, CustomRewardWrapper

env = gym.make("firecastrl/Wildfire-env0")

# Add detailed observations
env = CellObservationWrapper(
    env,
    properties=['ignition_time', 'fire_state', 'elevation', 'spread_rate']
)

# Add custom reward
env = CustomRewardWrapper(env, reward_fn=my_reward_function)

# Now ready for training!
obs, info = env.reset()
```

---

## Training with Stable-Baselines3

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import firecastrl_env

# Create and verify environment
env = gym.make("firecastrl/Wildfire-env0")
check_env(env, warn=True)

# Train PPO agent
model = PPO(
    "MultiInputPolicy",  # For Dict observation spaces
    env,
    verbose=1,
    tensorboard_log="./firecast_tensorboard/"
)

model.learn(total_timesteps=100_000)
model.save("firecast_ppo_agent")

# Evaluate trained agent
obs, info = env.reset()
for _ in range(500):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done or truncated:
        break

env.close()
```

---

## Configuration

Key parameters can be modified in `firecastrl_env/envs/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gridWidth` | 240 | Grid width (cells) |
| `gridHeight` | 160 | Grid height (cells) |
| `cellSize` | 250 | Cell size (meters) |
| `MAX_TIMESTEPS` | 2000 | Episode truncation limit |
| `HELICOPTER_SPEED` | 3 | Cells moved per action |
| `HELITACK_RADIUS` | 7 | Effective radius of water drops |

---

## Project Structure

```
firecastrl_env/
├── __init__.py                    # Environment registration
├── envs/
│   ├── __init__.py
│   ├── wildfire_env.py           # Main Gymnasium environment
│   ├── config.py                 # Configuration parameters
│   ├── environment/              # Core simulation components
│   │   ├── cell.py              # Grid cell representation
│   │   ├── enums.py             # Fire states, burn indices, etc.
│   │   ├── helper.py            # Utility functions
│   │   ├── vector.py            # 2D vector math
│   │   ├── wind.py              # Wind modeling
│   │   └── zone.py              # Vegetation zones
│   ├── fire_engine/             # Fire physics engine
│   │   ├── fire_engine.py      # Fire spread simulation
│   │   ├── fire_spread_rate.py # Spread rate calculations
│   │   └── utils.py            # Fire engine utilities
│   └── training_environments/   # Terrain data
│       ├── landcover_1.png     # Real landcover map
│       └── heightmap_1.png     # Real elevation data
└── wrappers/                    # Custom Gymnasium wrappers
    ├── __init__.py
    ├── cell_observation.py      # Detailed cell features
    ├── custom_reward.py         # Custom reward functions
    └── clip_reward.py           # Reward clipping utility
```

---

## Citation

If you use FirecastRL in your research, please cite:

```bibtex
@software{firecastrl2025,
  title={Spatiotemporal Wildfire Prediction and Reinforcement Learning for Helitack Suppression},
  author={Shaurya Mathur, Shreyas Bellary Manjunath, Nitin Kulkarni, Alina Vereshchaka},
  year={2025},
  url={https://sites.google.com/view/firecastrl}
}
```

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

**Shreyas Bellary Manjunath** - sbellary@buffalo.edu  
**Shaurya Mathur** - smathur4@buffalo.edu

**Project Link:** https://github.com/aisystems-lab/firecast-rl

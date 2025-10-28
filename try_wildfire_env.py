import argparse
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import ObservationWrapper, spaces
from stable_baselines3 import PPO

# Import registers the env via entry point in firecastrl_env/__init__.py
import firecastrl_env  # noqa: F401


def describe_obs(obs: dict):
    def shape_of(x):
        try:
            return tuple(x.shape)
        except Exception:
            return type(x).__name__

    keys = list(obs.keys())
    shapes = {k: shape_of(obs[k]) for k in keys}
    print("Observation keys:", keys)
    print("Observation shapes:", shapes)


class SanitizeObs(ObservationWrapper):
    def __init__(self, env, cap: float = 1e5):
        super().__init__(env)
        self.cap = float(cap)
        # Keep shapes; bound to [0,1] for cells/coords
        self.observation_space = spaces.Dict({
            'cells': spaces.Box(low=0.0, high=1.0, shape=(160, 240), dtype=np.float32),
            'helicopter_coord': spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            'quenched_cells': spaces.Box(low=0.0, high=np.finfo(np.float32).max, shape=(1,), dtype=np.float32),
        })

    def observation(self, obs):
        cells = np.array(obs['cells'], dtype=np.float32)
        # Map inf/NaN -> cap, clip and normalize to [0,1]
        cells = np.nan_to_num(cells, nan=self.cap, posinf=self.cap, neginf=0.0)
        cells = np.clip(cells, 0.0, self.cap) / self.cap

        hc = np.array(obs['helicopter_coord'], dtype=np.float32)
        hc = np.array([hc[0] / 239.0, hc[1] / 159.0], dtype=np.float32)

        qc = np.array(obs['quenched_cells'], dtype=np.float32)
        return {'cells': cells, 'helicopter_coord': hc, 'quenched_cells': qc}


def main():
    parser = argparse.ArgumentParser(description="Quick smoke test for firecastrl-env")
    parser.add_argument("--steps", type=int, default=1000, help="Max steps to run")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reset")
    parser.add_argument("--verbose", action="store_true", help="Print per-step logs")
    parser.add_argument("--render", action="store_true", help="Visualize (after training) ignition times and helicopter")
    parser.add_argument("--train", action="store_true", help="Train PPO agent before rendering")
    parser.add_argument("--timesteps", type=int, default=20000, help="Total timesteps for PPO training")
    parser.add_argument("--save_model", type=str, default="ppo_firecastrl.zip", help="Path to save trained model")
    parser.add_argument("--render_steps", type=int, default=300, help="Steps to render after training")
    args = parser.parse_args()

    env = gym.make("firecastrl/Wildfire-env0")
    # IMPORTANT: sanitize observations to avoid inf/NaN -> NaN logits
    env = SanitizeObs(env, cap=1e5)
    obs, info = env.reset(seed=args.seed)
    print("Env reset OK.")
    describe_obs(obs)

    total_reward = 0.0
    steps_run = 0

    # Optional PPO training
    model = None
    if args.train:
        # Use MultiInputPolicy for Dict observations
        model = PPO("MultiInputPolicy", env, verbose=1)
        print(f"Training PPO for {args.timesteps} timesteps...")
        model.learn(total_timesteps=args.timesteps)
        try:
            model.save(args.save_model)
            print(f"Saved model to {args.save_model}")
        except Exception as e:
            print(f"Warning: could not save model: {e}")

    # If rendering (post-training), set up the figure
    if args.render:
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 6))
        arr = np.array(obs["cells"], dtype=np.float32)  # already normalized [0,1]
        vmin, vmax = 0.0, 1.0
        cmap = plt.cm.magma.copy()
        cmap.set_bad(color="#2c2c2c")
        im = ax.imshow(arr, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Ignition (normalized)")
        hx, hy = int(obs["helicopter_coord"][0]), int(obs["helicopter_coord"][1])
        heli = ax.scatter([hx], [hy], c="#00ffff", s=20, marker="x", linewidths=1.2, label="helicopter")
        ax.set_title("Ignition Times (NaN = not scheduled)")
        ax.set_xlabel("x (cols)")
        ax.set_ylabel("y (rows)")
        ax.legend(loc="upper right")
        plt.tight_layout()

    # Run rollout
    if args.render and (args.train and model is not None):
        # Render trained agent
        obs, info = env.reset(seed=args.seed)
        for t in range(args.render_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if args.verbose:
                print(f"t={t+1} action={action} reward={reward:.3f} terminated={terminated} truncated={truncated}")
            arr = np.array(obs["cells"], dtype=np.float32)
            im.set_data(arr)
            hx, hy = int(obs["helicopter_coord"][0]), int(obs["helicopter_coord"][1])
            heli.set_offsets([[hx, hy]])
            ax.set_title(f"Ignition Times — step {t+1}  reward {reward:.3f}")
            plt.pause(0.001)
            if terminated or truncated:
                break
    else:
        # Default: quick smoke test (random policy), optional render
        for t in range(args.steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps_run = t + 1
            if args.verbose:
                print(
                    f"t={steps_run} action={action} reward={reward:.3f} "
                    f"terminated={terminated} truncated={truncated}"
                )
            if args.render:
                arr = np.array(obs["cells"], dtype=np.float32)
                im.set_data(arr)
                hx, hy = int(obs["helicopter_coord"][0]), int(obs["helicopter_coord"][1])
                heli.set_offsets([[hx, hy]])
                ax.set_title(f"Ignition Times — step {steps_run}  reward {reward:.3f}")
                plt.pause(0.001)
            if terminated or truncated:
                break

    env.close()
    print(f"Done. Steps: {steps_run}, Total reward: {total_reward:.3f}")


if __name__ == "__main__":
    main()



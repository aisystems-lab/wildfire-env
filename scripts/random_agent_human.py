import argparse
import pathlib
import sys
import time

import gymnasium as gym

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import firecastrl_env  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a random FirecastRL episode with the Pygame human renderer."
    )
    parser.add_argument("--env-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=8.0, help="How fast to step the random policy.")
    args = parser.parse_args()

    env = gym.make("firecastrl/Wildfire-env0", env_id=args.env_id, render_mode="human")
    try:
        _, _ = env.reset(seed=args.seed)
        env.render()

        terminated = False
        truncated = False
        frame_delay = 1.0 / args.fps if args.fps > 0 else 0.0

        while not (terminated or truncated):
            if frame_delay > 0:
                time.sleep(frame_delay)
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            env.render()
    finally:
        env.close()


if __name__ == "__main__":
    main()

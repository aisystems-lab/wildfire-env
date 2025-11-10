"""
Test script for firecastrl_env - Tests Gymnasium compatibility and renders a random agent.

Usage:
    python try_wildfire_env.py --steps 100                    # Quick test
    python try_wildfire_env.py --steps 500 --render           # Test with rendering (60 FPS default)
    python try_wildfire_env.py --steps 500 --render --fps 10  # Custom FPS rendering
"""

import argparse
import gymnasium as gym
import numpy as np
import time

# Import to register the environment
import firecastrl_env


def test_gymnasium_compatibility(env):
    """Test basic Gymnasium API compatibility."""
    print("\n" + "="*60)
    print("Testing Gymnasium Compatibility")
    print("="*60)
    
    # Check required attributes
    checks = {
        "action_space": hasattr(env, 'action_space'),
        "observation_space": hasattr(env, 'observation_space'),
        "metadata": hasattr(env, 'metadata'),
        "spec": hasattr(env, 'spec'),
        "reset": callable(getattr(env, 'reset', None)),
        "step": callable(getattr(env, 'step', None)),
        "render": callable(getattr(env, 'render', None)),
        "close": callable(getattr(env, 'close', None)),
    }
    
    for check_name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")
    
    if all(checks.values()):
        print("\n✓ All Gymnasium compatibility checks passed!")
    else:
        print("\n✗ Some compatibility checks failed!")
        return False
    
    # Check action space
    print(f"\nAction Space: {env.action_space}")
    print(f"  Type: {type(env.action_space).__name__}")
    if hasattr(env.action_space, 'n'):
        print(f"  Number of actions: {env.action_space.n}")
    
    # Check observation space
    print(f"\nObservation Space: {env.observation_space}")
    print(f"  Type: {type(env.observation_space).__name__}")
    if isinstance(env.observation_space, gym.spaces.Dict):
        print(f"  Keys: {list(env.observation_space.spaces.keys())}")
        for key, space in env.observation_space.spaces.items():
            print(f"    - {key}: {space}")
    
    return True


def test_reset(env, seed=42):
    """Test environment reset."""
    print("\n" + "="*60)
    print("Testing Reset")
    print("="*60)
    
    try:
        obs, info = env.reset(seed=seed)
        print("✓ Reset successful")
        
        # Check observation structure
        if isinstance(obs, dict):
            print(f"\nObservation keys: {list(obs.keys())}")
            for key, value in obs.items():
                if hasattr(value, 'shape'):
                    print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
                    print(f"    min={np.min(value):.4f}, max={np.max(value):.4f}, mean={np.mean(value):.4f}")
                else:
                    print(f"  - {key}: {value}")
        
        print(f"\nInfo dict keys: {list(info.keys())}")
        
        # Check for NaN or Inf values
        has_issues = False
        if isinstance(obs, dict):
            for key, value in obs.items():
                if hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.number):
                    if np.any(np.isnan(value)):
                        print(f"⚠ Warning: {key} contains NaN values")
                        has_issues = True
                    if np.any(np.isinf(value)):
                        print(f"⚠ Warning: {key} contains Inf values")
                        has_issues = True
        
        if not has_issues:
            print("\n✓ No NaN or Inf values detected")
        
        return obs, info
    
    except Exception as e:
        print(f"✗ Reset failed with error: {e}")
        raise


def test_step(env, obs, num_steps=5):
    """Test environment step function."""
    print("\n" + "="*60)
    print(f"Testing Step (running {num_steps} random steps)")
    print("="*60)
    
    try:
        for i in range(num_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"\nStep {i+1}:")
            print(f"  Action: {action}")
            print(f"  Reward: {reward:.4f}")
            print(f"  Terminated: {terminated}")
            print(f"  Truncated: {truncated}")
            
            # Check return types
            assert isinstance(reward, (int, float, np.number)), "Reward should be numeric"
            assert isinstance(terminated, bool), "Terminated should be boolean"
            assert isinstance(truncated, bool), "Truncated should be boolean"
            assert isinstance(info, dict), "Info should be dictionary"
            
            if terminated or truncated:
                print(f"  Episode ended at step {i+1}")
                break
        
        print("\n✓ Step tests passed")
        return True
    
    except Exception as e:
        print(f"✗ Step failed with error: {e}")
        raise


def test_render(env):
    """Test rendering functionality."""
    print("\n" + "="*60)
    print("Testing Render")
    print("="*60)
    
    try:
        result = env.render()
        print(f"✓ Render successful (render_mode={env.render_mode})")
        if result is not None:
            if hasattr(result, 'shape'):
                print(f"  Returned array shape: {result.shape}")
        return True
    except Exception as e:
        print(f"✗ Render failed with error: {e}")
        return False


def run_random_agent(env, num_steps=100, render=False, fps=5, verbose=False):
    """Run a random agent for testing."""
    print("\n" + "="*60)
    print(f"Running Random Agent ({num_steps} steps)")
    print("="*60)
    
    obs, info = env.reset()
    total_reward = 0.0
    episode_rewards = []
    current_episode_reward = 0.0
    
    delay = 1.0 / fps if render and fps > 0 else 0.0
    
    for step in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        current_episode_reward += reward
        
        if render:
            env.render()
            if delay > 0:
                time.sleep(delay)
        
        if verbose:
            print(f"Step {step+1}: action={action}, reward={reward:.4f}, "
                  f"burning={info.get('cells_burning', 'N/A')}, "
                  f"burnt={info.get('cells_burnt', 'N/A')}")
        
        if terminated or truncated:
            episode_rewards.append(current_episode_reward)
            current_episode_reward = 0.0
            
            if step < num_steps - 1:
                obs, info = env.reset()
                if verbose:
                    print(f"  → Episode ended. Reset for next episode.")
    
    if current_episode_reward != 0.0:
        episode_rewards.append(current_episode_reward)
    
    print(f"\n✓ Random agent completed")
    print(f"  Total steps: {num_steps}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Average reward per step: {total_reward/num_steps:.4f}")
    if episode_rewards:
        print(f"  Episodes completed: {len(episode_rewards)}")
        print(f"  Average episode reward: {np.mean(episode_rewards):.2f}")


def test_with_wrappers():
    """Test environment with custom wrappers."""
    print("\n" + "="*60)
    print("Testing with Custom Wrappers")
    print("="*60)
    
    from firecastrl_env.wrappers import CellObservationWrapper, CustomRewardWrapper
    
    env = gym.make("firecastrl/Wildfire-env0")
    
    # Test CellObservationWrapper with default properties
    print("\n1. Testing CellObservationWrapper (default properties)...")
    wrapped_env = CellObservationWrapper(env)
    print(f"   Properties included: {wrapped_env.properties}")
    print(f"   Feature count: {wrapped_env.feature_count}")
    
    obs, info = wrapped_env.reset()
    print(f"   Observation keys: {list(obs.keys())}")
    print(f"   Detailed cells shape: {obs['detailed_cells'].shape}")
    print("   ✓ CellObservationWrapper works with defaults")
    wrapped_env.close()
    
    # Test CellObservationWrapper with custom properties
    print("\n2. Testing CellObservationWrapper (custom properties)...")
    env = gym.make("firecastrl/Wildfire-env0")
    wrapped_env = CellObservationWrapper(
        env,
        properties=['ignition_time', 'fire_state', 'is_river']
    )
    obs, info = wrapped_env.reset()
    print(f"   Properties included: {wrapped_env.properties}")
    print(f"   Detailed cells shape: {obs['detailed_cells'].shape}")
    print("   ✓ CellObservationWrapper works with custom properties")
    wrapped_env.close()
    
    # Test CustomRewardWrapper
    print("\n3. Testing CustomRewardWrapper...")
    
    def simple_reward(env, prev, curr):
        return 10.0 * curr['quenched_cells'] - 0.1 * curr['cells_burning']
    
    env = gym.make("firecastrl/Wildfire-env0")
    wrapped_env = CustomRewardWrapper(env, reward_fn=simple_reward)
    obs, info = wrapped_env.reset()
    obs, reward, terminated, truncated, info = wrapped_env.step(env.action_space.sample())
    print(f"   Custom reward: {reward:.4f}")
    print("   ✓ CustomRewardWrapper works")
    wrapped_env.close()
    
    # Test combined wrappers
    print("\n4. Testing combined wrappers...")
    env = gym.make("firecastrl/Wildfire-env0")
    env = CellObservationWrapper(env, properties=['ignition_time', 'fire_state'])
    env = CustomRewardWrapper(env)
    obs, info = env.reset()
    print(f"   Observation keys: {list(obs.keys())}")
    print("   ✓ Wrapper stacking works")
    env.close()
    
    print("\n✓ All wrapper tests passed!")


def main():
    parser = argparse.ArgumentParser(description="Test firecastrl_env Gymnasium compatibility")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--fps", type=int, default=None, help="Frames per second for rendering (default: uses env's metadata fps)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed step information")
    parser.add_argument("--test-wrappers", action="store_true", help="Test custom wrappers")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Firecast RL Environment Test Suite")
    print("="*60)
    
    # Create environment
    render_mode = "human" if args.render else None
    env = gym.make("firecastrl/Wildfire-env0", render_mode=render_mode)
    
    # Use environment's default FPS if not specified
    fps = args.fps if args.fps is not None else env.metadata.get("render_fps", 30)
    
    try:
        # Run tests
        test_gymnasium_compatibility(env)
        obs, info = test_reset(env, seed=args.seed)
        test_step(env, obs, num_steps=5)
        
        if args.render:
            test_render(env)
        
        # Run random agent
        run_random_agent(
            env,
            num_steps=args.steps,
            render=args.render,
            fps=fps,
            verbose=args.verbose
        )
        
        # Test wrappers if requested
        if args.test_wrappers:
            env.close()
            test_with_wrappers()
        
        print("\n" + "="*60)
        print("✓ All tests completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        env.close()


if __name__ == "__main__":
    main()

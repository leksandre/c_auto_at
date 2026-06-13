#!/usr/bin/env python3
"""
train_rl.py — RL training script for drone attack using PPO.

This script:
1. Creates the DroneAttackEnv Gymnasium environment
2. Wraps it for Stable-Baselines3 (vectorised environments)
3. Configures PPO with hyperparameters tuned for drone control
4. Optionally loads BC model weights as initial policy
5. Runs training with callbacks for evaluation and checkpointing
6. Saves the final model and exports to ONNX

Usage:
    python3 train_rl.py                          # Full training
    python3 train_rl.py --timesteps 100000       # Quick test
    python3 train_rl.py --resume ./logs/checkpoint.zip  # Resume training

Author: Generated for los_guidance RL pipeline
"""

import os
import sys

# ── ROS2 DDS configuration ──────────────────────────────────────────────
# FastDDS can segfault inside Docker containers. Use CycloneDDS with
# localhost-only mode for reliable operation in containerised environments.
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")

import argparse
import time
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drone_attack_env import DroneAttackEnv, make_env

# ──────────────────────────────────────────────────────────────────────────────
# Imports (with fallback for missing dependencies)
# ──────────────────────────────────────────────────────────────────────────────

try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import (
        EvalCallback,
        CheckpointCallback,
        CallbackList,
        BaseCallback,
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
    import torch

    SB3_AVAILABLE = True
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("   Install with: pip install gymnasium stable-baselines3 onnxruntime torch")
    SB3_AVAILABLE = False
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Custom Callbacks
# ──────────────────────────────────────────────────────────────────────────────

class TensorBoardCallback(BaseCallback):
    """Custom callback for logging additional metrics to TensorBoard."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Log infos from env if available
        if "episode" in self.locals.get("infos", [{}])[0]:
            info = self.locals["infos"][0]["episode"]
            self.episode_rewards.append(info["r"])
            self.episode_lengths.append(info["l"])
        return True


class HitRateCallback(BaseCallback):
    """Logs hit rate (episodes ending with 'hit' termination)."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.total_episodes = 0
        self.hits = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        for info in infos:
            if info and "terminal_info" in info:
                self.total_episodes += 1
                if info["terminal_info"].get("reason") == "hit":
                    self.hits += 1
        return True

    def get_hit_rate(self):
        if self.total_episodes == 0:
            return 0.0
        return self.hits / self.total_episodes


# ──────────────────────────────────────────────────────────────────────────────
# BC Weight Initialization
# ──────────────────────────────────────────────────────────────────────────────


def load_bc_weights_into_policy(policy, bc_weights_path: str, verbose: bool = True):
    """
    Load BC model weights into PPO policy network.

    The BC model (GuidanceMLP) has architecture: 9→128→128→64→3
    The PPO policy has architecture: 9→256→128→64→3 (pi network)

    We map compatible layers:
      - BC layer0 (9→128) → PPO pi_net[0] first 128 neurons
      - BC layer1 (128→128) → PPO pi_net[3] (256→128, copy 128→128 portion)
      - BC layer2 (128→64) → PPO pi_net[6] (128→64, direct copy)
      - BC layer3 (64→3) → PPO action_net (64→3, direct copy)

    Args:
        policy: SB3 ActorCriticPolicy instance
        bc_weights_path: Path to BC PyTorch .pth file
        verbose: Print detailed information
    """
    if not os.path.exists(bc_weights_path):
        print(f"⚠️  BC weights not found at: {bc_weights_path}")
        return False

    try:
        # Load BC state dict
        bc_state = torch.load(bc_weights_path, map_location="cpu", weights_only=True)

        # Build a mapping from BC layer names to PPO policy components
        # BC model layers (from train_mlp.py):
        #   layers.0.weight/bias → Linear(9, 128)
        #   layers.1.weight/bias → BatchNorm(128)
        #   layers.3.weight/bias → Linear(128, 128)
        #   layers.4.weight/bias → BatchNorm(128)
        #   layers.6.weight/bias → Linear(128, 64)
        #   layers.8.weight/bias → Linear(64, 3)

        # PPO policy networks:
        #   mlp_extractor.policy_net[0] = Linear(9, 256)
        #   mlp_extractor.policy_net[3] = Linear(256, 128)
        #   mlp_extractor.policy_net[6] = Linear(128, 64)
        #   action_net = Linear(64, 3)

        pi_net = policy.mlp_extractor.policy_net

        # ── Layer 0: BC Linear(9,128) → PPO Linear(9,256) ──────────────
        bc_w0 = bc_state["layers.0.weight"]  # (128, 9)
        bc_b0 = bc_state["layers.0.bias"]    # (128,)
        with torch.no_grad():
            # Copy into first 128 neurons of PPO's 256-neuron layer
            pi_net[0].weight[:128, :] = bc_w0
            pi_net[0].bias[:128] = bc_b0
            if verbose:
                print(f"   ✅ BC layer0 (9→128) → PPO pi_net[0] first 128 neurons")

        # ── Layer 1: BC Linear(128,128) → PPO Linear(256,128) ──────────
        bc_w1 = bc_state["layers.3.weight"]  # (128, 128)
        bc_b1 = bc_state["layers.3.bias"]    # (128,)
        with torch.no_grad():
            # Copy into first 128 input channels of PPO's 256→128 layer
            pi_net[3].weight[:, :128] = bc_w1
            pi_net[3].bias[:] = bc_b1
            if verbose:
                print(f"   ✅ BC layer1 (128→128) → PPO pi_net[3] first 128 input channels")

        # ── Layer 2: BC Linear(128,64) → PPO Linear(128,64) ────────────
        bc_w2 = bc_state["layers.6.weight"]  # (64, 128)
        bc_b2 = bc_state["layers.6.bias"]    # (64,)
        with torch.no_grad():
            pi_net[6].weight[:] = bc_w2
            pi_net[6].bias[:] = bc_b2
            if verbose:
                print(f"   ✅ BC layer2 (128→64) → PPO pi_net[6] (direct copy)")

        # ── Layer 3: BC Linear(64,3) → PPO action_net(64,3) ────────────
        bc_w3 = bc_state["layers.8.weight"]  # (3, 64)
        bc_b3 = bc_state["layers.8.bias"]    # (3,)
        with torch.no_grad():
            policy.action_net.weight[:] = bc_w3
            policy.action_net.bias[:] = bc_b3
            if verbose:
                print(f"   ✅ BC layer3 (64→3) → PPO action_net (direct copy)")

        if verbose:
            print(f"✅ BC weights loaded into PPO policy from: {bc_weights_path}")
        return True

    except Exception as e:
        print(f"⚠️  Failed to load BC weights: {e}")
        import traceback
        traceback.print_exc()
        return False


# ──────────────────────────────────────────────────────────────────────────────
# ONNX Model Deployment
# ──────────────────────────────────────────────────────────────────────────────


def deploy_model(
    onnx_src_path: str,
    scaler_src_path: str,
    deploy_onnx_path: str,
    deploy_scaler_path: str,
    params_yaml_path: str = None,
    verbose: bool = True,
):
    """
    Deploy the trained ONNX model to the C++ node's model directory.

    After training, this function:
    1. Copies the ONNX model to the deployment path
    2. Copies the scaler .npz to the deployment path
    3. Optionally updates params.yaml to enable neural guidance

    Args:
        onnx_src_path: Path to the trained ONNX model
        scaler_src_path: Path to the trained scaler .npz
        deploy_onnx_path: Destination path for the ONNX model (from params.yaml)
        deploy_scaler_path: Destination path for the scaler (from params.yaml)
        params_yaml_path: Path to params.yaml to auto-update (optional)
        verbose: Print detailed information
    """
    import shutil

    if verbose:
        print(f"\n📦 Deploying ONNX model...")
        print(f"   Source ONNX:  {onnx_src_path}")
        print(f"   Dest ONNX:    {deploy_onnx_path}")
        print(f"   Source Scaler: {scaler_src_path}")
        print(f"   Dest Scaler:   {deploy_scaler_path}")

    # Create destination directories
    os.makedirs(os.path.dirname(deploy_onnx_path), exist_ok=True)
    os.makedirs(os.path.dirname(deploy_scaler_path), exist_ok=True)

    # Copy ONNX model
    try:
        shutil.copy2(onnx_src_path, deploy_onnx_path)
        if verbose:
            print(f"   ✅ ONNX model deployed: {deploy_onnx_path}")
    except Exception as e:
        print(f"   ❌ Failed to copy ONNX model: {e}")
        return False

    # Copy scaler
    try:
        shutil.copy2(scaler_src_path, deploy_scaler_path)
        if verbose:
            print(f"   ✅ Scaler deployed: {deploy_scaler_path}")
    except Exception as e:
        print(f"   ❌ Failed to copy scaler: {e}")
        return False

    # Update params.yaml to enable neural guidance
    if params_yaml_path and os.path.exists(params_yaml_path):
        try:
            with open(params_yaml_path, "r") as f:
                yaml_content = f.read()

            # Update model_path and scaler_path
            import re
            yaml_content = re.sub(
                r'(model_path:\s*).*',
                f"\\1\"{deploy_onnx_path}\"",
                yaml_content,
            )
            yaml_content = re.sub(
                r'(scaler_path:\s*).*',
                f"\\1\"{deploy_scaler_path}\"",
                yaml_content,
            )
            # Set enabled: true
            yaml_content = re.sub(
                r'(enabled:\s*)false',
                f"\\1true",
                yaml_content,
            )

            with open(params_yaml_path, "w") as f:
                f.write(yaml_content)

            if verbose:
                print(f"   ✅ params.yaml updated: neural.enabled=true")
                print(f"      model_path={deploy_onnx_path}")
                print(f"      scaler_path={deploy_scaler_path}")
        except Exception as e:
            print(f"   ⚠️  Failed to update params.yaml: {e}")
            print(f"   Manual update required: set guidance.neural.enabled=true in {params_yaml_path}")
    else:
        if verbose:
            print(f"   ⚠️  params.yaml not updated (path not provided or not found)")
            print(f"   Manual update required: set guidance.neural.enabled=true")

    if verbose:
        print(f"   ✅ Deployment complete!")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Training Configuration
# ──────────────────────────────────────────────────────────────────────────────


def get_default_config():
    """Return default training configuration."""
    return {
        # Environment
        "num_envs": 4,
        "max_episode_time": 30.0,
        "hit_threshold": 1.5,
        "max_distance": 200.0,
        "control_rate_hz": 50,
        "randomize_on_reset": True,
        "target_trajectory": "random",
        # PPO hyperparameters
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "policy_kwargs": {
            "net_arch": {
                "pi": [256, 128, 64],
                "vf": [256, 128, 64],
            },
            "activation_fn": torch.nn.ReLU,
        },
        # Training
        "total_timesteps": 1_000_000,
        "eval_freq": 10000,
        "checkpoint_freq": 50000,
        "log_dir": "./logs",
        "model_dir": "./models",
        "tensorboard_log": "./logs/tb",
        # Deployment (auto-copy to C++ node)
        "deploy_onnx_path": "/simulation_ws/onnx_model/rl_drone_policy.onnx",
        "deploy_scaler_path": "/simulation_ws/onnx_model/rl_drone_policy_scaler.npz",
        "params_yaml_path": "../los_guidance/config/params.yaml",
    }

def get_quick_config():
    """Return minimal config for quick testing."""
    config = get_default_config()
    config["num_envs"] = 2
    config["total_timesteps"] = 100_000
    config["eval_freq"] = 5000
    config["checkpoint_freq"] = 25000
    config["n_steps"] = 1024
    config["batch_size"] = 32
    config["n_epochs"] = 5
    return config


# ──────────────────────────────────────────────────────────────────────────────
# Main Training Function
# ──────────────────────────────────────────────────────────────────────────────

def train(config, resume_path=None):
    """
    Run PPO training with the given configuration.

    Args:
        config: Training configuration dictionary
        resume_path: Optional path to checkpoint to resume from
    """
    # ── Setup directories ───────────────────────────────────────────────
    log_dir = config["log_dir"]
    model_dir = config["model_dir"]
    best_model_dir = os.path.join(model_dir, "best")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    # ── Create environments ─────────────────────────────────────────────
    print(f"📦 Creating {config['num_envs']} parallel environments...")

    env_fns = [
        make_env(
            rank=i,
            max_episode_time=config["max_episode_time"],
            hit_threshold=config["hit_threshold"],
            control_rate_hz=config["control_rate_hz"],
            randomize_on_reset=config["randomize_on_reset"],
            use_bc_model_as_prior=config.get("bc_model_path") is not None,
            bc_model_path=config.get("bc_model_path"),
        )
        for i in range(config["num_envs"])
    ]

    # NOTE: ROS2 nodes cannot be forked (DDS threads don't survive fork).
    # Always use DummyVecEnv (single-process sequential) for ROS2 environments.
    # For parallel training, run multiple training processes manually.
    env = DummyVecEnv(env_fns)

    # ── Evaluation environment ─────────────────────────────────────────
    eval_env = DummyVecEnv([
        make_env(
            rank=100,
            max_episode_time=config["max_episode_time"],
            hit_threshold=config["hit_threshold"],
            control_rate_hz=config["control_rate_hz"],
            randomize_on_reset=False,  # Deterministic eval
        )
    ])

    # ── Callbacks ──────────────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=log_dir,
        eval_freq=config["eval_freq"],
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=config["checkpoint_freq"],
        save_path=log_dir,
        name_prefix="rl_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    hit_rate_callback = HitRateCallback()
    tb_callback = TensorBoardCallback()

    callbacks = CallbackList([
        eval_callback,
        checkpoint_callback,
        hit_rate_callback,
        tb_callback,
    ])

    # ── Create or load model ───────────────────────────────────────────
    if resume_path and os.path.exists(resume_path):
        print(f"🔄 Resuming from checkpoint: {resume_path}")
        model = PPO.load(resume_path, env=env, tensorboard_log=config["tensorboard_log"])
    else:
        print("🚀 Creating new PPO model...")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=config["learning_rate"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
            clip_range=config["clip_range"],
            ent_coef=config["ent_coef"],
            vf_coef=config["vf_coef"],
            max_grad_norm=config["max_grad_norm"],
            policy_kwargs=config["policy_kwargs"],
            tensorboard_log=config["tensorboard_log"],
            verbose=1,
            device="auto",
        )

        # ── Load BC weights into policy (optional) ─────────────────────
        bc_weights_path = config.get("bc_weights_path")
        if bc_weights_path and os.path.exists(bc_weights_path):
            print(f"\n📦 Loading BC weights from: {bc_weights_path}")
            load_bc_weights_into_policy(model.policy, bc_weights_path)
        elif bc_weights_path:
            print(f"⚠️  BC weights not found at: {bc_weights_path}")

    # ── Train ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"🏋️  Starting training for {config['total_timesteps']:,} timesteps")
    print(f"   Envs: {config['num_envs']}, Device: {model.device}")
    print(f"   Logging to: {log_dir}")
    print(f"{'='*60}\n")

    start_time = time.time()
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callbacks,
        reset_num_timesteps=(resume_path is None),
        tb_log_name="PPO",
    )
    training_time = time.time() - start_time

    # ── Save final model ───────────────────────────────────────────────
    final_model_path = os.path.join(model_dir, "rl_drone_final.zip")
    model.save(final_model_path)
    print(f"\n✅ Final model saved to: {final_model_path}")

    # Save VecNormalize stats
    norm_path = os.path.join(model_dir, "vec_normalize.pkl")
    env.save(norm_path)
    print(f"✅ VecNormalize stats saved to: {norm_path}")

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"📊 Training Summary")
    print(f"{'='*60}")
    print(f"   Total timesteps: {config['total_timesteps']:,}")
    print(f"   Training time:   {training_time:.1f}s ({training_time/60:.1f}min)")
    print(f"   Hit rate:        {hit_rate_callback.get_hit_rate()*100:.1f}%")
    print(f"   Model saved to:  {final_model_path}")
    print(f"{'='*60}\n")

    # ── Export to ONNX ─────────────────────────────────────────────────
    try:
        from export_rl_model import export_to_onnx, save_scaler_params
        onnx_path = os.path.join(model_dir, "rl_drone_policy.onnx")
        export_to_onnx(model, onnx_path)
        print(f"✅ ONNX model exported to: {onnx_path}")

        # Also save scaler with BC-compatible keys
        scaler_npz_path = os.path.join(model_dir, "rl_drone_policy_scaler.npz")
        save_scaler_params(onnx_path, 9, 3, verbose=True)
        print(f"✅ Scaler saved to: {scaler_npz_path}")
    except ImportError:
        print("⚠️  ONNX export skipped (export_rl_model.py not found)")
    except Exception as e:
        print(f"⚠️  ONNX export failed: {e}")

    # ── Deploy to C++ node directory (optional) ────────────────────────
    deploy_onnx = config.get("deploy_onnx_path")
    deploy_scaler = config.get("deploy_scaler_path")
    params_yaml = config.get("params_yaml_path")
    if deploy_onnx and deploy_scaler:
        print(f"\n📦 Deploying model to C++ node...")
        deploy_model(
            onnx_src_path=onnx_path,
            scaler_src_path=os.path.join(model_dir, "rl_drone_policy_scaler.npz"),
            deploy_onnx_path=deploy_onnx,
            deploy_scaler_path=deploy_scaler,
            params_yaml_path=params_yaml,
        )
    else:
        print(f"\n⚠️  Auto-deploy disabled. Set --deploy-dir or deploy config.")
        print(f"   Manual deploy: copy {onnx_path} → {config.get('deploy_onnx_path', '?')}")

    # Cleanup
    env.close()
    eval_env.close()

    return model


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RL training for drone attack")
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Total timesteps (overrides default)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--num-envs", type=int, default=None,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--bc-weights", type=str, default=None,
        help="Path to BC model weights for policy initialisation"
    )
    parser.add_argument(
        "--bc-model", type=str, default=None,
        help="Path to BC ONNX model for action prior"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test mode (100k timesteps, 2 envs)"
    )
    parser.add_argument(
        "--deploy-dir", type=str, default=None,
        help="Deploy ONNX model to this directory (e.g., /home/kopty/Загрузки/onnx_model/)"
    )
    parser.add_argument(
        "--no-deploy", action="store_true",
        help="Skip auto-deployment even if deploy paths are configured"
    )
    args = parser.parse_args()

    # Select config
    if args.quick:
        config = get_quick_config()
        print("⚡ Quick test mode: 100k timesteps, 2 envs")
    else:
        config = get_default_config()

    # Override from CLI
    if args.timesteps is not None:
        config["total_timesteps"] = args.timesteps
    if args.num_envs is not None:
        config["num_envs"] = args.num_envs
    if args.bc_model is not None:
        config["bc_model_path"] = args.bc_model
    if args.bc_weights is not None:
        config["bc_weights_path"] = args.bc_weights

    # Handle deployment
    if args.no_deploy:
        config["deploy_onnx_path"] = None
        config["deploy_scaler_path"] = None
        print("   Auto-deploy disabled (--no-deploy)")
    elif args.deploy_dir is not None:
        config["deploy_onnx_path"] = os.path.join(args.deploy_dir, "rl_drone_policy.onnx")
        config["deploy_scaler_path"] = os.path.join(args.deploy_dir, "scaler_params.npz")
        print(f"   Deploy target: {args.deploy_dir}")

    # Device detection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # Run training
    train(config, resume_path=args.resume)


if __name__ == "__main__":
    main()
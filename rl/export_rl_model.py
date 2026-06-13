#!/usr/bin/env python3
"""
export_rl_model.py — Export trained RL model to ONNX format.

This script:
1. Loads a trained Stable-Baselines3 PPO model
2. Exports the policy network to ONNX format
3. Saves the observation scaler parameters
4. Validates the exported model by comparing outputs

The exported ONNX model can be deployed in C++ for real-time inference
in the los_guidance ROS2 node.

Usage:
    python3 export_rl_model.py --model ./models/rl_drone_final.zip
    python3 export_rl_model.py --model ./logs/best_model/best_model.zip --output ./models/rl_drone.onnx

Author: Generated for los_guidance RL pipeline
"""

import argparse
import os
import sys
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    from stable_baselines3 import PPO
    from stable_baselines3.common.policies import ActorCriticPolicy

    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"❌ Missing PyTorch/SB3: {e}")
    TORCH_AVAILABLE = False

try:
    import onnx
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# ONNX Export
# ──────────────────────────────────────────────────────────────────────────────

class PolicyWrapper(nn.Module):
    """
    Wrapper around SB3 policy network for ONNX export.

    Extracts the action distribution parameters (mean action) from the
    policy network, ignoring the value function.
    """

    def __init__(self, policy: ActorCriticPolicy):
        super().__init__()
        # Extract the policy network (actor)
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net

        # Get observation dimensions
        self.obs_dim = policy.observation_space.shape[0]
        self.action_dim = policy.action_space.shape[0]

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: observation → mean action.

        Args:
            observation: (batch_size, obs_dim) tensor

        Returns:
            mean_action: (batch_size, action_dim) tensor
        """
        # Extract features
        latent_pi, _ = self.mlp_extractor(observation)

        # Get mean action
        mean_action = self.action_net(latent_pi)

        return mean_action


def export_to_onnx(
    model_path: str,
    output_path: str,
    obs_dim: int = 9,
    action_dim: int = 3,
    verbose: bool = True,
):
    """
    Export trained SB3 PPO model to ONNX format.

    The exported model matches the BC model (guidance_mlp.onnx) interface:
      Input:  state['batch', 9]  — [u_los_xyz, lambda_xyz, roll, pitch, yaw]
      Output: action['batch', 3] — [acc_x, acc_y, acc_z] (body frame)

    Args:
        model_path: Path to SB3 model (.zip)
        output_path: Output ONNX file path
        obs_dim: Observation dimension (default: 9)
        action_dim: Action dimension (default: 3)
        verbose: Print detailed information
    """
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available. Cannot export model.")
        return False

    if verbose:
        print(f"📦 Loading model from: {model_path}")

    # ── Load model ─────────────────────────────────────────────────────
    try:
        model = PPO.load(model_path, device="cpu")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

    if verbose:
        print(f"   Policy architecture:")
        print(f"   {model.policy}")

    # ── Create wrapper ─────────────────────────────────────────────────
    policy_wrapper = PolicyWrapper(model.policy)
    policy_wrapper.eval()

    # ── Create dummy input ─────────────────────────────────────────────
    dummy_input = torch.randn(1, obs_dim, dtype=torch.float32)

    # ── Export to ONNX ─────────────────────────────────────────────────
    if verbose:
        print(f"\n🔄 Exporting to ONNX: {output_path}")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    try:
        torch.onnx.export(
            policy_wrapper,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["state"],       # Match BC model input name
            output_names=["action"],     # Match BC model output name
            dynamic_axes={
                "state": {0: "batch_size"},
                "action": {0: "batch_size"},
            },
            verbose=False,
        )
        print(f"✅ ONNX model saved to: {output_path}")
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return False

    # ── Validate ONNX model ────────────────────────────────────────────
    if ONNX_AVAILABLE:
        validate_onnx(model_path, output_path, policy_wrapper, dummy_input, verbose)
    else:
        print("   ⚠️  ONNX validation skipped (onnxruntime not installed)")

    # ── Save scaler parameters ─────────────────────────────────────────
    save_scaler_params(output_path, obs_dim, action_dim, verbose)

    return True


def validate_onnx(
    model_path: str,
    onnx_path: str,
    policy_wrapper: PolicyWrapper,
    dummy_input: torch.Tensor,
    verbose: bool = True,
):
    """
    Validate ONNX model by comparing outputs with PyTorch model.

    Args:
        model_path: Path to original SB3 model
        onnx_path: Path to exported ONNX model
        policy_wrapper: PyTorch policy wrapper
        dummy_input: Dummy input tensor
        verbose: Print detailed information
    """
    if verbose:
        print("\n🔍 Validating ONNX model...")

    # ── PyTorch inference ──────────────────────────────────────────────
    with torch.no_grad():
        torch_output = policy_wrapper(dummy_input).numpy()

    # ── ONNX inference ─────────────────────────────────────────────────
    try:
        ort_session = ort.InferenceSession(onnx_path)
        input_name = ort_session.get_inputs()[0].name
        onnx_output = ort_session.run(None, {input_name: dummy_input.numpy()})[0]
    except Exception as e:
        print(f"❌ ONNX inference failed: {e}")
        return

    # ── Compare ────────────────────────────────────────────────────────
    max_diff = np.max(np.abs(torch_output - onnx_output))
    mean_diff = np.mean(np.abs(torch_output - onnx_output))

    if verbose:
        print(f"   PyTorch output: {torch_output.flatten()}")
        print(f"   ONNX output:    {onnx_output.flatten()}")
        print(f"   Max difference:  {max_diff:.6e}")
        print(f"   Mean difference: {mean_diff:.6e}")

    if max_diff < 1e-4:
        print("✅ ONNX validation PASSED (outputs match)")
    else:
        print(f"⚠️  ONNX validation: max diff = {max_diff:.6e} (may be acceptable)")


def save_scaler_params(
    onnx_path: str,
    obs_dim: int,
    action_dim: int,
    verbose: bool = True,
):
    """
    Save observation scaler parameters alongside the ONNX model.

    Creates a .npz file with BC-compatible keys: X_mean, X_scale, y_max.
    This matches the format used by guidance_mlp.onnx's scaler_params.npz.

    Args:
        onnx_path: Path to ONNX model
        obs_dim: Observation dimension
        action_dim: Action dimension
        verbose: Print detailed information
    """
    base_dir = os.path.dirname(onnx_path)
    base_name = os.path.splitext(os.path.basename(onnx_path))[0]

    # Default scaler: identity (no normalisation)
    # In practice, compute these from training data statistics
    X_mean = np.zeros(obs_dim, dtype=np.float32)
    X_scale = np.ones(obs_dim, dtype=np.float32)
    y_max = np.ones(action_dim, dtype=np.float32)

    scaler_path = os.path.join(base_dir, f"{base_name}_scaler.npz")
    np.savez_compressed(scaler_path, X_mean=X_mean, X_scale=X_scale, y_max=y_max)

    if verbose:
        print(f"\n📊 Scaler saved to: {scaler_path}")
        print(f"   X_mean: {X_mean}")
        print(f"   X_scale: {X_scale}")
        print(f"   y_max: {y_max}")

    # Also save as text for easy inspection
    txt_path = os.path.join(base_dir, f"{base_name}_scaler.txt")
    with open(txt_path, "w") as f:
        f.write(f"# Scaler parameters for {base_name}.onnx\n")
        f.write(f"# obs_dim={obs_dim}, action_dim={action_dim}\n")
        f.write(f"X_mean={','.join(f'{v:.6f}' for v in X_mean)}\n")
        f.write(f"X_scale={','.join(f'{v:.6f}' for v in X_scale)}\n")
        f.write(f"y_max={','.join(f'{v:.6f}' for v in y_max)}\n")

    if verbose:
        print(f"   Text scaler saved to: {txt_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export trained RL model to ONNX format"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to trained SB3 model (.zip)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output ONNX file path"
    )
    parser.add_argument(
        "--obs-dim", type=int, default=9,
        help="Observation dimension (default: 9)"
    )
    parser.add_argument(
        "--action-dim", type=int, default=3,
        help="Action dimension (default: 3)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress detailed output"
    )

    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base = os.path.splitext(os.path.basename(args.model))[0]
        output_path = os.path.join(
            os.path.dirname(args.model) or ".",
            f"{base}.onnx",
        )

    # Ensure .onnx extension
    if not output_path.endswith(".onnx"):
        output_path += ".onnx"

    # Export
    success = export_to_onnx(
        model_path=args.model,
        output_path=output_path,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        verbose=not args.quiet,
    )

    if success:
        print(f"\n🎉 Export complete!")
        print(f"   ONNX model: {output_path}")
        print(f"   Use in C++ with onnxruntime or libtorch")
    else:
        print("\n❌ Export failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
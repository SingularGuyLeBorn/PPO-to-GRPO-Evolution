"""
PPO vs GRPO: Benchmark Comparison

This script compares the key differences between PPO and GRPO
in terms of architecture, compute requirements, and memory usage.
"""
import torch
import torch.nn as nn


def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_memory(model, batch_size=1, seq_len=128):
    """Estimate memory usage for a forward pass (in MB)"""
    param_memory = count_parameters(model) * 4 / (1024 ** 2)  # float32
    # Rough estimate for activations
    act_memory = batch_size * seq_len * 768 * 4 / (1024 ** 2)
    return param_memory + act_memory


def compare_ppo_vs_grpo():
    """
    Compare PPO vs GRPO architecture requirements

    PPO: Policy + Value (Critic) + Reward Model = 3 models
    GRPO: Policy + Reward Model = 2 models (no critic)
    """
    print("=" * 70)
    print("PPO vs GRPO: Architecture & Resource Comparison")
    print("=" * 70)

    # Model size assumptions (based on 7B-scale LLM)
    base_params = 7_000_000_000  # 7B parameters

    # PPO requires 3 models
    ppo_policy = base_params
    ppo_critic = base_params  # Value network (same size)
    ppo_reward = base_params * 0.5  # Reward model (smaller)
    ppo_total = ppo_policy + ppo_critic + ppo_reward

    # GRPO requires 2 models
    grpo_policy = base_params
    grpo_reward = base_params * 0.5
    grpo_total = grpo_policy + grpo_reward

    print(f"\n{'Metric':<40} {'PPO':<15} {'GRPO':<15}")
    print("-" * 70)
    print(f"{'Number of Models':<40} {'3 (P+V+R)':<15} {'2 (P+R)':<15}")
    print(
        f"{'Total Parameters':<40} "
        f"{ppo_total/1e9:<15.1f}B "
        f"{grpo_total/1e9:<15.1f}B"
    )
    print(
        f"{'Saved Parameters':<40} {'-':<15} "
        f"{(ppo_total - grpo_total)/1e9:<15.1f}B ({(ppo_total - grpo_total)/ppo_total*100:.0f}%)"
    )

    # Memory comparison (fp16)
    mem_per_model = 2 * base_params / (1024 ** 3)  # ~13 GB for 7B in fp16
    ref_mem = 2 * base_params / (1024 ** 3)  # Reference model for GRPO

    print(f"\n{'Memory (fp16, approx)':<40} {'PPO':<15} {'GRPO':<15}")
    print("-" * 70)
    print(
        f"{'Training Memory':<40} "
        f"{ppo_total * 2 / (1024**3):<15.1f}GB "
        f"{(grpo_total + base_params) * 2 / (1024**3):<15.1f}GB"
    )
    print(
        f"{'Peak Memory Saving':<40} {'-':<15} "
        f"{(ppo_total - grpo_total - base_params) * 2 / (1024**3):<15.1f}GB"
    )

    # Training speed comparison
    print(f"\n{'Training Speed (relative)':<40} {'PPO':<15} {'GRPO':<15}")
    print("-" * 70)
    print(f"{'Forward/Backward Passes':<40} {'3 per step':<15} {'2 per step + ref':<15}")
    print(f"{'Estimated Speedup':<40} {'1.0x (baseline)':<15} {'~1.3x':<15}")

    print("\n" + "=" * 70)
    print("Key Takeaway: GRPO achieves ~30% compute savings by")
    print("eliminating the value network, while maintaining or")
    print("improving training stability through group-relative rewards.")
    print("=" * 70)


if __name__ == "__main__":
    compare_ppo_vs_grpo()

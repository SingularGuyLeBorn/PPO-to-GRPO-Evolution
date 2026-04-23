"""
GRPO Example: Training for LLM alignment (simplified demonstration)
"""
import torch
import torch.nn as nn
import numpy as np
from grpo_agent import PolicyNetwork, GRPOTrainer


def create_mock_data(batch_size=4, group_size=8, vocab_size=1000, seq_len=32):
    """Create mock training data for demonstration"""
    prompts = torch.randint(0, vocab_size, (batch_size, 10))
    responses = torch.randint(0, vocab_size, (batch_size * group_size, seq_len))
    old_log_probs = torch.randn(batch_size * group_size, seq_len) * 0.1
    rewards = torch.randn(batch_size * group_size) * 0.5 + 0.3
    return prompts, responses, old_log_probs, rewards


def main():
    vocab_size = 1000
    embed_dim = 128
    hidden_dim = 256
    group_size = 8
    batch_size = 4

    # Initialize policy and reference networks
    policy = PolicyNetwork(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
    )
    ref_policy = PolicyNetwork(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
    )

    # Copy weights to reference (in practice, ref is a frozen checkpoint)
    ref_policy.load_state_dict(policy.state_dict())

    # Initialize GRPO trainer
    trainer = GRPOTrainer(
        policy_net=policy,
        ref_policy_net=ref_policy,
        lr=1e-5,
        clip_eps=0.2,
        beta=0.04,
        group_size=group_size,
    )

    # Training loop
    num_steps = 50
    for step in range(num_steps):
        prompts, responses, old_log_probs, rewards = create_mock_data(
            batch_size=batch_size,
            group_size=group_size,
            vocab_size=vocab_size,
        )

        metrics = trainer.train_step(
            prompt_ids=prompts,
            response_ids=responses,
            old_log_probs=old_log_probs,
            rewards=rewards,
        )

        if (step + 1) % 10 == 0:
            print(
                f"Step {step+1:3d} | Loss: {metrics['loss']:.4f} | "
                f"Policy Loss: {metrics['policy_loss']:.4f} | "
                f"KL Loss: {metrics['kl_loss']:.4f}"
            )

    print("GRPO training completed!")


if __name__ == "__main__":
    main()

"""
GRPO (Group Relative Policy Optimization) Implementation
Reference: DeepSeek 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PolicyNetwork(nn.Module):
    """Policy network for GRPO (no critic needed!)"""

    def __init__(self, vocab_size, embed_dim=768, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                batch_first=True,
            ),
            num_layers=6,
        )
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        if attention_mask is not None:
            x = self.transformer(x, src_key_padding_mask=~attention_mask.bool())
        else:
            x = self.transformer(x)
        logits = self.lm_head(x)
        return logits

    def get_log_prob(self, input_ids, target_ids, attention_mask=None):
        logits = self.forward(input_ids, attention_mask)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)


class GRPOTrainer:
    """
    GRPO Trainer: Group Relative Policy Optimization

    Key innovation: Replaces the value network with group-relative
    advantage estimation, making it value-free.
    """

    def __init__(
        self,
        policy_net,
        ref_policy_net,
        lr=1e-6,
        clip_eps=0.2,
        beta=0.04,
        group_size=8,
    ):
        self.policy = policy_net
        self.ref_policy = ref_policy_net
        self.ref_policy.eval()  # Freeze reference policy
        self.clip_eps = clip_eps
        self.beta = beta
        self.group_size = group_size
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def compute_group_advantage(self, rewards):
        """
        Compute group-relative advantages using z-score normalization

        Args:
            rewards: Tensor of shape [batch_size * group_size]
                    containing rewards for each response

        Returns:
            advantages: Tensor of shape [batch_size * group_size]
        """
        rewards = rewards.view(-1, self.group_size)
        mean = rewards.mean(dim=-1, keepdim=True)
        std = rewards.std(dim=-1, keepdim=True) + 1e-8
        advantages = (rewards - mean) / std
        return advantages.flatten()

    def compute_kl_divergence(self, log_probs, ref_log_probs):
        """
        Compute KL divergence between current policy and reference policy

        D_KL(pi_theta || pi_ref) = E[log(pi_theta/pi_ref)]
        """
        return (log_probs - ref_log_probs).mean()

    def train_step(
        self,
        prompt_ids,
        response_ids,
        old_log_probs,
        rewards,
        attention_mask=None,
    ):
        """
        Single GRPO training step

        Args:
            prompt_ids: Input prompts [batch_size, prompt_len]
            response_ids: Model responses [batch_size * group_size, response_len]
            old_log_probs: Old log probabilities [batch_size * group_size, response_len]
            rewards: Reward scores [batch_size * group_size]
            attention_mask: Optional attention mask

        Returns:
            loss: Training loss value
        """
        # Compute group-relative advantages
        advantages = self.compute_group_advantage(rewards)

        # Get current policy log probabilities
        log_probs = self.policy.get_log_prob(
            prompt_ids, response_ids, attention_mask
        )

        # Get reference policy log probabilities
        with torch.no_grad():
            ref_log_probs = self.ref_policy.get_log_prob(
                prompt_ids, response_ids, attention_mask
            )

        # Importance sampling ratio (token-level)
        ratio = torch.exp(log_probs - old_log_probs)

        # Clipped surrogate objective (token-level)
        surr1 = ratio * advantages.unsqueeze(-1)
        surr2 = (
            torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
            * advantages.unsqueeze(-1)
        )
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL divergence regularization
        kl_loss = self.compute_kl_divergence(log_probs, ref_log_probs)

        # GRPO total loss
        loss = policy_loss + self.beta * kl_loss

        # Update policy
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
        }

    def generate_group_samples(self, policy, prompts, num_generations=8):
        """
        Generate a group of responses for each prompt

        This simulates the group sampling process in GRPO
        """
        all_responses = []
        for prompt in prompts:
            group_responses = []
            for _ in range(num_generations):
                # Simplified generation (replace with actual sampling)
                response = policy.generate(prompt, max_length=128)
                group_responses.append(response)
            all_responses.extend(group_responses)
        return all_responses

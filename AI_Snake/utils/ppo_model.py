"""Neural network architectures for PPO."""

import torch
import torch.nn as nn
from typing import Tuple


class ActorCritic(nn.Module):
    """Shared backbone with separate policy and value heads."""

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        features = self.feature_extractor(x)
        logits = self.policy_head(features)
        values = self.value_head(features)
        distribution = torch.distributions.Categorical(logits=logits)
        return distribution, values

"""Multimodality Behavior cloning policy"""

import numpy as np
from typing import Dict, Tuple
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import torch.nn.functional as F
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.control_utils.mmmlp import MMMLP
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy


class MBCLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, horizon, obs_dim, action_dim, n_action_steps, n_obs_steps, n_max_modality, **kwargs):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        # Assemble model; input n_obs_steps, predict n_action_steps
        self.model = MMMLP(x_dim=obs_dim * n_obs_steps, y_dim=action_dim * horizon, k=n_max_modality)
        self.normalizer = LinearNormalizer()
        self.bias_factor = 10.0

    # ========= inference  ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert "obs" in obs_dict
        assert "past_action" not in obs_dict

        nobs = self.normalizer["obs"].normalize(obs_dict["obs"])
        nobs = nobs[:, : self.n_obs_steps, :]

        # Predict using MMMLP
        action_pred, p_pred = self.model(nobs.view(nobs.shape[0], -1))  # (B, k * action_dim), (B, k)
        p_pred = F.softmax(p_pred, dim=-1)

        # Un-normalize
        action_pred = action_pred.reshape(action_pred.shape[0], self.model.k * self.horizon, self.action_dim)
        action_pred = self.normalizer["action"].unnormalize(action_pred)
        action_pred = action_pred.reshape(action_pred.shape[0], self.model.k, self.horizon, self.action_dim)
        # Randomly select one for each batch for k modalities if p > p_thresh
        p_thresh = 0.5 * (1.0 / self.model.k)
        # If p_pred is smaller than p_thresh, set it to 0
        p_pred = torch.where(p_pred < p_thresh, torch.zeros_like(p_pred), p_pred)

        # Sample actions using p_pred
        action_sample_list = []
        for i in range(action_pred.shape[0]):
            action_sample = action_pred[i, torch.multinomial(p_pred[i], 1).item(), ...]
            action_sample_list.append(action_sample)
        action_pred = torch.stack(action_sample_list, dim=0)
        result = {"action": action_pred, "action_pred": action_pred, "p_pred": p_pred}
        return result

    # ========= training  ============
    def update_training_meta(self, epoch: int, **kwargs):
        # increase bias factor with time
        self.bias_factor = np.clip(np.sqrt(epoch) * 10.0, a_min=10.0, a_max=100.0)

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        assert "valid_mask" not in batch
        nbatch = self.normalizer.normalize(batch)
        nobs = nbatch["obs"][:, : self.n_obs_steps, :]
        nactions = nbatch["action"][:, : self.horizon, :]
        batch_size = nactions.shape[0]
        loss = self.model.criterion(nobs.view(batch_size, -1), nactions.view(batch_size, -1), lambda_=self.bias_factor)
        return loss

import numpy as np
import matplotlib.pyplot as plt
import numba
from copy import deepcopy


########################################## Common ##########################################
@numba.jit(nopython=True)
def create_indices(episode_ends: np.ndarray, sequence_length: int, episode_mask: np.ndarray, pad_before: int = 0, pad_after: int = 0, debug: bool = True) -> np.ndarray:
    episode_mask.shape == episode_ends.shape
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)

    indices = list()
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            # skip episode
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert start_offset >= 0
                assert end_offset >= 0
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


########################################## Dataset part ##########################################
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Union


class SequenceDataset(Dataset):
    def __init__(self, episodes, sequence_length: int = 8, pad_before: int = 1, pad_after: int = 7) -> None:
        super().__init__()
        self.episodes = episodes
        self.states = []
        self.actions = []
        self.labels = []
        self.epsiode_ends = []
        for episode in episodes:
            self.states.append(episode["state"])
            self.actions.append(episode["action"])
            self.labels.append(episode["label"])
            self.epsiode_ends.append(episode["state"].shape[0])
        self.states = np.concatenate(self.states, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)
        self.labels = np.array(self.labels)
        self.epsiode_ends = np.cumsum(self.epsiode_ends)
        # Generate sample index
        episode_mask = np.ones(self.epsiode_ends.shape, dtype=bool)
        self.sequence_length = sequence_length
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.indices = create_indices(self.epsiode_ends, sequence_length=sequence_length, episode_mask=episode_mask, pad_before=pad_before, pad_after=pad_after)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = idx % len(self.indices)
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        result = dict()
        sample_states = self.states[buffer_start_idx:buffer_end_idx]
        sample_actions = self.actions[buffer_start_idx:buffer_end_idx]
        if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
            result["state"] = np.zeros([self.sequence_length, sample_states.shape[1]])
            result["action"] = np.zeros([self.sequence_length, sample_actions.shape[1]])
            if sample_start_idx > 0:
                result["state"][:sample_start_idx] = sample_states[0]
                result["action"][:sample_start_idx] = sample_actions[0]
            if sample_end_idx < self.sequence_length:
                result["state"][sample_end_idx:] = sample_states[-1]
                result["action"][sample_end_idx:] = sample_actions[-1]
        else:
            result["state"] = sample_states
            result["action"] = sample_actions
        return result

    def all_state_actions(self):
        # Generate all state action pairs
        state_list = []
        action_list = []
        for idx in range(len(self)):
            data = self[idx]
            state_list.append(data["state"])
            action_list.append(data["action"])
        return np.stack(state_list), np.stack(action_list)


########################################## Algorithm ##########################################
class DiffusionKernelRegression:
    """Estimated diffusion kernel regression."""

    def __init__(
        self,
        states,
        actions,
        state_weights,
        action_weights,
        horizon: int = 8,
        n_obs_steps: int = 2,
        n_act_steps: int = 4,
        knn_max: int = 30,
        diffusion_steps: int = 100,
        scheduler_type: str = "linear",
        **kwargs
    ):
        self.states = states
        self.actions = actions
        self.state_weights = state_weights
        self.action_weights = action_weights
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_act_steps = n_act_steps
        self.diffusion_steps = diffusion_steps
        self.knn_max = knn_max
        self.alpha_t, self.beta_t, self.alpha_bar_t, self.sigma_t, self.h_t = self.get_scheduler(scheduler_type=scheduler_type)
        self.do_clip = True
        # Normalization
        self.stats = {
            "state": {
                "max": np.max(np.reshape(self.states, [-1, self.states.shape[-1]]), axis=0),
                "min": np.min(np.reshape(self.states, [-1, self.states.shape[-1]]), axis=0),
            },
            "action": {
                "max": np.max(np.reshape(self.actions, [-1, self.actions.shape[-1]]), axis=0),
                "min": np.min(np.reshape(self.actions, [-1, self.actions.shape[-1]]), axis=0),
            },
        }

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], **kwargs):
        batch_size = kwargs.get("batch_size", 4)
        state = obs_dict["state"].cpu().numpy()
        if state.ndim == 2:
            state = state[-1, :]
        local_states, local_actions = self.compute_neighbors(state, self.knn_max)
        # Normalize the states and actions
        local_states = ((local_states - self.stats["state"]["min"]) / (self.stats["state"]["max"] - self.stats["state"]["min"])) * 2 - 1
        local_actions = ((local_actions - self.stats["action"]["min"]) / (self.stats["action"]["max"] - self.stats["action"]["min"])) * 2 - 1
        state = ((state - self.stats["state"]["min"]) / (self.stats["state"]["max"] - self.stats["state"]["min"])) * 2 - 1
        # Apply the weights
        weighted_local_states = local_states * self.state_weights[None, :]
        weighted_local_actions = (local_actions * self.action_weights[None, None, :]).reshape(-1, self.n_act_steps * self.actions.shape[-1])
        weighted_state = (state * self.state_weights).reshape(1, -1)

        action_samples = []
        for _ in range(batch_size):
            action_sample = np.random.randn(1, self.n_act_steps * self.actions.shape[-1])
            for i in range(self.diffusion_steps - 1, -1, -1):
                # Compute the gradient of the state
                state_diff = weighted_state - weighted_local_states
                action_diff = action_sample - weighted_local_actions
                # Compute the diffusion kernel
                kernel = np.exp(-(np.linalg.norm(state_diff, axis=1) ** 2 + np.linalg.norm(action_diff, axis=1) ** 2) / (2 * self.h_t[i] ** 2))
                kernel = np.clip(kernel, 1e-8, 1)
                # FIXME: When sigma is too small, the kernel will be around 0, which will cause nan in the action predictions
                # Compute the action prediction: x_0 = sum_i (k_i * y_i) / sum_i (k_i)
                action_pred = np.sum(kernel[:, None] * weighted_local_actions, axis=0) / np.sum(kernel)
                if i > 0:
                    # Update the step
                    action_sample = (
                        np.sqrt(self.alpha_t[i]) * (1 - self.alpha_bar_t[i - 1]) / (1 - self.alpha_bar_t[i]) * action_sample
                        + np.sqrt(self.alpha_bar_t[i - 1]) * self.beta_t[i] / (1 - self.alpha_bar_t[i]) * action_pred
                    )
                    # Add noise
                    action_sample += self.sigma_t[i - 1] * np.random.randn(1, self.n_act_steps * self.actions.shape[-1])
                if self.do_clip:
                    action_sample = np.clip(action_sample, -1, 1)
            # Unnormalize the action
            action_sample = action_sample.reshape(self.n_act_steps, self.actions.shape[-1])
            action_sample = (action_sample + 1) / 2  # Normalize to [0, 1]
            action_sample = action_sample * (self.stats["action"]["max"] - self.stats["action"]["min"]) + self.stats["action"]["min"]
            action_samples.append(action_sample)
        return np.stack(action_samples, axis=0)

    def compute_neighbors(self, state, knn_max):
        # Compute neighbors based on the state
        states_key = self.states[:, -1, :]
        weighted_states_key = states_key * self.state_weights[None, :]
        weighted_state = state * self.state_weights
        dist = np.linalg.norm(weighted_states_key - weighted_state[None, :], axis=1)
        idx = np.argsort(dist)[:knn_max]
        return self.states[idx, -1, :], self.actions[idx]

    def get_scheduler(self, beta_start=0.0001, beta_end=0.02, scheduler_type="linear"):
        # Get the scheduler for the diffusion steps; alpha_t, beta_t
        t = np.arange(self.diffusion_steps)
        if scheduler_type == "linear":
            beta_t = np.linspace(beta_start, beta_end, self.diffusion_steps)
        elif scheduler_type == "quadratic":
            beta_t = np.linspace(beta_start**0.5, beta_end**0.5, self.diffusion_steps) ** 2
        else:
            raise ValueError("Invalid scheduler type")
        alpha_t = 1 - beta_t
        alpha_bar_t = np.cumprod(alpha_t)
        sigma_t = (1 - alpha_bar_t[1:]) / (1 - alpha_bar_t[:-1]) * beta_t[1:]
        h_t = np.sqrt((1 - alpha_bar_t) / alpha_bar_t)
        return alpha_t, beta_t, alpha_bar_t, sigma_t, h_t

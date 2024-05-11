from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset


class PushTStateDataset(BaseLowdimDataset):
    """PushT dataset; using full state"""

    def __init__(self, zarr_path, horizon=1, pad_before=0, pad_after=0, state_key="state", action_key="action", seed=42, val_ratio=0.0, max_train_episodes=None, return_next_state=False):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=[state_key, action_key])
        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        pad_after = pad_after if not return_next_state else pad_after + 1  # add one more step for next state
        self.sampler = SequenceSampler(replay_buffer=self.replay_buffer, sequence_length=horizon, pad_before=pad_before, pad_after=pad_after, episode_mask=train_mask)
        self.action_key = action_key
        self.state_key = state_key
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(replay_buffer=self.replay_buffer, sequence_length=self.horizon, pad_before=self.pad_before, pad_after=self.pad_after, episode_mask=~self.train_mask)
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = self._sample_to_data(self.replay_buffer)
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        state = sample[self.state_key]
        obs = np.array(state)
        obs_next = np.roll(obs, -1, axis=0)
        data = {
            "obs": obs[:-1, ...],  # T, D_o
            "obs_next": obs_next[:-1, ...],  # T, D_o
            "action": sample[self.action_key],  # T, D_a
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

"""Pure data driven policy; Frequence policy."""

import numpy as np
import torch
from typing import Dict, Union
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.policy.base_sa_policy import BaseSAPolicy

###############################  Action Policy #################################
class KNNPolicy(BaseImagePolicy):
    """KNN Policy field."""

    def __init__(self, zarr_path, horizon=1, pad_before=0, pad_after=0, kernel=None, knn=4):
        print("Start loading dataset to memory...")
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=["img", "state", "action", "control", "demo_type"])
        print("Dataset loaded.")
        self.sampler = SequenceSampler(replay_buffer=self.replay_buffer, sequence_length=horizon, pad_before=pad_before, pad_after=pad_after)
        self.states = self.replay_buffer["state"]
        # Emphasize the last dim of states
        self.states[..., -1] = self.states[..., -1] / np.pi * 200
        episode_lengths = self.replay_buffer.episode_lengths
        self.episode_ids = [[i] * l for i, l in enumerate(episode_lengths)]
        self.episode_ids = np.stack(sum(self.episode_ids, [])).reshape(-1, 1)
        self.step_ids = [list(range(l)) for l in episode_lengths]
        self.step_ids = np.stack(sum(self.step_ids, [])).reshape(-1, 1)
        # param
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.zarr_path = zarr_path
        self.kernel = kernel
        self.knn = knn

    @property
    def device(self):
        # Override property
        return torch.device("cpu")

    @property
    def dtype(self):
        # Override property
        return torch.float32
    
    @property
    def n_act_steps(self):
        return self.pad_after + 1
    
    @property
    def n_obs_steps(self):
        return self.pad_before + 1

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], **kwargs):
        """Randomly choose an action from the kth closest state in the replay buffer.
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert "past_action" not in obs_dict  # not implemented yet
        # read state
        info = kwargs["info"]
        if isinstance(info, dict):
            infos = [info]
        elif isinstance(info, list):
            infos = info
        elif isinstance(info, tuple):
            infos = list(info)
        else:
            raise ValueError("Invalid info type.")
        action_list = []
        for info in infos:
            pos_agent = info["pos_agent"]
            block_pose = info["block_pose"]
            if pos_agent.ndim == 2 and block_pose.ndim == 2:
                pos_agent = pos_agent[-1]
                block_pose = block_pose[-1]
            state = np.concatenate([pos_agent, block_pose])
            state[-1] = state[-1] / np.pi * 200  # emphasize the last dim
            # Find the closest state in the current episode
            distances = np.linalg.norm(self.states - state, axis=1)
            # Nearest k neighbors
            closest_state_idx = np.argsort(distances)[: self.knn]
            # Randomly choose one of the closest states
            closest_state_idx = np.random.choice(closest_state_idx)
            closest_episode_id = self.episode_ids[closest_state_idx].item()
            # Find the closest action
            closest_step_id = self.step_ids[closest_state_idx].item()
            closest_episode = self.replay_buffer.get_episode(closest_episode_id)
            closest_actions = closest_episode["action"]
            closest_actions = closest_actions[closest_step_id:]
            # Crop or pad to horizon
            if len(closest_actions) < self.horizon:
                closest_actions = np.pad(closest_actions, ((0, self.horizon - len(closest_actions)), (0, 0)), mode="edge")
            elif len(closest_actions) > self.horizon:
                closest_actions = closest_actions[: self.horizon]
            closest_actions = closest_actions[None, ...]
            action_list.append(closest_actions)
        actions = np.concatenate(action_list, axis=0)
        return {"action": torch.from_numpy(actions).to(device=self.device, dtype=self.dtype)}


class KNNSAPolicy(KNNPolicy, BaseSAPolicy):
    def __init__(self, zarr_path, horizon=1, pad_before=0, pad_after=0, kernel=None, knn=4):
        super().__init__(zarr_path, horizon, pad_before, pad_after, kernel, knn)

    def predict_state_action(self, obs_dict: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        assert "past_action" not in obs_dict  # not implemented yet
        # read state
        info = kwargs["info"]
        if isinstance(info, dict):
            infos = [info]
        elif isinstance(info, list):
            infos = info
        elif isinstance(info, tuple):
            infos = list(info)
        else:
            raise ValueError("Invalid info type.")
        action_list = []
        state_list = []
        for info in infos:
            pos_agent = info["pos_agent"]
            block_pose = info["block_pose"]
            if pos_agent.ndim == 2 and block_pose.ndim == 2:
                pos_agent = pos_agent[-1]
                block_pose = block_pose[-1]
            state = np.concatenate([pos_agent, block_pose])
            state[-1] = state[-1] / np.pi * 200  # emphasize the last dim
            # Find the closest state in the current episode
            distances = np.linalg.norm(self.states - state, axis=1)
            # Nearest k neighbors
            closest_state_idx = np.argsort(distances)[: self.knn]
            # Randomly choose one of the closest states
            closest_state_idx = np.random.choice(closest_state_idx)
            closest_episode_id = self.episode_ids[closest_state_idx].item()
            # Find the closest action
            closest_step_id = self.step_ids[closest_state_idx].item()
            closest_episode = self.replay_buffer.get_episode(closest_episode_id)
            closest_actions = closest_episode["action"]
            closest_actions = closest_actions[closest_step_id:]
            closest_states = closest_episode["state"]
            closest_states = closest_states[closest_step_id:]
            # Crop or pad to horizon
            if len(closest_actions) < self.horizon:
                closest_actions = np.pad(closest_actions, ((0, self.horizon - len(closest_actions)), (0, 0)), mode="edge")
                closest_states = np.pad(closest_states, ((0, self.horizon - len(closest_states)), (0, 0)), mode="edge")
            elif len(closest_actions) > self.horizon:
                closest_actions = closest_actions[: self.horizon]
                closest_states = closest_states[: self.horizon]
            closest_actions = closest_actions[None, ...]
            closest_states = closest_states[None, ...]
            action_list.append(closest_actions)
        actions = np.concatenate(action_list, axis=0)
        states = np.concatenate(state_list, axis=0)
        return {"action": torch.from_numpy(actions).to(device=self.device, dtype=self.dtype), "state": torch.from_numpy(states).to(device=self.device, dtype=self.dtype)}
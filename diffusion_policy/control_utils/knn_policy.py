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

    def __init__(self, zarr_path, horizon=1, pad_before=0, pad_after=0, kernel=None, knn=4, keys=["img", "state", "action", "control", "demo_type"]):
        print("Start loading dataset to memory...")
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=keys)
        print("Dataset loaded.")
        self.sampler = SequenceSampler(replay_buffer=self.replay_buffer, sequence_length=horizon, pad_before=pad_before, pad_after=pad_after)
        self.states = np.copy(self.replay_buffer["state"])  # local copy
        # Emphasize the last dim of states
        self.states_weights = np.ones(self.states.shape[-1])
        # self.states[..., -1] = self.states[..., -1] / np.pi * 200
        # Apply weights to the last channel
        self.states = self.states * self.states_weights
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
        #
        self.nearest_threshold = 0.1

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
        knn = kwargs.get("knn", self.knn)
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
            # Find the k  closest state in the current episode
            closest_state_idx = self.get_closet_state(state, knn, allow_same_episode=kwargs.get("allow_same_episode", False))
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

    def get_closet_state(self, state, knn, allow_same_episode=False):
        """Get the closest state in the replay buffer."""
        state = np.copy(state)
        # state[-1] = state[-1] / np.pi * 200  # emphasize the last dim
        state = state * self.states_weights
        distances = np.linalg.norm(self.states - state, axis=1)
        if allow_same_episode:
            closest_state_idx = np.argsort(distances)[:knn]
        else:
            closest_state_idx = []
            for i in range(knn):
                closest_state_idx.append(np.argmin(distances))
                distances[closest_state_idx[-1]] = np.inf
                # mask the same episode
                closest_episode_id = self.episode_ids[closest_state_idx[-1]].item()
                distances[self.episode_ids[:, 0] == closest_episode_id] = np.inf
            closest_state_idx = np.array(closest_state_idx)
        # [DEBUG] check state difference
        # print("--------------------")
        # print("closest_episode_id:", self.episode_ids[closest_state_idx])
        # print("state difference:", np.linalg.norm(self.states[closest_state_idx] - state, axis=1))
        # print("state:", state)
        # print("closest_state:", self.states[closest_state_idx])
        return closest_state_idx


class KNNSAPolicy(KNNPolicy, BaseSAPolicy):
    def __init__(self, zarr_path, horizon=1, pad_before=0, pad_after=0, kernel=None, knn=4, keys=["img", "state", "action", "control", "demo_type"]):
        super().__init__(zarr_path, horizon, pad_before, pad_after, kernel, knn, keys)

    def predict_state_action(self, obs_dict: Dict[str, torch.Tensor], knn, **kwargs) -> Dict[str, torch.Tensor]:
        # Read state
        if "state" not in obs_dict:
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
            states = []
            for info in infos:
                pos_agent = info["pos_agent"]
                block_pose = info["block_pose"]
                if pos_agent.ndim == 2 and block_pose.ndim == 2:
                    pos_agent = pos_agent[-1]
                    block_pose = block_pose[-1]
                state = np.concatenate([pos_agent, block_pose])
                states.append(state)
            states = np.stack(states)
        else:
            states = obs_dict["state"]
            if isinstance(states, torch.Tensor):
                states = states.cpu().numpy()

        # Compute the k nearest neighbors
        closest_state_idx_list = []
        for state in states:
            # Find the k  closest state in the current episode
            closest_state_idx = self.get_closet_state(state, knn, allow_same_episode=kwargs.get("allow_same_episode", False))
            # Randomly choose one of the knn closest states
            chosen_num = 1 if states.shape[0] != 1 else knn
            assert chosen_num <= closest_state_idx.shape[0], "Not enough samples in the replay buffer."
            closest_state_idx = np.random.choice(closest_state_idx, chosen_num, replace=False)
            if states.shape[0] == 1:
                closest_state_idx_list = closest_state_idx
            else:
                closest_state_idx_list.append(closest_state_idx)

        pred_action_list = []
        pred_state_list = []
        for closest_state_idx in closest_state_idx_list:
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
            pred_action_list.append(closest_actions)
            pred_state_list.append(closest_states)
        pred_actions = np.concatenate(pred_action_list, axis=0)
        pred_states = np.concatenate(pred_state_list, axis=0)
        # Truncate the action
        pred_actions = pred_actions[:, : self.n_act_steps, :]
        pred_next_states = pred_states[:, 1 : self.n_act_steps + 1, :]  # Shift one step as we want to predict the next state
        return {"action": torch.from_numpy(pred_actions).to(device=self.device, dtype=self.dtype), "state": torch.from_numpy(pred_next_states).to(device=self.device, dtype=self.dtype)}

from typing import Dict
import torch
import numpy as np
import copy
import os
import hashlib
from tqdm.auto import tqdm
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.control_utils.trajectory_stitching import stitch_trajectory


class PushTControlDataset(BaseImageDataset):
    def __init__(self, zarr_path, horizon=1, pad_before=0, pad_after=0, seed=42, val_ratio=0.0, max_train_episodes=None, enable_stitching=False):
        """New parameters:
        enable_stitching: bool, whether to enable trajectory stitching. Default is False.
        """
        super().__init__()
        print("Start loading dataset to memory...")
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=["img", "state", "action", "control", "demo_type"])
        print("Dataset loaded.")
        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)
        self.sampler = SequenceSampler(replay_buffer=self.replay_buffer, sequence_length=horizon, pad_before=pad_before, pad_after=pad_after, episode_mask=train_mask)
        # param
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.zarr_path = zarr_path
        self.enable_stitching = enable_stitching
        if enable_stitching:
            self._compute_stitching_mapping(seed=seed, use_cache=True)

    def _compute_stitching_mapping(self, seed, use_cache=False):
        stiching_hash = hashlib.md5(f"{self.zarr_path}_{seed}_{self.replay_buffer.n_steps}_{len(self.sampler)}".encode()).hexdigest()
        cache_path = self.zarr_path.replace(".zarr", f"_stitching_mapping_{stiching_hash}.npz")
        if use_cache and os.path.exists(cache_path):
            data = np.load(cache_path)
            self.stitch_mapping = data["stitch_mapping"]
            return
        print("Computing stitching mapping...")
        # Build array pair: (state, episode_id, step_id)
        states = self.replay_buffer["state"]
        # Emphasize the last dim of states
        states[..., -1] = states[..., -1] / np.pi * 200
        episode_lengths = self.replay_buffer.episode_lengths
        episode_ids = [[i] * l for i, l in enumerate(episode_lengths)]
        episode_ids = np.stack(sum(episode_ids, [])).reshape(-1, 1)
        step_ids = [list(range(l)) for l in episode_lengths]
        step_ids = np.stack(sum(step_ids, [])).reshape(-1, 1)

        self.stitch_mapping = -np.ones([len(self.sampler), 2], dtype=np.int32)
        violating_mask = (self.replay_buffer["demo_type"] == 2).squeeze()
        for idx in tqdm(range(len(self.sampler))):
            sample = self.sampler.sample_sequence(idx)
            demo_type = sample["demo_type"].max() if "demo_type" in sample else 0
            if demo_type == 2:
                # Skip the demo_type 2; violating demonstration
                continue
            buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.sampler.indices[idx]
            sample_episode_id = episode_ids[buffer_start_idx].item()
            [closest_episode_id, closest_step_id, stitched_state, stitched_action] = stitch_trajectory(
                self.replay_buffer, sample_episode_id, sample, states, episode_ids, step_ids, mask=violating_mask, enable_render=False
            )
            self.stitch_mapping[idx, 0] = closest_episode_id
            self.stitch_mapping[idx, 1] = closest_step_id
        print("Stitching mapping computed.")
        if use_cache:
            np.savez(cache_path, stitch_mapping=self.stitch_mapping)
            print(f"Saved stitching mapping to {cache_path}")

    def _stitich_action(self, sample_idx):
        closest_episode_id, closest_step_id = self.stitch_mapping[sample_idx]
        closest_episode = self.replay_buffer.get_episode(closest_episode_id)
        closest_actions = closest_episode["action"]
        stitched_action = closest_actions[closest_step_id:]
        if len(stitched_action) < self.horizon:
            stitched_action = np.pad(stitched_action, ((0, self.horizon - len(stitched_action)), (0, 0)), mode="edge")
        elif len(stitched_action) > self.horizon:
            stitched_action = stitched_action[: self.horizon]
        return stitched_action.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        demo_type = sample["demo_type"].max()
        # Add control to the data
        control = sample["control"]
        control = np.moveaxis(control, -1, 1) / 255
        data["obs"]["control"] = control
        # Add stitching to the data with 50% probability
        if self.enable_stitching and np.random.rand() < 0.5 and demo_type != 2:
            stitched_action = self._stitich_action(idx)
            # [Debugging] Visualize the stitching
            # self._visualize_sample(data, stitched_action)
            data["action"] = stitched_action
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(replay_buffer=self.replay_buffer, sequence_length=self.horizon, pad_before=self.pad_before, pad_after=self.pad_after, episode_mask=~self.train_mask)
        val_set.train_mask = ~self.train_mask
        val_set.enable_stitching = False
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {"action": self.replay_buffer["action"], "agent_pos": self.replay_buffer["state"][..., :2]}
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer["image"] = get_image_range_normalizer()
        normalizer["control"] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample["state"][:, :2].astype(np.float32)  # (agent_posx2, block_posex3)
        image = np.moveaxis(sample["img"], -1, 1) / 255

        data = {
            "obs": {
                "image": image,  # T, 3, 96, 96
                "agent_pos": agent_pos,  # T, 2
            },
            "action": sample["action"].astype(np.float32),  # T, 2
            "demo_type": sample["demo_type"].astype(np.int32),  # T, 1
        }
        return data

    def _visualize_sample(self, data, stitched_action=None):
        image = data["obs"]["image"][0]
        # Resize image
        image = np.moveaxis(image, 0, -1)
        image = (image * 255).astype(np.uint8)
        image = cv2.resize(image, (512, 512))
        # Draw agent position
        agent_pos = data["obs"]["agent_pos"]
        cv2.circle(image, tuple(agent_pos[0].astype(int)), 5, (0, 255, 0), -1)
        # Draw actions
        actions = data["action"]
        for i in range(actions.shape[0]):
            color = (255, 0, 0)
            cv2.circle(image, tuple(actions[i].astype(int)), 2, color, -1)
        # Draw stitched actions
        if stitched_action is not None:
            for i in range(stitched_action.shape[0]):
                cv2.circle(image, tuple(stitched_action[i].astype(int)), 2, (0, 0, 255), 1)
        cv2.imshow("image", image)
        cv2.waitKey(0)


if __name__ == "__main__":
    import os
    import cv2

    pad_control = True
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    zarr_path = os.path.join(root_path, "data/kowndi_pusht_demo_v2_repulse.zarr")
    dataset = PushTControlDataset(zarr_path=zarr_path, horizon=16, pad_before=1, pad_after=7, seed=42, val_ratio=0.1, max_train_episodes=None, enable_stitching=False)
    for i in range(len(dataset)):
        print(f"Processing {i}/{len(dataset)}")
        control = dataset[i]["obs"]["control"].cpu().numpy()[0].transpose(1, 2, 0)
        demo_type = dataset[i]["demo_type"].cpu().numpy()[0]
        print(f"demo_type: {demo_type}")
        if np.max(control) > 0:
            control_image = (control * 255).astype(np.uint8)
            # cv2.imshow("control", control_image)
            # cv2.waitKey(0)
            # image = (dataset[i]["obs"]["image"].cpu().numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)
            # cv2.imshow("image", image)
            # cv2.waitKey(0)

from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset


class PushTControlDataset(PushTImageDataset):
    def __init__(self, zarr_path, horizon=1, pad_before=0, pad_after=0, seed=42, val_ratio=0.0, max_train_episodes=None, pad_control=False):
        """New parameters:
        pad_control: bool, whether to pad control data on uncontrolled episodes. Default is False.
        """
        super().__init__(zarr_path=zarr_path, horizon=horizon, pad_before=pad_before, pad_after=pad_after, seed=seed, val_ratio=val_ratio, max_train_episodes=max_train_episodes)
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=["img", "state", "action", "control", "demo_type"])
        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)
        self.sampler = SequenceSampler(replay_buffer=self.replay_buffer, sequence_length=horizon, pad_before=pad_before, pad_after=pad_after, episode_mask=train_mask)
        self._classify_by_demo_type()
        self.pad_control = pad_control

    def _classify_by_demo_type(self):
        """Classify the dataset by demo_type."""
        self.demo_type_indices = {}
        for i in range(len(self.sampler)):
            sample = self.sampler.sample_sequence(i)
            demo_type = sample["demo_type"].max()
            if demo_type not in self.demo_type_indices:
                self.demo_type_indices[demo_type] = []
            self.demo_type_indices[demo_type].append(i)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        # Add demo_type
        demo_type = sample["demo_type"]
        data["demo_type"] = demo_type
        # Add control to the data
        if demo_type.max() == 0 and self.pad_control:
            # Randomly sample a controlled episode
            random_idx = np.random.choice(self.demo_type_indices[1])
            random_sample = self.sampler.sample_sequence(random_idx)
            control = random_sample["control"]
        else:
            control = sample["control"]
        control = np.moveaxis(control, -1, 1) / 255
        data["obs"]["control"] = control

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

    def get_normalizer(self, mode="limits", **kwargs):
        normalizer = super().get_normalizer(mode, **kwargs)
        normalizer["control"] = get_image_range_normalizer()
        return normalizer


if __name__ == "__main__":
    import os
    import cv2

    pad_control = True
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    zarr_path = os.path.join(root_path, "data/kowndi_pusht_demo_v1_repulse.zarr")
    dataset = PushTControlDataset(zarr_path=zarr_path, horizon=1, pad_before=0, pad_after=0, seed=42, val_ratio=0.0, max_train_episodes=None, pad_control=pad_control)
    for i in range(len(dataset)):
        print(f"Processing {i}/{len(dataset)}")
        control = dataset[i]["obs"]["control"].cpu().numpy()[0].transpose(1, 2, 0)
        demo_type = dataset[i]["demo_type"].cpu().numpy()[0]
        print(f"demo_type: {demo_type}")
        if np.max(control) > 0:
            control_image = (control * 255).astype(np.uint8)
            cv2.imshow("control", control_image)
            cv2.waitKey(0)
            image = (dataset[i]["obs"]["image"].cpu().numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)
            cv2.imshow("image", image)
            cv2.waitKey(0)

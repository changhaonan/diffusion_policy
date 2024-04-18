from typing import Dict
import torch
import numpy as np
import copy
import os
from tqdm.auto import tqdm
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset


class PushTControlDataset(BaseImageDataset):
    def __init__(self, zarr_path, horizon=1, pad_before=0, pad_after=0, seed=42, val_ratio=0.0, max_train_episodes=None, pad_control=False):
        """New parameters:
        pad_control: bool, whether to pad control data on uncontrolled episodes. Default is False.
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
        self._classify_by_demo_type(use_cache=True)
        self.pad_control = pad_control

    def _classify_by_demo_type(self, use_cache=False):
        """Classify the dataset by demo_type."""
        cache_path = self.zarr_path.replace(".zarr", "_demo_type_indices.npz")
        if use_cache and os.path.exists(cache_path):
            data = np.load(cache_path)
            self.demo_type_indices = {k: v for k, v in data.items()}
            return
        else:
            self.demo_type_indices = {}
            print("Generating demo_type indices...")
            for i in tqdm(range(len(self.sampler))):
                sample = self.sampler.sample_sequence(i)
                demo_type = f"{sample['demo_type'].max()}"
                if demo_type not in self.demo_type_indices:
                    self.demo_type_indices[demo_type] = []
                self.demo_type_indices[demo_type].append(i)
            if use_cache:
                np.savez(cache_path, **self.demo_type_indices)
                print(f"Saved demo_type indices to {cache_path}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        demo_type = sample["demo_type"]
        # Add control to the data
        if demo_type.max() == 0 and self.pad_control:
            # Randomly sample a controlled episode
            random_idx = np.random.choice(self.demo_type_indices["1"])
            random_sample = self.sampler.sample_sequence(random_idx)
            control = random_sample["control"]
        else:
            control = sample["control"]
        control = np.moveaxis(control, -1, 1) / 255
        data["obs"]["control"] = control

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(replay_buffer=self.replay_buffer, sequence_length=self.horizon, pad_before=self.pad_before, pad_after=self.pad_after, episode_mask=~self.train_mask)
        val_set.train_mask = ~self.train_mask
        val_set._classify_by_demo_type()
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


if __name__ == "__main__":
    import os
    import cv2

    pad_control = True
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    zarr_path = os.path.join(root_path, "data/kowndi_pusht_demo_v2_repulse.zarr")
    dataset = PushTControlDataset(zarr_path=zarr_path, horizon=1, pad_before=0, pad_after=0, seed=42, val_ratio=0.0, max_train_episodes=None, pad_control=pad_control)
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

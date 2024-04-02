import numpy as np
import zarr
import torch
from diffusion_policy.demo_utils.misc_utils import get_data_stats, normalize_data, create_sample_indices, sample_sequence


# dataset
class DPDataset(torch.utils.data.Dataset):
    """Dataset for vanilla diffusion policy;
    General diffusion policy data contains:
    - image: (N, 3, 96, 96)
    - agent_pos: (N, 2)
    - action: (N, 2)
    """

    def __init__(self, dataset_path: str, pred_horizon: int, obs_horizon: int, action_horizon: int):

        # read from zarr dataset
        dataset_root = zarr.open(dataset_path, "r")

        # float32, [0, 1], (N, 96, 96, 3)
        train_image_data = dataset_root["data"]["img"][:]
        train_image_data = np.moveaxis(train_image_data, -1, 1)
        # (N, 3, 96, 96)

        # (N, D)
        train_data = {
            # first two dims of state vector are agent (i.e. gripper) locations
            "agent_pos": dataset_root["data"]["state"][:, :2],
            "action": dataset_root["data"]["action"][:],
        }
        episode_ends = dataset_root["meta"]["episode_ends"][:]

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(episode_ends=episode_ends, sequence_length=pred_horizon, pad_before=obs_horizon - 1, pad_after=action_horizon - 1)

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        # images are already normalized
        normalized_train_data["image"] = train_image_data

        self.indices = indices
        self.episode_ends = episode_ends
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )

        # discard unused observations
        nsample["image"] = nsample["image"][: self.obs_horizon, :]
        nsample["agent_pos"] = nsample["agent_pos"][: self.obs_horizon, :]
        return nsample


class CDPDataset(DPDataset):
    """Dataset for control diifusion policy;"""

    def __init__(self, dataset_path: str, pred_horizon: int, obs_horizon: int, action_horizon: int, annotator: any):
        super().__init__(dataset_path, pred_horizon, obs_horizon, action_horizon)
        self.annotator = annotator

    def __getitem__(self, idx):
        nsample = super().__getitem__(idx)
        # annotate
        anno, anno_image = self.annotator.annotate(nsample, stats=self.stats)
        nsample["anno_image"] = anno_image.astype(np.float32)[None, ...]
        nsample["anno"] = anno.astype(np.float32)
        return nsample


if __name__ == "__main__":
    # @markdown ### **Dataset Demo**
    import os
    import gdown

    # download demonstration data from Google Drive
    dataset_path = "pusht_cchi_v7_replay.zarr.zip"
    if not os.path.isfile(dataset_path):
        id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
        gdown.download(id=id, output=dataset_path, quiet=False)

    # parameters
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    # |o|o|                             observations: 2
    # | |a|a|a|a|a|a|a|a|               actions executed: 8
    # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

    # create dataset from file
    dataset = DPDataset(dataset_path=dataset_path, pred_horizon=pred_horizon, obs_horizon=obs_horizon, action_horizon=action_horizon)
    # save training data statistics (min, max) for each dim
    stats = dataset.stats

    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=4,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True,
    )

    # visualize data in batch
    batch = next(iter(dataloader))
    print("batch['image'].shape:", batch["image"].shape)
    print("batch['agent_pos'].shape:", batch["agent_pos"].shape)
    print("batch['action'].shape", batch["action"].shape)

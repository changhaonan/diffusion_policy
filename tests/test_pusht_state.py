"""Test the things related to the pusht_state."""

from diffusion_policy.dataset.pusht_state_dataset import PushTStateDataset


if __name__ == "__main__":
    # 1. Test dataset
    zarr_path = "/home/harvey/Project/diffusion_policy/data/kowndi_pusht_demo_v2_repulse.zarr"
    dataset = PushTStateDataset(
        zarr_path=zarr_path,
        horizon=16,
        pad_before=1,
        pad_after=7,
        state_key="state",
        action_key="action",
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
    )

    for i in range(10):
        sample = dataset[i]
        for k, v in sample.items():
            print(k, v.shape)

"""Check the dataset"""

import os
import numpy as np
import zarr
import cv2
from tqdm.auto import tqdm


def read_from_path(zarr_path, mode="r"):
    root = zarr.open(os.path.expanduser(zarr_path), mode)
    for key, value in root.items():
        print(f"{key}: {value}")
    return root


def visualize_dataset(root, wait_time=18):
    data = root["data"]
    meta = root["meta"]
    episode_ends = np.array(meta["episode_ends"])
    for i in range(0, len(episode_ends), 1):
        if i == 0:
            start = 0
        else:
            start = episode_ends[i - 1]
        end = episode_ends[i]
        print(f"Episode {i}: {start} -> {end}")
        image = data["img"][start:end]
        control = data["control"][start:end]
        for j in range(len(image)):
            vis_image = cv2.addWeighted(image[j], 0.5, control[j], 0.5, 0)
            cv2.imshow(f"episode-{i}", vis_image)
            cv2.waitKey(wait_time)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    root_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    # src_data = "kowndi_pusht_demo_repulse_no_control.zarr"
    src_data = "kowndi_pusht_demo_v0_region_aug0.zarr"
    # src_data = "kowndi_pusht_demo_v0_region.zarr"
    root = read_from_path(os.path.join(root_dir, src_data))
    visualize_dataset(root)

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


def visualize_dataset(root, wait_time=10):
    data = root["data"]
    meta = root["meta"]
    episode_ends = np.array(meta["episode_ends"])
    for i in range(len(episode_ends) - 1):
        start, end = episode_ends[i], episode_ends[i + 1]
        print(f"Episode {i}: {start} -> {end}")
        image = data["img"][start:end]
        control = data["control"][start:end]
        for j in range(len(image)):
            vis_image = np.concatenate([image[j], control[j]], axis=1)
            cv2.imshow(f"episode-{i}", vis_image)
            cv2.waitKey(wait_time)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    root_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    src_data = "kowndi_pusht_demo_repulse_no_control.zarr"
    root = read_from_path(os.path.join(root_dir, src_data))
    visualize_dataset(root)

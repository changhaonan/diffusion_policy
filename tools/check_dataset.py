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


def visualize_dataset(root, wait_time=18, shuffle=False):
    data = root["data"]
    meta = root["meta"]
    episode_ends = np.array(meta["episode_ends"])
    print(f"Number of episodes: {len(episode_ends)}")
    for i in range(0, len(episode_ends), 1):
        if shuffle:
            i = np.random.randint(0, len(episode_ends))
        if i == 0:
            continue
        else:
            start = episode_ends[i - 1]
        end = episode_ends[i]
        print(f"Episode {i}: {start} -> {end}")
        image = data["img"][start:end]
        control = data["control"][start:end]
        demo_type = data["demo_type"][start:end].max()
        for j in range(len(image)):
            vis_image = cv2.addWeighted(image[j], 0.5, control[j], 0.5, 0)
            # Add text on demo_type
            cv2.putText(vis_image, f"D: {demo_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow(f"episode-{i}", vis_image)
            cv2.waitKey(wait_time)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    wait_time = 100
    server_type = "ilab" if os.path.exists("/common/users") else "local"
    netid = "hc856"
    data_version = 1
    control_type = "repulse"
    if server_type == "local":
        data_src = "./data"
    elif server_type == "ilab":
        data_src = f"/common/users/{netid}/Project/diffusion_policy/data"

    src_data = f"{data_src}/kowndi_pusht_demo_v{data_version}_{control_type}.zarr"
    root = read_from_path(src_data)
    visualize_dataset(root, wait_time=wait_time, shuffle=True)

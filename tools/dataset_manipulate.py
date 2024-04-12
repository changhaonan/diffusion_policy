"""Manipulate the dataset; Merge, split, and shuffle the dataset."""

import os
import numpy as np
import zarr
import random
from tqdm.auto import tqdm


def read_from_path(zarr_path, mode="r"):
    root = zarr.open(os.path.expanduser(zarr_path), mode)
    for key, value in root.items():
        print(f"{key}: {value}")
    return root


def convert(root, export_path, convertion_str, **kwargs):
    if convertion_str == "control2no_control":
        return control2no_control(root, export_path, **kwargs)
    elif convertion_str == "control2ratio_control":
        return control2ratio_control(root, export_path, **kwargs)
    else:
        raise ValueError(f"Unknown convertion_str: {convertion_str}")


def control2no_control(root, export_path):
    """Each no control data is followed with control_repeat control data."""
    data = root["data"]
    meta = root["meta"]
    episode_ends = np.array(meta["episode_ends"])
    data_nc = {}
    episode_ends_nc = [0]
    for key, value in data.items():
        data_nc[key] = []

    # Create export zarr file
    export_root = zarr.open(export_path, "w")
    export_data = export_root.create_group("data")
    export_meta = export_root.create_group("meta")

    # Traverse the data
    for i in tqdm(range(len(episode_ends) - 1)):
        start, end = episode_ends[i], episode_ends[i + 1]
        control = data["control"][start:end]
        if np.sum(control) == 0:
            # No control data
            for key, value in data.items():
                data_nc[key].append(value[start:end])
            episode_ends_nc.append(episode_ends_nc[-1] + end - start)
    # Save the data
    for key, value in data_nc.items():
        export_data.create_dataset(key, data=np.concatenate(value, axis=0))
    export_meta.create_dataset("episode_ends", data=episode_ends_nc)
    return export_root


def control2ratio_control(root, export_path, ratio=0.5):
    """Convert control data to ratio% control data + no control data."""
    data = root["data"]
    meta = root["meta"]
    episode_ends = np.array(meta["episode_ends"])
    data_rc = {}
    episode_ends_rc = [0]
    for key, value in data.items():
        data_rc[key] = []

    # Create export zarr file
    export_root = zarr.open(export_path, "w")
    export_data = export_root.create_group("data")
    export_meta = export_root.create_group("meta")

    # Traverse the data
    count_nc = 0
    count_c = 0
    for i in tqdm(range(len(episode_ends) - 1)):
        start, end = episode_ends[i], episode_ends[i + 1]
        control = data["control"][start:end]
        if np.sum(control) == 0:
            # No control data
            for key, value in data.items():
                data_rc[key].append(value[start:end])
            episode_ends_rc.append(episode_ends_rc[-1] + end - start)
            count_nc += 1
        else:
            if random.random() < ratio:
                # Control data
                for key, value in data.items():
                    data_rc[key].append(value[start:end])
                episode_ends_rc.append(episode_ends_rc[-1] + end - start)
                count_c += 1

    print(f"count_nc: {count_nc}, count_c: {count_c}")
    # Save the data
    for key, value in data_rc.items():
        export_data.create_dataset(key, data=np.concatenate(value, axis=0))
    export_meta.create_dataset("episode_ends", data=episode_ends_rc)
    return export_root


if __name__ == "__main__":
    server_type = "local" if not os.path.exists("/common/users") else "ilab"
    netid = "hc856"
    control_type = "repulse"
    if server_type == "local":
        data_src = "./data"
    elif server_type == "ilab":
        data_src = f"/common/users/{netid}/Project/diffusion_policy/data"

    ratio = 0.3
    src_data = f"kowndi_pusht_demo_v0_{control_type}.zarr"
    tar_data = f"kowndi_pusht_demo_v0_{control_type}_rc_{ratio}.zarr"
    tar_data = os.path.join(data_src, tar_data)
    # convertion_str = "control2no_control"
    convertion_str = "control2ratio_control"

    root = read_from_path(os.path.join(data_src, src_data))
    convert_root = convert(root, tar_data, convertion_str, ratio=ratio)

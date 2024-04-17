"""Manipulate the dataset; Merge, split, and shuffle the dataset.
Notice: Compressor matters!!!
"""

import os
import numcodecs
import numpy as np
import zarr
import random
from tqdm.auto import tqdm
from diffusion_policy.common.replay_buffer import ReplayBuffer


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
    elif convertion_str == "convert_v0_to_v1":
        return convert_v0_to_v1(root, export_path, **kwargs)
    elif convertion_str == "convert_v1_to_v2":
        return convert_v1_to_v2(root, export_path, **kwargs)
    else:
        raise ValueError(f"Unknown convertion_str: {convertion_str}")


def control2no_control(root, export_path, cpr=None):
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
        export_data.create_dataset(key, data=np.concatenate(value, axis=0), compressor=cpr)
    export_meta.create_dataset("episode_ends", data=episode_ends_nc, compressor=cpr)
    return export_root


def control2ratio_control(root, export_path, ratio=0.5, cpr=None):
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
        export_data.create_dataset(key, data=np.concatenate(value, axis=0), compressor=cpr)
    export_meta.create_dataset("episode_ends", data=episode_ends_rc, compressor=cpr)
    return export_root


def convert_v0_to_v1(root, export_path, **kwargs):
    """V0 data doesn't have demo_type."""
    data = root["data"]
    meta = root["meta"]
    episode_ends = np.array(meta["episode_ends"])

    replay_buffer = ReplayBuffer.create_from_path(export_path, mode="a")
    # Traverse the data
    for i in tqdm(range(len(episode_ends) - 1)):
        start, end = episode_ends[i], episode_ends[i + 1]
        control = data["control"][start:end]
        if np.sum(control) == 0:
            # No control data
            demo_type = np.zeros([end - start, 1], dtype=np.int32)
        else:
            # Control data
            demo_type = np.ones([end - start, 1], dtype=np.int32)
        # No control data
        data_new = {}
        for key, value in data.items():
            data_new[key] = value[start:end]
        data_new["demo_type"] = demo_type

        # Save the data
        replay_buffer.add_episode(data_new, compressors="disk")


def convert_v1_to_v2(root, export_path, **kwargs):
    """V2 has larger chunk size; faster to read."""
    data = root["data"]
    meta = root["meta"]
    episode_ends = np.array(meta["episode_ends"])

    replay_buffer = ReplayBuffer.create_from_path(export_path, mode="a")
    # Traverse the data
    for i in tqdm(range(len(episode_ends) - 1)):
        start, end = episode_ends[i], episode_ends[i + 1]
        data_new = {}
        for key, value in data.items():
            data_new[key] = value[start:end]
        # Save the data
        replay_buffer.add_episode(data_new, compressors="disk")


if __name__ == "__main__":
    server_type = "local" if not os.path.exists("/common/users") else "ilab"
    netid = "hc856"
    cpr = None
    control_type = "repulse"
    if server_type == "local":
        data_src = "./data"
    elif server_type == "ilab":
        data_src = f"/common/users/{netid}/Project/diffusion_policy/data"

    ratio = 0.3
    src_data = f"kowndi_pusht_demo_v1_{control_type}.zarr"
    tar_data = f"kowndi_pusht_demo_v2_{control_type}.zarr"
    tar_data = os.path.join(data_src, tar_data)
    # convertion_str = "control2no_control"
    # convertion_str = "control2ratio_control"
    # convertion_str = "convert_v0_to_v1"
    convertion_str = "convert_v1_to_v2"

    root = read_from_path(os.path.join(data_src, src_data))
    convert(root, tar_data, convertion_str, cpr=cpr)

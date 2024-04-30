"""Testing dinosaur dataset."""

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from torch.utils.data import TensorDataset
from diffusion_kernel_regression import DiffusionKernelRegression


def dino_dataset(data_path, n=8000):
    df = pd.read_csv(data_path, sep="\t")
    df = df[df["dataset"] == "dino"]

    rng = np.random.default_rng(42)
    ix = rng.integers(0, len(df), n)
    x = df["x"].iloc[ix].tolist()
    x = np.array(x) + rng.normal(size=len(x)) * 0.15
    y = df["y"].iloc[ix].tolist()
    y = np.array(y) + rng.normal(size=len(x)) * 0.15
    x = (x / 54 - 1) * 4
    y = (y / 48 - 1) * 4
    X = np.stack((x, y), axis=1)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def draw_data_and_trajectory(data, trajectory):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(data[:, 0], data[:, 1], np.zeros_like(data[:, 0]), s=1, c="black")
    scale = np.max(np.abs(trajectory)) / trajectory.shape[1]
    for i in range(trajectory.shape[0]):
        color = plt.cm.jet(i / trajectory.shape[0])
        ax.plot(trajectory[i, :, 0], trajectory[i, :, 1], np.arange(trajectory.shape[1], 0, -1) * scale, c=color)
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    ax.axis("equal")
    plt.show()


if __name__ == "__main__":
    # Params
    data_path = "/home/harvey/Project/diffusion_policy/data/kernel_regression/DatasaurusDozen.tsv"
    dataset = dino_dataset(data_path)
    data = dataset.tensors[0].cpu().numpy()

    batch_size = 8
    diffusion_steps = 100
    scheduler_type = "squaredcos_cap_v2"
    # scheduler_type = "linear"

    dkr = DiffusionKernelRegression(datas=data, diffusion_steps=diffusion_steps, scheduler_type=scheduler_type)
    for i in range(10):
        samples, trajectory = dkr.conditional_sampling(batch_size=batch_size, return_trajectory=True)
        trajectory = trajectory.squeeze()
        if trajectory.ndim == 2:
            trajectory = trajectory[None, :]
        draw_data_and_trajectory(data, trajectory)

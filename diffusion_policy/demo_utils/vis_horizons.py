import numpy as np
import zarr
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from diffusion_policy.demo_utils.dataset import DPDataset
import os
import gdown
from scipy.interpolate import CubicSpline
from scipy.special import comb
import pickle
import copy

def interpolation(method: str = "bezier", waypoints: np.array = None, num_points: int = 100):
    if method == "bezier":
        return bezier_curve(waypoints, num_points)
    elif method == "spline":
        return spline_interpolation(waypoints, num_points)
    elif method == "catmull_rom_spline":
        return catmull_rom_spline(waypoints, num_points)
    else:
        raise ValueError(f"Invalid interpolation method: {method}")
    
def spline_interpolation(waypoints, num_points=100):
    """
    Generate a smooth trajectory through waypoints using spline interpolation.
    
    Args:
    - waypoints (np.array): An array of waypoints of shape (N, 2).
    - num_points (int): Number of points to generate along the trajectory.
    
    Returns:
    - np.array: Generated points along the trajectory.
    """
    t = np.linspace(0, 1, len(waypoints))
    cs_x = CubicSpline(t, waypoints[:, 0])
    cs_y = CubicSpline(t, waypoints[:, 1])
    t_new = np.linspace(0, 1, num_points)
    interpolated_points = np.vstack([cs_x(t_new), cs_y(t_new)]).T
    return interpolated_points

def catmull_rom_spline(points, n=100):
    # Catmull-Rom to Bezier conversion matrix
    m = np.array([[0, 1, 0, 0], [-0.5, 0, 0.5, 0], [1, -2.5, 2, -0.5], [-0.5, 1.5, -1.5, 0.5]])
    result = []
    for i in range(len(points) - 3):
        p = np.array([points[i], points[i+1], points[i+2], points[i+3]])
        for t in np.linspace(0, 1, n):
            tvec = np.array([1, t, t**2, t**3])
            result.append(tvec @ m @ p)
    return np.array(result)

def bezier_curve(points, n=100):
    n_points = len(points)
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    t = np.linspace(0, 1, n)
    polynomial_array = np.array([comb(n_points - 1, i) * (t**(n_points - 1 - i)) * (1 - t)**i for i in range(n_points)])
    xvals = np.dot(x, polynomial_array)
    yvals = np.dot(y, polynomial_array)
    return np.vstack([xvals, yvals]).T

if __name__ == "__main__":
    # add argparse
    import argparse
    parser = argparse.ArgumentParser(description="Generate new prediction horizons")
    parser.add_argument("--dataset_path", type=str, default="data/kowndi_pusht_demo_repulse.zarr", help="Path to the dataset")
    parser.add_argument("--pred_horizon", type=int, default=16, help="Prediction horizon")
    parser.add_argument("--obs_horizon", type=int, default=2, help="Observation horizon")
    parser.add_argument("--action_horizon", type=int, default=8, help="Action horizon")
    parser.add_argument("--interpolation_method", type=str, default="bezier", choices=["bezier", "spline", "catmull_rom_spline"] ,help="Interpolation method to use")
    parser.add_argument("--num_sample_waypoints", type=int, default=3, help="Number of waypoints to sample from each prediction horizon of length 16 - 2")
    parser.add_argument("--vis_num_episodes", type=int, default=200, help="Number of episodes to plot")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--save_path", type=str, default="new_episodes.pkl", help="Path to save the new episodes")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--aug_mode", type=str, default="add2prev", choices=["store_new", "add2prev"], help="Augmentation mode")
    args = parser.parse_args()

    args.debug = True
    np.random.seed(args.seed) # For reproducibility
    # if not os.path.exists(args.dataset_path):
    #     id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
    #     gdown.download(id=id, output=args.dataset_path, quiet=False)

    # o > o observations: 2
    #     a > a > a > a > a > a > a > a actions: 8
    # p > p > p > p > p > p > p > p > p > p > p > p > p > p > p > p predictions: 16

    dataset_root = zarr.open(args.dataset_path, "r")
    agent_pos = dataset_root["data"]["state"][:, :2] 
    episode_ends = dataset_root["meta"]["episode_ends"][:]
    episodes = [agent_pos[:episode_ends[0]]]



    for i in range(0, len(episode_ends), 4):
        seed = i // 4
        plt.figure(figsize=(20, 5))
        plt.suptitle(f"Seed {seed}")
        for j in range(4):
            plt.subplot(1, 4, j + 1)
            subplot_title = f" {i + j} Original" if j == 0 else f"{i + j} Repulse {j-1}"
            if i + j < len(episode_ends):  # Check to avoid index out of range
                start_idx = 0 if i + j == 0 else episode_ends[i + j - 1]
                end_idx = episode_ends[i + j]
                # marker = "o" if j == 0 else "x"
                marker = "o"
                color = "r" if j == 0 else "b"
                for x, y in agent_pos[start_idx:end_idx]:
                    plt.plot(x, y, marker=marker, color=color)  # Added marker for visibility
                    plt.title(subplot_title)
            else:
                plt.title(f"No data for plot {i + j}")

            # if i + j == 0:
            #     subplot_title = f" 0 Original"
            #     for x, y in agent_pos[0:episode_ends[i + j]]:
            #         plt.plot(x, y)
            #     plt.title(subplot_title)
            # else:
            #     for x, y in agent_pos[episode_ends[i + j - 1]:episode_ends[i + j]]:
            #         plt.plot(x, y)
            #     plt.title(subplot_title)
        plt.show()
        # break

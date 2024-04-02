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

    dataset = DPDataset(dataset_path=args.dataset_path, pred_horizon=args.pred_horizon, obs_horizon=args.obs_horizon, action_horizon=args.action_horizon)
    
    episodes = []
    new_episodes = []
    for entry in tqdm(dataset, total=len(dataset), desc="Generating New Pred Horizons"):
        action = entry["action"]
        sample_indices1 = np.random.choice(range(1, args.pred_horizon//2), args.num_sample_waypoints//2, replace=False)
        sample_indices2 = np.random.choice(range(args.pred_horizon//2 + args.pred_horizon % 2 + 1, args.pred_horizon-1), args.num_sample_waypoints//2, replace=False)
        
        waypoints = action[sample_indices1]
        # Append the first and last action to the waypoints
        waypoints = np.vstack([action[0], action[sample_indices1], action[args.pred_horizon//2] , action[sample_indices2], action[-1]])
        interpolated_points = interpolation(args.interpolation_method, waypoints, args.pred_horizon)
        new_episodes.append(interpolated_points)
        episodes.append(action)

    dataset_new = copy.deepcopy(dataset)
    for idx, entry in enumerate(dataset_new):
        entry["action"] = new_episodes[idx]
    
    if args.aug_mode == "store_new":
        save_path = "new_episodes.pkl"
        with open(args.save_path, "wb") as f:
            pickle.dump(dataset_new, f)

    elif args.aug_mode == "add2prev":
        save_path = "add2prev_episodes.pkl"
        dataset = dataset + dataset_new      
        with open(save_path, "wb") as f:
            pickle.dump(dataset, f)
 
    if args.debug:
        # Plot the generated episodes
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sample_existing_episode_indices = np.random.choice(len(episodes), args.vis_num_episodes, replace=False)
        for idx in sample_existing_episode_indices:
            plt.plot(episodes[idx][:, 0], episodes[idx][:, 1], label=f"Index {idx}")
        plt.title("Existing Pred Horizon")
    
        plt.subplot(1, 2, 2)
        sampled_episode_indices = sample_existing_episode_indices 
        for idx in sampled_episode_indices:
            plt.plot(new_episodes[idx][:, 0], new_episodes[idx][:, 1], label=f"Index {idx}")
        plt.title(f"Generated Pred Horizon (wps: {args.num_sample_waypoints})")
        plt.show()

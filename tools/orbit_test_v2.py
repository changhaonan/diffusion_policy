"""Orbit testing fitting with KNN policy"""

import os
import zarr
import numpy as np
from matplotlib import pyplot as plt
from diffusion_policy.common.replay_buffer import ReplayBuffer

#######################  Generating data #######################


# Define the spiral function in polar coordinates (r = a + b*theta)
def spiral(theta, beta=0.0, gamma=1.0, a=0.0, b=0.1):
    r = a + b * theta
    return gamma * r * np.cos(theta + beta), r * np.sin(theta + beta)


# Derivative of the spiral function to compute gradient
def spiral_gradient(theta, a=0, b=0.1):
    # Gradient in polar coordinates
    dr_dtheta = b
    r = a + b * theta
    # Conversion to Cartesian coordinates
    dx_dtheta = dr_dtheta * np.cos(theta) - r * np.sin(theta)
    dy_dtheta = dr_dtheta * np.sin(theta) + r * np.cos(theta)
    return dx_dtheta, dy_dtheta


def generate_raw_data(num_sample, circle_round, reverse_B: bool = False, vis: bool = False):
    # Parameters
    a_A = 0
    b_A = 0.05
    gamma_A = 0.6

    a_B = 0.15
    b_B = 0.05
    gamma_B = 1.3

    # Generate theta values
    theta_values = np.linspace(2 * circle_round * np.pi, 0, num_sample)

    # Compute spiral coordinates & gradients
    x_values_A, y_values_A = spiral(theta=theta_values, gamma=gamma_A, a=a_A, b=b_A)
    gradients_A = np.zeros((theta_values.shape[0], 2))
    gradients_A[:-1, 0] = x_values_A[1:] - x_values_A[:-1]
    gradients_A[:-1, 1] = y_values_A[1:] - y_values_A[:-1]

    x_values_B, y_values_B = spiral(theta=theta_values, gamma=gamma_B, a=a_B, b=b_B)
    # Reverse the trajectory of B
    if reverse_B:
        x_values_B = x_values_B[::-1]
        y_values_B = y_values_B[::-1]
    gradients_B = np.zeros((theta_values.shape[0], 2))
    gradients_B[:-1, 0] = x_values_B[1:] - x_values_B[:-1]
    gradients_B[:-1, 1] = y_values_B[1:] - y_values_B[:-1]

    use_mask = False
    if use_mask:
        # Filter out when theta is around np.pi / 2
        mask = np.logical_and(theta_values > 8 * np.pi, theta_values < 8.3 * np.pi)
        masked_state = np.stack([x_values_A[mask], y_values_A[mask]], axis=1)
        x_values_A[mask] = np.ones_like(x_values_A[mask])
        y_values_A[mask] = np.ones_like(y_values_A[mask])
        gradients_A[mask] = np.zeros_like(gradients_A[mask])
        x_values_B[mask] = np.ones_like(x_values_B[mask])
        y_values_B[mask] = np.ones_like(y_values_B[mask])
        gradients_B[mask] = np.zeros_like(gradients_B[mask])
    else:
        masked_state = np.stack([x_values_A, y_values_A], axis=1)

    ax = None
    if vis:
        # Plot the spiral and its gradient field
        fig, ax = plt.subplots()
        ax.quiver(x_values_A, y_values_A, gradients_A[:, 0], gradients_A[:, 1], color="red", label="Gradient Field - A")
        ax.quiver(x_values_B, y_values_B, gradients_B[:, 0], gradients_B[:, 1], color="blue", label="Gradient Field - B")
        # plt.plot(x_values, y_values, label="Spiral Curve")
        ax.axis("equal")
        plt.title("Spiral Curve and Its Gradient Field")
        plt.legend()
        plt.show()
    # Format into data dict
    episodes = []
    episodes.append({"state": np.stack([x_values_A, y_values_A], axis=1), "action": gradients_A, "label": 0})
    episodes.append({"state": np.stack([x_values_B, y_values_B], axis=1), "action": gradients_B, "label": 1})
    return episodes, masked_state


def draw_state_action(states, actions, pred_states=None, epsiodes=None, save_path=None):
    fig, ax = plt.subplots()
    if epsiodes is not None:
        for _i, episode in enumerate(epsiodes):
            _state = episode["state"]
            _action = episode["action"]
            color = "red" if _i == 0 else "blue"
            ax.quiver(_state[:, 0], _state[:, 1], _action[:, 0], _action[:, 1], color=color)
    # ax.scatter(states[:, 0], states[:, 1], label="State", c="black", alpha=0.5)
    # if pred_states is not None:
    #     for _i in range(pred_states.shape[0]):
    #         color = plt.cm.jet(_i / pred_states.shape[0])
    #         ax.plot(pred_states[_i, :, 0], pred_states[_i, :, 1], c=color, marker="x", markersize=2)
    # if actions.ndim == 2:
    #     actions = actions[None, :]
    ax.axis("equal")
    plt.title("State and Action")
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def save_dataset_to_zarr(output: str, episodes):
    # create replay buffer in read-write mode
    replay_buffer = ReplayBuffer.create_from_path(output, mode="a")
    for episode in episodes:
        replay_buffer.add_episode({"state": episode["state"], "action": episode["action"]}, compressors="disk")


if __name__ == "__main__":
    # Parameters
    num_sample = 500
    circle_round = 8
    reverse_B = False
    n_obs_steps = 1
    n_act_steps = 4
    horizon = 8
    diffusion_steps = 1000
    knn_max = 10
    batch_size = 8
    # scheduler_type = "linear"
    scheduler_type = "squaredcos_cap_v2"

    # Generate orbit data
    assert n_obs_steps + n_act_steps <= horizon, "n_obs_steps + n_act_steps should be less than horizon"
    # Generate raw data
    epsiodes, masked_state = generate_raw_data(num_sample=num_sample, circle_round=circle_round, reverse_B=reverse_B, vis=False)

    # Check
    draw_state_action(states=np.array([]), actions=np.array([]), epsiodes=epsiodes)
    # Save the dataset to zarr
    zarr_path = "/Users/haonanchang/Projects/diffusion_policy/data/orbit.zarr"
    if not os.path.exists(zarr_path):
        save_dataset_to_zarr(zarr_path, epsiodes)

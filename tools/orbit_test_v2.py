"""Orbit testing fitting with KNN policy"""

import cv2
import os
from typing import List
import zarr
import math
import numpy as np
from matplotlib import pyplot as plt
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.control_utils.knn_policy import KNNSAPolicy

#######################  Generating data #######################


# Define the spiral function in polar coordinates (r = a + b*theta)
def spiral(theta, beta=0.0, gamma=1.0, a=0.0, b=0.1):
    r = a + b * theta
    gamma_a = math.sqrt(gamma / (1 + gamma))
    gamma_b = math.sqrt(1 / (1 + gamma))
    return gamma_a * r * np.cos(theta + beta), gamma_b * r * np.sin(theta + beta)


# # Derivative of the spiral function to compute gradient
# def spiral_gradient(theta, a=0, b=0.1):
#     # Gradient in polar coordinates
#     dr_dtheta = b
#     r = a + b * theta
#     # Conversion to Cartesian coordinates
#     dx_dtheta = dr_dtheta * np.cos(theta) - r * np.sin(theta)
#     dy_dtheta = dr_dtheta * np.sin(theta) + r * np.cos(theta)
#     return dx_dtheta, dy_dtheta


def generate_raw_data(num_sample, circle_round, reverse_B: bool = False, vis: bool = False):
    # Parameters
    a_A = 0
    b_A = 0.02
    gamma_A = 0.5

    a_B = 0.15
    b_B = 0.02
    gamma_B = 2.0

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


#######################  Wrap orbit into Env #######################
import gym
from gym import spaces


class OrbitEnv(gym.Env):

    def __init__(self, episodes) -> None:
        super().__init__()
        self.episodes = episodes
        self.agent_pos = np.array([0, 0])
        self.prev_agent_pos = np.array([0, 0])
        self.bg_image = self.draw_background_image()
        self.actions = None
        self._seed = 0

    def seed(self, seed=None):
        self._seed = seed

    def reset(self):
        # Randomly select initial
        rng = np.random.RandomState(self._seed)
        self.agent_pos = rng.uniform(-1, 1, size=(2,))
        self.prev_agent_pos = self.agent_pos
        return self.agent_pos

    def render(self):
        # Render env
        vis_image = self.bg_image.copy()
        # Draw agent
        vis_image = cv2.circle(vis_image, (int(self.agent_pos[0] * 128 + 128), int(self.agent_pos[1] * 128 + 128)), 3, [0, 255, 0], 1)
        # Draw action
        if self.actions is not None:
            acted_pos = np.copy(self.prev_agent_pos)
            for action in self.actions:
                # acted_pos += action
                vis_image = cv2.arrowedLine(
                    vis_image,
                    (int(acted_pos[0] * 128 + 128), int(acted_pos[1] * 128 + 128)),
                    (int((acted_pos + action)[0] * 128 + 128), int((acted_pos + action)[1] * 128 + 128)),
                    [0, 255, 0],
                    1,
                )
                acted_pos += action
        # Draw previous agent
        vis_image = cv2.circle(vis_image, (int(self.prev_agent_pos[0] * 128 + 128), int(self.prev_agent_pos[1] * 128 + 128)), 3, [0, 0, 255], 1)
        return vis_image

    def step(self, actions):
        self.prev_agent_pos = np.copy(self.agent_pos)
        self.actions = actions
        # Update agent position
        for action in actions:
            self.agent_pos += action
        return self.agent_pos

    def draw_background_image(self):
        background_image = np.zeros((256, 256, 3), dtype=np.uint8)
        # Draw all states trajectory
        for i, episode in enumerate(self.episodes):
            color = [255, 0, 0] if i == 0 else [0, 0, 255]
            states = episode["state"]
            scaled_states = states * 128 + 128  # Scale to 256x256
            for j in range(scaled_states.shape[0]):
                x, y = scaled_states[j]
                background_image = cv2.circle(background_image, (int(x), int(y)), 2, color, 1)
        # Draw the goal region: a cirlce of radius 0.1
        background_image = cv2.circle(background_image, (128, 128), 12, [0, 255, 0], 1)
        return background_image


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

    # Load KNN policy
    sa_policy = KNNSAPolicy(zarr_path=zarr_path, horizon=horizon, pad_before=n_obs_steps - 1, pad_after=n_act_steps - 1, knn=knn_max, keys=["state", "action"])

    # Test the policy
    env = OrbitEnv(episodes=epsiodes)
    env.seed(3)
    obs = env.reset()

    # Dynamic test
    sa_policy.reset()

    for i in range(1000):
        pred = sa_policy.predict_state_action({"state": np.stack([obs] * n_obs_steps, axis=0)}, knn=3, allow_same_episode=True)
        pred_states, pred_actions = pred["state"], pred["action"]
        pred_actions = pred_actions.cpu().numpy()
        # Randomly sample an action
        sample_idx = np.random.randint(0, pred_actions.shape[0])
        pred_actions = pred_actions[sample_idx]
        obs = env.step(pred_actions)
        img = env.render()

        print(f"Step {i}, obs: {obs}")
        if np.linalg.norm(obs) < 0.1:
            break
        cv2.imshow("Orbit", img)
        cv2.waitKey(1)

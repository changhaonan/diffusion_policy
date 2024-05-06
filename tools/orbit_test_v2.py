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
from diffusion_policy.control_utils.policy_sample_tree import PolicySampleTree


#######################  Generating data #######################
# Define the spiral function in polar coordinates (r = a + b*theta)
def spiral(theta, beta=0.0, gamma=1.0, a=0.0, b=0.1):
    r = a + b * theta
    gamma_a = math.sqrt(gamma / (1 + gamma))
    gamma_b = math.sqrt(1 / (1 + gamma))
    return gamma_a * r * np.cos(theta + beta), gamma_b * r * np.sin(theta + beta)


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

    def set_state(self, state):
        self.agent_pos = state

    def render(self):
        # Render env
        vis_image = self.bg_image.copy()
        # Draw agent
        vis_image = cv2.circle(vis_image, (int(self.agent_pos[0] * 256 + 256), int(self.agent_pos[1] * 256 + 256)), 3, [0, 255, 0], 1)
        # # Draw action
        # if self.actions is not None:
        #     acted_pos = np.copy(self.prev_agent_pos)
        #     for action in self.actions:
        #         # acted_pos += action
        #         vis_image = cv2.arrowedLine(
        #             vis_image,
        #             (int(acted_pos[0] * 256 + 256), int(acted_pos[1] * 256 + 256)),
        #             (int((acted_pos + action)[0] * 256 + 256), int((acted_pos + action)[1] * 256 + 256)),
        #             [0, 255, 0],q
        #             1,
        #         )
        #         acted_pos += action
        # Draw previous agent
        vis_image = cv2.circle(vis_image, (int(self.prev_agent_pos[0] * 256 + 256), int(self.prev_agent_pos[1] * 256 + 256)), 3, [0, 0, 255], 1)
        return vis_image

    def step(self, actions):
        self.prev_agent_pos = np.copy(self.agent_pos)
        self.actions = actions
        # Update agent position
        for action in actions:
            self.agent_pos += action
        return self.agent_pos

    def draw_background_image(self):
        background_image = np.zeros((512, 512, 3), dtype=np.uint8)
        # Draw all states trajectory
        for i, episode in enumerate(self.episodes):
            color = [255, 0, 0] if i == 0 else [0, 0, 255]
            states = episode["state"]
            scaled_states = states * 256 + 256  # Scale to 512x512
            for j in range(scaled_states.shape[0]):
                x, y = scaled_states[j]
                background_image = cv2.circle(background_image, (int(x), int(y)), 2, color, 1)
        # Draw the goal region: a cirlce of radius 0.1
        background_image = cv2.circle(background_image, (256, 256), 12, [0, 255, 0], 1)
        return background_image


def visualize_orbit_tree(states, actions, values, skeletons, env: OrbitEnv):
    root_state = states[0]
    env.reset()
    env.set_state(root_state[-1, :])
    img = env.render()
    # cv2.imshow("Orbit", img)
    # cv2.waitKey(0)
    # Visualize
    for idx, skeleton in enumerate(skeletons):
        vis_img = np.copy(img)
        # visq_img = img
        for node_idx in skeleton:
            color = plt.cm.jet(idx / len(skeletons))
            color = [int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)]
            state = states[node_idx]
            action = actions[node_idx]
            # Draw the state and action
            vis_img = cv2.circle(vis_img, (int(state[-1, 0] * 256 + 256), int(state[-1, 1] * 256 + 256)), 2, [255, 255, 255], 1)
            # Draw the action
            if action is not None:
                for _i in range(action.shape[0]):
                    acted_pos = np.copy(state[-1])
                    for _j in range(action.shape[1]):
                        act = action[_i, _j]
                        vis_img = cv2.arrowedLine(
                            vis_img, (int(acted_pos[0] * 256 + 256), int(acted_pos[1] * 256 + 256)), (int((acted_pos + act)[0] * 256 + 256), int((acted_pos + act)[1] * 256 + 256)), color, 1
                        )
                        vis_img = cv2.circle(vis_img, (int(acted_pos[0] * 256 + 256), int(acted_pos[1] * 256 + 256)), 3, color, 1)
                        acted_pos += act
        skeleton_values = [values[node_idx] for node_idx in skeleton]
        # print(f"Values: {skeleton_values}")
        skeleton_str = ", ".join([str(node_idx) for node_idx in skeleton])
        cv2.imshow(f"Orbit-{skeleton_str}", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def get_reward_func(control_type, **kwargs):
    # Return a value function based on the cotrol type:
    if control_type == "goal":
        def reward_func(states):
            if states.ndim == 2:
                states = states[-1]  # Only consider the last state
            goal = np.array([0.0, 0.0])
            return -np.linalg.norm(states - goal)
        return reward_func
    elif control_type == "avoid":
        def reward_func(states):
            if states.ndim == 1:
                states = states[None, :]
            obstacle = kwargs.get("obstacle", np.array([0.3, 0.3]))
            # Compute the nearest distance to the obstacle
            dist = np.linalg.norm(states - obstacle, axis=1)
            min_dist = np.min(dist)
            return min_dist
        return reward_func
    else:
        raise ValueError("Invalid control type")


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
    control_type = "goal"  # goal, avoid, attract...

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
    policy_tree = PolicySampleTree(policy=sa_policy, k_sample=2, max_depth=3)

    # Test the policy
    env = OrbitEnv(episodes=epsiodes)
    env.seed(3)
    test_env = OrbitEnv(episodes=epsiodes)
    obs = env.reset()

    # Dynamic test
    sa_policy.reset()

    reward_func = get_reward_func(control_type)

    for i in range(1000):
        policy_tree.reset()
        # Test tree
        policy_tree.expand_tree({"state": np.stack([obs] * n_obs_steps, axis=0)}, reward_func=reward_func)
        states, actions, values, skeletons, branch_idxs = policy_tree.export()
        visualize_orbit_tree(states, actions, values, skeletons, test_env)
        # Rank actions based on the value
        skeleton_values = [values[skeleton[1]] for skeleton in skeletons]
        best_skeleton_idx = np.argmax(skeleton_values)
        best_skeleton = skeletons[best_skeleton_idx]
        best_action = actions[best_skeleton[0]][branch_idxs[best_skeleton[1]]]
        obs = env.step(best_action)
        print(f"Step {i}, obs: {obs}")
        ## Testing different control effects; Selecting the best action

        # pred = sa_policy.predict_state_action({"state": np.stack([obs] * n_obs_steps, axis=0)}, knn=3, allow_same_episode=True)
        # pred_states, pred_actions = pred["state"], pred["action"]
        # pred_actions = pred_actions.cpu().numpy()
        # # Randomly sample an action
        # sample_idx = np.random.randint(0, pred_actions.shape[0])
        # pred_actions = pred_actions[sample_idx]
        # obs = env.step(pred_actions)
        # img = env.render()

        # # Expand the policy tree
        # print(f"Step {i}, obs: {obs}")
        if np.linalg.norm(obs) < 0.1:
            break
        # cv2.imshow("Orbit", img)
        # cv2.waitKey(1)

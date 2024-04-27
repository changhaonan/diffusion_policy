import numpy as np
import matplotlib.pyplot as plt
import numba
from copy import deepcopy

########################################## Common ##########################################


@numba.jit(nopython=True)
def create_indices(episode_ends: np.ndarray, sequence_length: int, episode_mask: np.ndarray, pad_before: int = 0, pad_after: int = 0, debug: bool = True) -> np.ndarray:
    episode_mask.shape == episode_ends.shape
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)

    indices = list()
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            # skip episode
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert start_offset >= 0
                assert end_offset >= 0
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


########################################## Dataset Generation ##########################################
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


def generate_raw_data(vis: bool = False):
    # Parameters
    a_A = 0
    b_A = 0.05
    gamma_A = 0.6
    
    a_B = 0.15
    b_B = 0.05
    gamma_B = 1.3
    reverse_B = False

    num_sample = 400
    circle_round = 8

    # Generate theta values
    theta_values = np.linspace(0, 2 * circle_round * np.pi, num_sample)

    # Compute spiral coordinates & gradients
    x_values_A, y_values_A = spiral(theta=theta_values, gamma=gamma_A, a=a_A, b=b_A)
    gradients_A = np.zeros((num_sample, 2))
    gradients_A[:-1, 0] = x_values_A[1:] - x_values_A[:-1]
    gradients_A[:-1, 1] = y_values_A[1:] - y_values_A[:-1]

    x_values_B, y_values_B = spiral(theta=theta_values, gamma=gamma_B, a=a_B, b=b_B)
    # Reverse the trajectory of B
    if reverse_B:
        x_values_B = x_values_B[::-1]
        y_values_B = y_values_B[::-1]
    gradients_B = np.zeros((num_sample, 2))
    gradients_B[:-1, 0] = x_values_B[1:] - x_values_B[:-1]
    gradients_B[:-1, 1] = y_values_B[1:] - y_values_B[:-1]
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
    return episodes


def draw_state_action(state, action, n_obs_steps=2, epsiodes=None):
    fig, ax = plt.subplots()
    if epsiodes is not None:
        for _i, episode in enumerate(epsiodes):
            _state = episode["state"]
            _action = episode["action"]
            # ax.plot(state[:, 0], state[:, 1], label="State", c="black")
            color = "red" if _i == 0 else "blue"
            ax.quiver(_state[:, 0], _state[:, 1], _action[:, 0], _action[:, 1], color=color, label="Action")
    ax.scatter(state[:n_obs_steps, 0], state[:n_obs_steps, 1], label="State", c="black", alpha=0.5)
    # ax.quiver(state[:, 0], state[:, 1], action[:, 0], action[:, 1], color="green", label="Action")
    # Predict state
    cum_action = np.cumsum(action, axis=0)
    pred_state = state[0] + cum_action
    ax.plot(pred_state[:, 0], pred_state[:, 1], label="Predicted State", c="green", marker="x")

    ax.axis("equal")
    plt.title("State and Action")
    plt.legend()
    plt.show()


########################################## Dataset part ##########################################
import torch
from torch.utils.data import DataLoader, Dataset


class SequenceDataset(Dataset):
    def __init__(self, episodes, sequence_length: int = 8, pad_before: int = 1, pad_after: int = 7) -> None:
        super().__init__()
        self.episodes = episodes
        self.states = []
        self.actions = []
        self.labels = []
        self.epsiode_ends = []
        for episode in episodes:
            self.states.append(episode["state"])
            self.actions.append(episode["action"])
            self.labels.append(episode["label"])
            self.epsiode_ends.append(episode["state"].shape[0])
        self.states = np.concatenate(self.states, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)
        self.labels = np.array(self.labels)
        self.epsiode_ends = np.cumsum(self.epsiode_ends)
        # Generate sample index
        episode_mask = np.ones(self.epsiode_ends.shape, dtype=bool)
        self.sequence_length = sequence_length
        self.indices = create_indices(self.epsiode_ends, sequence_length=sequence_length, episode_mask=episode_mask, pad_before=pad_before, pad_after=pad_after)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = idx % len(self.indices)
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        result = dict()
        sample_states = self.states[buffer_start_idx:buffer_end_idx]
        sample_actions = self.actions[buffer_start_idx:buffer_end_idx]
        if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
            result["state"] = np.zeros([self.sequence_length, sample_states.shape[1]])
            result["action"] = np.zeros([self.sequence_length, sample_actions.shape[1]])
            if sample_start_idx > 0:
                result["state"][:sample_start_idx] = sample_states[0]
                result["action"][:sample_start_idx] = sample_actions[0]
            if sample_end_idx < self.sequence_length:
                result["state"][sample_end_idx:] = sample_states[-1]
                result["action"][sample_end_idx:] = sample_actions[-1]
        else:
            result["state"] = sample_states
            result["action"] = sample_actions
        return result


if __name__ == "__main__":
    # Parameters
    n_obs_steps = 2
    # Generate raw data
    epsiodes = generate_raw_data(vis=True)
    print("Done!")

    # Generate dataset
    dataset = SequenceDataset(epsiodes)

    # Test dataset
    for i in range(10):
        idx = np.random.randint(len(dataset))
        data = dataset[idx]
        # print(data["state"].shape, data["action"].shape)
        draw_state_action(data["state"], data["action"], n_obs_steps, epsiodes)

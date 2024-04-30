import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusion_kernel_regression import DiffusionKernelRegressionPolicy, SequenceDataset


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


def draw_state_action(state, actions, epsiodes=None):
    fig, ax = plt.subplots()
    if epsiodes is not None:
        for _i, episode in enumerate(epsiodes):
            _state = episode["state"]
            _action = episode["action"]
            # ax.plot(state[:, 0], state[:, 1], label="State", c="black")
            color = "red" if _i == 0 else "blue"
            ax.quiver(_state[:, 0], _state[:, 1], _action[:, 0], _action[:, 1], color=color)
    ax.scatter(state[:, 0], state[:, 1], label="State", c="black", alpha=0.5)

    if actions.ndim == 2:
        actions = actions[None, :]
    for i in range(actions.shape[0]):
        # Predict state
        cum_action = np.cumsum(actions[i], axis=0)
        pred_state = state[-1] + cum_action
        pred_state = np.concatenate([state[-1][None, :], pred_state], axis=0)
        # pred_state = state[1] + cum_action[1:]
        ax.plot(pred_state[:, 0], pred_state[:, 1], c="green", marker="x", markersize=3)
    ax.axis("equal")
    plt.title("State and Action")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Parameters
    num_sample = 2000
    circle_round = 16
    reverse_B = False

    n_obs_steps = 1
    n_act_steps = 4
    horizon = 8
    diffusion_steps = 100
    knn_max = 100
    batch_size = 8
    scheduler_type = "linear"
    assert n_obs_steps + n_act_steps <= horizon, "n_obs_steps + n_act_steps should be less than horizon"
    # Generate raw data
    epsiodes = generate_raw_data(num_sample=num_sample, circle_round=circle_round, reverse_B=reverse_B, vis=True)
    print("Done!")

    # Generate dataset
    dataset = SequenceDataset(epsiodes, sequence_length=horizon, pad_before=n_obs_steps - 1, pad_after=n_act_steps - 1)

    # Generate the policy
    full_state, full_action = dataset.all_state_actions()
    state = full_state[:, :n_obs_steps, :]
    action = full_action[:, n_obs_steps - 1 : n_obs_steps - 1 + n_act_steps, :]
    policy = DiffusionKernelRegressionPolicy(
        states=state, actions=action, horizon=horizon, n_obs_steps=n_obs_steps, n_act_steps=n_act_steps, knn_max=knn_max, diffusion_steps=diffusion_steps, scheduler_type=scheduler_type
    )

    # # Test dataset
    # for i in range(10):
    #     idx = np.random.randint(len(dataset))
    #     # print(data["state"].shape, data["action"].shape)
    #     draw_state_action(state[idx, :], action[idx, :], epsiodes)

    # Test policy
    for i in range(100):
        idx = np.random.randint(len(dataset))
        data = dataset[idx]
        action_pred = policy.predict_action({"state": torch.tensor(data["state"][:n_obs_steps, :], dtype=torch.float32)}, batch_size=batch_size)
        draw_state_action(data["state"][:n_obs_steps, :], action_pred, epsiodes)

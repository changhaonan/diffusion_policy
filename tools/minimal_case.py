"""Minimal fitting case. Testing the idea of Gaussian Field."""

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm


def multi_modality_data(num_sample=200):
    z = np.random.rand(num_sample, 2).astype(np.float32)
    x1 = z[:, 0]
    x2 = z[:, 1]
    y1 = z[:, 0] ** 2
    y2 = -z[:, 1] ** 2
    xy = np.concatenate([np.stack([x1, y1], axis=-1), np.stack([x2, y2], axis=-1)], axis=0)
    return z.transpose().reshape([-1, 1]), xy


class MinimalDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MMMLP(nn.Module):
    """Multi-modal MLP."""

    def __init__(self, x_dim: int, y_dim: int, k: int = 4):
        super(MMMLP, self).__init__()
        self.fc1 = nn.Linear(x_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, k * (y_dim + 1))
        self.k = k

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, self.k, x.shape[-1] // self.k)
        return x[:, :, :-1], x[:, :, -1]

    def criterion(self, x, y, lambda_=2):
        y_pred, p_pred = self.forward(x)
        y_diff = y[:, None, :] - y_pred

        # Select the closest target
        reg_loss = torch.sum(y_diff**2, dim=-1)  # regresion loss
        # Amplify the loss of the closest target
        lowest_idx = torch.argmin(reg_loss, dim=1)
        reg_loss[torch.arange(reg_loss.shape[0]), lowest_idx] = reg_loss[torch.arange(reg_loss.shape[0]), lowest_idx] * lambda_
        reg_loss = reg_loss.mean()

        # Prob using cross entropy
        p = F.softmax(p_pred, dim=-1)
        target = torch.zeros_like(p)
        target[torch.arange(target.shape[0]), lowest_idx] = 1
        prob_loss = F.cross_entropy(p_pred, lowest_idx)

        return reg_loss + prob_loss * 0.1


class GaussianActionField(nn.Module):
    """Gaussian Action Field is bounded to samples. For samples, we bind a high-dimenstional Gaussian ball."""

    def __init__(self, x: np.ndarray, y: np.ndarray, k: int = 4, sigma_init: float = 0.01):
        """Param:
        x: (N, x_dim)
        y: (N, y_dim)
        k: max number of modalities; in Gaussian Field, explain as the number of viewing angles.
        """
        super(GaussianActionField, self).__init__()
        assert x.shape[0] == y.shape[0]
        num_sample = x.shape[0]
        x_dim = x.shape[1]
        y_dim = y.shape[1]
        project_matrix = nn.Parameter(torch.Tensor(num_sample, x_dim, k))  # project x-dim to k-dim
        project_matrix.data = torch.randn_like(project_matrix)
        self.register_parameter("project_matrix", project_matrix)
        color_matrix = nn.Parameter(torch.Tensor(num_sample, y_dim))
        color_matrix.data = torch.from_numpy(y)
        self.register_parameter("color_matrix", color_matrix)
        pos_matrix = nn.Parameter(torch.Tensor(num_sample, x_dim), requires_grad=False)
        pos_matrix.data = torch.from_numpy(x)
        self.register_parameter("pos_matrix", pos_matrix)
        sigma_matrix = nn.Parameter(torch.Tensor(num_sample, k), requires_grad=False)
        sigma_matrix.data = torch.ones_like(sigma_matrix) * sigma_init
        self.register_parameter("sigma_matrix", sigma_matrix)
        self.k = k

    def freeze(self, attr: str):
        """Freeze the attribute."""
        getattr(self, attr).requires_grad = False

    def unfreeze(self, attr: str):
        """Unfreeze the attribute."""
        getattr(self, attr).requires_grad = True

    def forward(self, x):
        """Generate the render of the Gaussian Field at x.
        x: (B, x_dim)
        return: (B, k, y_dim)
        """
        # Project x diff to k-dim
        project_matrix = self.project_matrix
        x_diff = x[:, None, :] - self.pos_matrix[None, :, :]  # (B, N, x_dim)
        x_diff = torch.einsum("bni,nio->bno", x_diff, project_matrix)  # (B, N, k)
        # Apply the Gaussian kernel
        g = self._gaussian_kernel(x_diff)  # (B, N, k)
        # Averge the color
        color_matrix = self.color_matrix  # (N, y_dim)
        color = g.transpose(1, 2) @ color_matrix  # (B, k, y_dim)
        return color

    def criterion(self, x, y):
        """Compute the loss of the Gaussian Field.
        x: (B, x_dim)
        y: (B, y_dim)
        """
        y_pred = self.forward(x)  # (B, k, y_dim)
        # Select the closest color as the target
        y_diff = y[:, None, :] - y_pred
        loss = torch.sum(y_diff**2, dim=-1)
        # Amplify the loss of the closest color
        lowest_idx = torch.argmin(loss, dim=1)
        loss[torch.arange(loss.shape[0]), lowest_idx] = loss[torch.arange(loss.shape[0]), lowest_idx] * 2
        loss = loss.mean()
        # reg_loss = self.project_regularize()
        return loss

    def project_regularize(self):
        """Regularize the norm of the project matrix."""
        project_matrix = self.project_matrix
        return torch.norm(project_matrix, dim=-1).mean()

    def _gaussian_kernel(self, x_diff):
        """Gaussian kernel; operating on k modalities seperately.
        x_diff: (B, N, k)
        """
        sigma_matrix = self.sigma_matrix
        g = torch.exp(-(x_diff**2) / (2 * sigma_matrix**2))
        return g


if __name__ == "__main__":
    z, xy = multi_modality_data(num_sample=200)
    # 3d scatter
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xy[:, 0], xy[:, 1], z, label="mode", s=5)
    ax.set_aspect("equal")
    ax.set_axis_on()
    plt.legend()
    plt.show()

    # Dataset
    dataset = MinimalDataset(z, xy)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Fitting the Gaussian Field
    # model = GaussianActionField(z.reshape([-1, 1]), xy.reshape([-1, 2]), k=2)
    model = MMMLP(1, 2, k=4).to(torch.float32)

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    total_epoch = 10000
    with tqdm(total=total_epoch) as pbar:
        for epoch in range(total_epoch):
            # model.freeze("color_matrix")
            # Freeze color matrix during first half
            # if epoch < 1000:
            #     model.freeze("color_matrix")
            # else:
            #     model.unfreeze("color_matrix")
            lambda_ = 10 * np.log(epoch + 1)  # Increasing modality biasing
            for _x, _y in loader:
                loss = model.criterion(_x, _y, lambda_=lambda_)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            pbar.update(1)
            pbar.set_description(f"loss: {loss.item():02f}")
    print("Done fitting.")

    # Inference
    z_test = np.random.rand(100, 2)
    z_test = z_test.transpose().reshape([-1, 1])
    x_test = torch.from_numpy(z_test).float()
    y_test, p_test = model.forward(x_test)
    y_test = y_test.detach().numpy()
    p_test = F.softmax(p_test, dim=-1)
    p_test = p_test.detach().numpy()

    for _k in range(model.k):
        alpha = p_test[:, _k].mean()
        print(f"Mode {_k}, alpha: {alpha}")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(y_test[:, _k, 0], y_test[:, _k, 1], z_test, label="pred", color="b", s=10, alpha=alpha)
        ax.scatter(xy[:, 0], xy[:, 1], z, label="gt", color="r", s=5)
        ax.set_aspect("equal")
        ax.set_axis_on()
        plt.legend()
        plt.show()

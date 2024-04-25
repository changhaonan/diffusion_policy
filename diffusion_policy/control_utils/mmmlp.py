"""Multi-modality MLP"""

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


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

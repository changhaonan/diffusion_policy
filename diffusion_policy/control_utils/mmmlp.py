"""Multi-modality Net"""

from typing import Union
import logging
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
from diffusion_policy.model.diffusion.conv1d_components import Downsample1d, Upsample1d, Conv1dBlock
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalResidualBlock1D

logger = logging.getLogger(__name__)


class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim, down_dims=[256, 512, 1024], kernel_size=3, n_groups=8, cond_predict_scale=False):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        cond_dim = global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale),
            ]
        )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale),
                        ConditionalResidualBlock1D(dim_out, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale),
                        ConditionalResidualBlock1D(dim_in, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def forward(self, sample: torch.Tensor, global_cond: torch.Tensor, **kwargs):
        """
        x: (B, T, input_dim)
        global_cond: (B, global_cond_dim)
        output: (B, T, input_dim)
        """
        sample = einops.rearrange(sample, "b h t -> b t h")
        global_feature = global_cond

        # encode local features
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b t h -> b h t")
        return x


class MMNet(nn.Module):
    """Multi-modal MLP."""

    def __init__(self, obs_dim: int, act_dim: int, obs_steps: int, action_steps: int, k: int = 4, down_dims=[256, 512, 1024], kernel_size=3, n_groups=8, **kwargs):
        super(MMNet, self).__init__()
        # Use conditional unet1d
        self.act_net_list = nn.ModuleList()
        for _ in range(k):
            self.act_net_list.append(ConditionalUnet1D(input_dim=act_dim, global_cond_dim=obs_dim * obs_steps, down_dims=down_dims, kernel_size=kernel_size, n_groups=n_groups))
        self.k = k
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.action_steps = action_steps
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def forward(self, obs):
        # Init zero query
        x = torch.zeros(obs.shape[0], self.action_steps, self.act_dim, device=obs.device)
        ys = []
        for net in self.act_net_list:
            x = net(x, obs)
            ys.append(x)
        y = torch.stack(ys, dim=1)  # (B, k, T, act_dim)
        y = y.view(y.shape[0], self.k, -1)  # (B, k, T * act_dim)
        return y, torch.ones(y.shape[0], self.k, device=y.device) / self.k

    def criterion(self, x, y, lambda_=2):
        y_pred, p_pred = self.forward(x)
        y_diff = y[:, None, :] - y_pred

        # Select the closest target
        reg_loss = torch.sum(y_diff**2, dim=-1)  # regresion loss
        # Amplify the loss of the closest target
        lowest_idx = torch.argmin(reg_loss, dim=1)
        reg_loss[torch.arange(reg_loss.shape[0]), lowest_idx] = reg_loss[torch.arange(reg_loss.shape[0]), lowest_idx] * lambda_
        reg_loss = reg_loss.mean()

        # # Prob using cross entropy
        # p = F.softmax(p_pred, dim=-1)
        # target = torch.ones_like(p)
        # target[torch.arange(target.shape[0]), lowest_idx] = 1
        # prob_loss = F.cross_entropy(p_pred, lowest_idx)

        return reg_loss

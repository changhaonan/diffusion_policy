"""Controllable U-Net for 1D diffusion model."""

from typing import Union
import torch
import einops
import logging
import torch.nn as nn
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.diffusion.conv1d_components import Downsample1d, Upsample1d, Conv1dBlock
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D, ConditionalResidualBlock1D

logger = logging.getLogger(__name__)


class ControlUnet1D(ConditionalUnet1D):
    def __init__(
        self,
        input_dim,
        control_cond_dim=None,
        only_mid_control=False,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
        integrate_type="concat",
        **kwargs,
    ):
        if integrate_type == "concat":
            global_cond_dim = control_cond_dim + global_cond_dim
        super().__init__(input_dim, local_cond_dim, global_cond_dim, diffusion_step_embed_dim, down_dims, kernel_size, n_groups, cond_predict_scale)
        self.only_mid_control = only_mid_control
        self.integrate_type = integrate_type
        if control_cond_dim is None or (integrate_type == "concat"):
            # No explicit control
            self.control_modules = nn.ModuleList([])
        else:
            all_dims = [input_dim] + list(down_dims)
            start_dim = down_dims[0]

            dsed = diffusion_step_embed_dim
            cond_dim = dsed + control_cond_dim

            in_out = list(zip(all_dims[:-1], all_dims[1:]))

            self.control_modules = nn.ModuleList([])
            for ind, (dim_in, dim_out) in enumerate(in_out):
                is_last = ind >= (len(in_out) - 1)
                self.control_modules.append(
                    nn.ModuleList(
                        [
                            ConditionalResidualBlock1D(dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale),
                            ConditionalResidualBlock1D(dim_out, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale),
                            Downsample1d(dim_out) if not is_last else nn.Identity(),
                        ]
                    )
                )
        logger.info("number of parameters: %e after add control", sum(p.numel() for p in self.parameters()))

    def forward(self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int], control_cond=None, global_cond=None, **kwargs):
        """
        x: (B, T, input_dim)
        timestep: (B,) or int, diffusion step
        control_cond: (B, control_cond_dim)
        global_cond: (B, global_cond_dim)
        output: (B, T, input_dim)
        """
        sample = einops.rearrange(sample, "b h t -> b t h")

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        timestep_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            if control_cond is not None and self.integrate_type == "concat":
                global_cond = torch.cat([global_cond, control_cond], axis=-1)
                control_cond = None
            global_feature = torch.cat([timestep_feature, global_cond], axis=-1)
        else:
            global_feature = timestep_feature
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        # Control part
        if control_cond is not None:
            control_feature = torch.cat([timestep_feature, control_cond], axis=-1)
        else:
            control_feature = timestep_feature
        x_c = sample
        h_c = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.control_modules):
            x_c = resnet(x_c, control_feature)
            x_c = resnet2(x_c, control_feature)
            h_c.append(x_c)
            x_c = downsample(x_c)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        if h_c:
            x += h_c.pop()  # mid control

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            # The correct condition should be:
            # if idx == (len(self.up_modules)-1) and len(h_local) > 0:
            # However this change will break compatibility with published checkpoints.
            # Therefore it is left as a comment.
            x = resnet2(x, global_feature)
            x = upsample(x)
            if not self.only_mid_control and h_c:
                x += h_c.pop()

        x = self.final_conv(x)

        x = einops.rearrange(x, "b t h -> b h t")
        return x

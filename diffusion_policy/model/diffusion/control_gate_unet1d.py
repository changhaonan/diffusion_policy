"""Controllable U-Net for 1D diffusion model with gate design."""

from typing import Union
import torch
import einops
import logging
import torch.nn as nn
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.diffusion.conv1d_components import Downsample1d, Upsample1d, Conv1dBlock
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D, ConditionalResidualBlock1D

logger = logging.getLogger(__name__)


class ControlGateUnet1D(nn.Module):
    """
    The idea of gate control unet is: We encode the control signal in encoder;
    In the decoder part, we have a gate token that controls if the system should be controlled or not.
    The system is forced to decouple the control signal from the input signal.
    """

    def __init__(
        self,
        input_dim,
        control_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
        integrate_type="concat",
        **kwargs,
    ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        gate_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim
            if control_cond_dim is not None and integrate_type == "concat":
                cond_dim += control_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        # Encoders
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

        # Decoders
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim + dsed, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim + dsed, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale),
            ]
        )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim=cond_dim + dsed, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale),
                        ConditionalResidualBlock1D(dim_in, dim_in, cond_dim=cond_dim + dsed, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.gate_encoder = gate_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv
        self.integrate_type = integrate_type
        self.control_cond_dim = control_cond_dim
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def forward(self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int], gate=Union[torch.Tensor, int], control_cond=None, global_cond=None, **kwargs):
        """
        x: (B, T, input_dim)
        timestep: (B,) or int, diffusion step
        control_cond: (B, control_cond_dim)
        global_cond: (B, global_cond_dim)
        output: (B, T, input_dim)
        gate: (B,)
        """
        sample = einops.rearrange(sample, "b h t -> b t h")

        # time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        # gate
        gates = gate
        if not torch.is_tensor(gates):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            gates = torch.tensor([gates], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(gates) and len(gates.shape) == 0:
            gates = gates[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        gates = gates.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        gate_feature = self.gate_encoder(gates)

        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)
            if control_cond is not None and self.integrate_type == "concat":
                global_feature = torch.cat([global_feature, control_cond], axis=-1)
                control_cond = None

        x = sample
        h = []
        # Encoding; Add control
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        # Decoding; Add gate
        global_feature_gate = torch.cat([gate_feature, global_feature], axis=-1)
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature_gate)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature_gate)
            x = resnet2(x, global_feature_gate)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b t h -> b h t")
        return x

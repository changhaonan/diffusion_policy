"""Diffusion UNet with low-dimensional state-action policy."""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_sa_policy import BaseSAPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator


class DiffusionUnetLowdimSAPolicy(BaseSAPolicy):
    def __init__(
        self,
        model: ConditionalUnet1D,
        noise_scheduler: DDPMScheduler,
        horizon,
        obs_dim,
        action_dim,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_local_cond=False,
        obs_as_global_cond=False,
        pred_action_steps_only=False,
        oa_step_convention=False,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        assert local_cond is None and global_cond is not None
        model = self.model
        scheduler = self.noise_scheduler
        trajectory = torch.randn(size=condition_data.shape, dtype=condition_data.dtype, device=condition_data.device, generator=generator)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # predict model output
            model_output = model(trajectory, t, local_cond=None, global_cond=global_cond)

            # compute previous sample
            trajectory = scheduler.step(model_output, t, trajectory, generator=generator, **kwargs).prev_sample

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" & "next_obs" key
        result: must include "action" key and "next_obs" key
        """

        assert "obs" in obs_dict, "obs key must be in obs_dict"
        assert "next_obs" in obs_dict, "next_obs key must be in obs_dict"
        nobs = self.normalizer["obs"].normalize(obs_dict["obs"])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        global_cond = nobs[:, :To].reshape(nobs.shape[0], -1)
        cond_data = torch.zeros(size=(B, self.n_action_steps, Da + Do), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling
        nsample = self.conditional_sample(cond_data, cond_mask, local_cond=None, global_cond=global_cond, **self.kwargs)

        # unnormalize prediction
        action_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(action_pred)
        next_obs_pred = nsample[..., Da:]
        next_obs_pred = self.normalizer["next_obs"].unnormalize(next_obs_pred)

        # trunc output
        act_start = To
        act_end = act_start + self.n_action_steps
        action = action_pred[:, act_start:act_end]

        nobs_end = act_end
        nobs_start = nobs_end - To
        next_obs = next_obs_pred[:, nobs_start:nobs_end]

        result = {
            "action": action,
            "action_pred": action_pred,
            "next_obs": next_obs,
            "next_obs_pred": next_obs_pred,
        }
        return result

    # ========== training ===========
    def set_normalizer(self, normalizer: LinearNormalizer):
        return super().set_normalizer(normalizer)

    def compute_loss(self, batch):
        # normalize input
        assert "valid_mask" not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch["obs"]
        action = nbatch["action"]
        next_obs = nbatch["next_obs"]

        # Sample noise that we'll add to the images
        trajectory = torch.cat([obs, next_obs], dim=-1)
        global_cond = obs[:, : self.n_obs_steps, :].reshape(obs.shape[0], -1)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each sample
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=trajectory.device).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, local_cond=None, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "residual":
            pred = noisy_trajectory + pred
        elif pred_type == "conditional":
            pred = pred
        else:
            raise ValueError(f"Invalid prediction type {pred_type}")

        loss = F.mse_loss(pred, trajectory, reduction="none")
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss

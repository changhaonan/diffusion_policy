import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy


class ControlDiffusionUnetImagePolicy(DiffusionUnetImagePolicy):
    def __init__(self, integrate_type, **kwargs):
        super().__init__(**kwargs)
        self.integrate_type = integrate_type
        if self.integrate_type == "overlay" or self.integrate_type == "concat":
            self.control_model = None  # control model is not used
        else:
            raise NotImplementedError(f"integrate_type {self.integrate_type} not supported")
        assert self.obs_as_global_cond, "control diffusion policy requires obs_as_global_cond=True"

    def _generate_condition(self, nobs):
        """
        nobs: normalized observation dict
        return: condition_data, condition_mask, obs_cond, control_data
        """
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # condition through global feature
        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        # reshape back to B, Do
        obs_cond = nobs_features.reshape(B, -1)
        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        if self.control_model is None:
            control_data = None
        else:
            raise NotImplementedError(f"integrate_type {self.integrate_type} not supported")
        return cond_data, cond_mask, obs_cond, control_data

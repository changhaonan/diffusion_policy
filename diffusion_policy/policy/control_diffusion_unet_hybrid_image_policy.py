from typing import Dict, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.control_gate_unet1d import ControlGateUnet1D
from diffusion_policy.model.diffusion.control_unet1d import ControlUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
from robomimic.models.obs_nets import ObservationEncoder
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


class ControlDiffusionUnetHybridImagePolicy(BaseImagePolicy):
    """Based on DiffusionUnetHybridImagePolicy; Control signal is an image."""

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_global_cond=True,
        crop_shape=(76, 76),
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        obs_encoder_group_norm=False,
        eval_fixed_crop=False,
        # parameters passed to step,
        control_model="control_gate_unet",
        integrate_type="concat",
        only_mid_control=False,
        cfg_ratio=0.3,
        **kwargs,
    ):
        """Integrate type:
        1. concat: concatenate obs and control features.
        2. controlnet: use to apply explicit control signal.
        """
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta["obs"]
        obs_config = {"low_dim": [], "rgb": [], "depth": [], "scan": []}
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr["shape"]
            obs_key_shapes[key] = list(shape)

            type = attr.get("type", "low_dim")
            if type == "rgb":
                obs_config["rgb"].append(key)
            elif type == "low_dim":
                obs_config["low_dim"].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(algo_name="bc_rnn", hdf5_type="image", task_name="square", dataset_type="ph")

        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality["obs_randomizer_class"] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=action_dim,
            device="cpu",
        )

        obs_encoder: ObservationEncoder = policy.nets["policy"].nets["encoder"].nets["obs"]

        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(root_module=obs_encoder, predicate=lambda x: isinstance(x, nn.BatchNorm2d), func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features))
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets

        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(input_shape=x.input_shape, crop_height=x.crop_height, crop_width=x.crop_width, num_crops=x.num_crops, pos_enc=x.pos_enc),
            )

        # create diffusion model
        obs_feature_dim, control_feature_dim = self._compute_obs_control_shape(obs_encoder)

        input_dim = action_dim + obs_feature_dim
        input_dim = action_dim
        global_cond_dim = obs_feature_dim * n_obs_steps
        control_cond_dim = control_feature_dim * n_obs_steps if control_feature_dim is not None else None

        if control_model == "control_gate_unet":
            model_fn = ControlGateUnet1D
        elif control_model == "control_unet":
            model_fn = ControlUnet1D
        else:
            raise ValueError(f"Unsupported control model: {control_model}")
        model = model_fn(
            input_dim=input_dim,
            control_cond_dim=control_cond_dim,
            only_mid_control=only_mid_control,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            integrate_type=integrate_type,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(action_dim=action_dim, obs_dim=0 if obs_as_global_cond else obs_feature_dim, max_n_obs_steps=n_obs_steps, fix_obs_steps=True, action_visible=False)
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.control_feature_dim = control_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        ################################ Control related parameters ################################
        self.integrate_type = integrate_type
        self.use_cfg = cfg_ratio > 0
        self.cfg_ratio = cfg_ratio
        self.mask_prob = 0.1 if self.use_cfg else 0.0
        assert self.obs_as_global_cond, "control diffusion policy requires obs_as_global_cond=True"
        if self.use_cfg:
            assert not (control_model == "control_gate_unet"), "control_gate_unet requires cfg_ratio=0"
        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))

    def _compute_obs_control_shape(self, obs_encoder):
        obs_dim = 0
        control_dim = 0
        for k in obs_encoder.obs_shapes:
            feat_shape = obs_encoder.obs_shapes[k]
            if obs_encoder.obs_randomizers[k] is not None:
                feat_shape = obs_encoder.obs_randomizers[k].output_shape_in(feat_shape)
            if obs_encoder.obs_nets[k] is not None:
                feat_shape = obs_encoder.obs_nets[k].output_shape(feat_shape)
            if obs_encoder.obs_randomizers[k] is not None:
                feat_shape = obs_encoder.obs_randomizers[k].output_shape_out(feat_shape)
            if k == "control":
                control_dim = int(np.prod(feat_shape))
            else:
                obs_dim += int(np.prod(feat_shape))
        return obs_dim, control_dim

    # ========= inference  ============
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        gate=1,
        control_cond=None,
        global_cond=None,
        generator=None,
        # keyword arguments to scheduler.step\
        **kwargs,
    ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(size=condition_data.shape, dtype=condition_data.dtype, device=condition_data.device, generator=generator)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, gate=gate, control_cond=control_cond, global_cond=global_cond)
            if self.use_cfg:
                uncontrol_cond = torch.zeros_like(control_cond)
                uncontrol_model_output = model(trajectory, t, gate=gate, control_cond=uncontrol_cond, global_cond=global_cond)
                model_output = model_output + self.cfg_ratio * (model_output - uncontrol_model_output)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(model_output, t, trajectory, generator=generator, **kwargs).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], gate: Union[int, torch.Tensor] = 1) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert "past_action" not in obs_dict  # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        global_cond = None
        # condition through global feature
        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))

        nobs_features = self.obs_encoder(this_nobs)  # (agent_pos, image, control)
        nobs_features, ncontrol_features = nobs_features[:, :Do], nobs_features[:, Do:]
        # reshape back to B, Do
        global_cond = nobs_features.reshape(B, -1)
        control_cond = ncontrol_features.reshape(B, -1)
        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling
        nsample = self.conditional_sample(cond_data, cond_mask, gate=gate, control_cond=control_cond, global_cond=global_cond, **self.kwargs)

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {"action": action, "action_pred": action_pred}
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        global_cond = None
        trajectory = nactions
        cond_data = trajectory

        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)  # (agent_pos, image, control)
        nobs_features, ncontrol_features = nobs_features[:, : self.obs_feature_dim], nobs_features[:, self.obs_feature_dim :]

        if self.use_cfg:
            # randomly mask control signal
            cfg_mask = torch.rand(batch_size, 1, device=self.device) < self.mask_prob
            cfg_mask = torch.repeat_interleave(cfg_mask, self.n_obs_steps, dim=1).reshape(-1, 1)
            ncontrol_features = ncontrol_features * ~cfg_mask
        # reshape back to B, Do
        global_cond = nobs_features.reshape(batch_size, -1)
        control_cond = ncontrol_features.reshape(batch_size, -1)

        demo_type = batch["demo_type"][:, 0, 0]  # (B,)
        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=trajectory.device).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, control_cond=control_cond, global_cond=global_cond, gate=demo_type)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss

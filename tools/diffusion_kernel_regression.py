import numpy as np
import matplotlib.pyplot as plt
import numba
from copy import deepcopy
import math


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


########################################## Dataset part ##########################################
import os
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Union
from diffusers import AutoencoderKL
from PIL import Image
from diffusers.image_processor import VaeImageProcessor
import torch
import hashlib
import tqdm


class SequenceDataset(Dataset):
    """Sequential 1D dataset."""

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
        self.pad_before = pad_before
        self.pad_after = pad_after
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

    def all_state_actions(self):
        # Generate all state action pairs
        state_list = []
        action_list = []
        for idx in range(len(self)):
            data = self[idx]
            state_list.append(data["state"])
            action_list.append(data["action"])
        return np.stack(state_list), np.stack(action_list)


class ImageDataset(Dataset):
    """Image dataset."""

    def __init__(self, image_root_dir):
        super().__init__()
        self.image_root_dir = image_root_dir
        self.image_files = os.listdir(image_root_dir)
        self.image_files = [f for f in self.image_files if f.endswith(".png") or f.endswith(".jpg")]
        self.image_files = [os.path.join(image_root_dir, f) for f in self.image_files]
        # Load VAE
        self.dev = 0
        self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
        self.vae = self.vae.to(self.dev)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.latents = []
        self.encode_images(use_cache=True)

    def encode_images(self, use_cache: bool = True):
        cache_file = os.path.join(self.image_root_dir, "latents.npy")
        if use_cache and os.path.exists(cache_file):
            self.latents = np.load(cache_file).squeeze()
            print("Loaded latents from cache...")
            return self.latents
        else:
            # Encode images
            self.latents = []
            for image_file in tqdm.tqdm(self.image_files):
                img = Image.open(image_file)
                pixel_values = self.image_processor.numpy_to_pt(self.image_processor.normalize(self.image_processor.resize(self.image_processor.pil_to_numpy(img), 255, 255)))
                latents = self.vae.encode(pixel_values.to(self.dev)).latent_dist.sample()
                self.latents.append(latents.detach().cpu().numpy())
            self.latents = np.stack(self.latents, axis=0)
            np.save(cache_file, self.latents)
            return self.latents

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        idx = idx % len(self.image_files)
        image = Image.open(self.image_files[idx])
        pixel_values = self.image_processor.numpy_to_pt(self.image_processor.normalize(self.image_processor.resize(self.image_processor.pil_to_numpy(image), 255, 255)))
        latents = self.latents[idx]
        return {"image": pixel_values, "latent": latents}


########################################## Algorithm ##########################################
class DiffusionKernelRegression:
    """Diffusion kernel regression."""

    def __init__(self, datas, conditions=None, knn_max: int = 32, diffusion_steps: int = 100, scheduler_type: str = "linear") -> None:
        self.datas = datas
        if conditions is not None:
            assert datas.shape[0] == conditions.shape[0], "Samples and conditions should have the same length"
            self.conditions = conditions
        else:
            self.conditions = datas
        self.clip_sample = True
        self.use_robust_kernel = True
        self.partition_threshold = 0.0
        self.knn_max = knn_max
        self.diffusion_steps = diffusion_steps

        self.stats = {}
        self._compute_stats(self.datas, "datas")
        self.datas = self._normalize(self.datas, self.stats["datas"])
        if conditions is not None:
            self._compute_stats(self.conditions, "conditions")
            self.conditions = self._normalize(self.conditions, self.stats["conditions"])
        self.alpha_t, self.beta_t, self.alpha_bar_t, self.sigma_t, self.h_t = self._get_scheduler(scheduler_type=scheduler_type)

    def conditional_sampling(self, condition=None, batch_size=4, **kwargs):
        # Normalize the condition
        if condition is not None:
            if isinstance(condition, torch.Tensor):
                condition = condition.cpu().numpy()
            condition = np.reshape(condition, [-1, self.conditions.shape[-1]])
            condition = self._normalize(condition, self.stats["conditions"])
            local_conditions, local_datas = self._compute_neighbors(condition, self.knn_max)
        else:
            local_datas = self.datas
            local_conditions = None
        samples = []
        for _iter in tqdm.tqdm(range(batch_size)):
            sample = np.random.randn(1, self.datas.shape[-1])
            for i in tqdm.tqdm(range(self.diffusion_steps - 1, -1, -1), leave=False):
                data_diff = sample - local_datas
                condition_diff = condition - local_conditions if condition is not None else np.zeros_like(data_diff)
                kernel, partition = self._compute_kernel(data_diff, condition_diff, self.h_t[i], robust=self.use_robust_kernel)
                # Kernel regression
                data_pred = np.sum(kernel[:, None] * local_datas, axis=0) / np.sum(kernel)
                print(f"{_iter}| Current partition: {partition}")
                if i > 20 and partition > self.partition_threshold:
                    # Update the step
                    sample = (
                        np.sqrt(self.alpha_t[i]) * (1 - self.alpha_bar_t[i - 1]) / (1 - self.alpha_bar_t[i]) * sample
                        + np.sqrt(self.alpha_bar_t[i - 1]) * self.beta_t[i] / (1 - self.alpha_bar_t[i]) * data_pred
                    )
                    # Add noise
                    sample += self.sigma_t[i - 1] * np.random.randn(1, self.datas.shape[-1])
                else:
                    break
                if self.clip_sample:
                    sample = np.clip(sample, -1, 1)
            samples.append(sample)
        samples = np.stack(samples, axis=0)
        # Unnormalize the datas
        samples = self._unnormalize(samples, self.stats["datas"])
        return samples

    def _compute_stats(self, data, key):
        self.stats[key] = {}
        self.stats[key]["max"] = np.max(data, axis=0)
        self.stats[key]["min"] = np.min(data, axis=0)

    def _normalize(self, data, stats):
        return ((data - stats["min"]) / (stats["max"] - stats["min"])) * 2 - 1

    def _unnormalize(self, data, stats):
        return ((data + 1) / 2) * (stats["max"] - stats["min"]) + stats["min"]

    def _compute_kernel(self, data_diff, condition_diff, h_t, robust=True):
        if not robust:
            kernel = np.exp(-(np.linalg.norm(data_diff, axis=1) ** 2 + np.linalg.norm(condition_diff, axis=1) ** 2) / (2 * h_t**2))
            kernel = np.clip(kernel, 1e-8, 1)
            partition = np.sum(kernel)
        else:
            k_diff = -(np.linalg.norm(data_diff, axis=1) ** 2 + np.linalg.norm(condition_diff, axis=1) ** 2)
            # FIXME: Adjust by dimenstion
            # k_diff = k_diff / np.sqrt(data_diff.shape[-1])
            partition = np.sum(np.exp(k_diff / (2 * h_t**2)))
            min_k_diff = np.max(k_diff)
            k_diff = k_diff - min_k_diff
            kernel = np.exp(k_diff / (2 * h_t**2))
            kernel = np.clip(kernel, 1e-8, 1)
        return kernel, partition

    def _compute_neighbors(self, condition, knn_max):
        # Compute neighbors based on the state
        dist = np.linalg.norm(self.conditions - condition[None, :], axis=-1).squeeze()
        idx = np.argsort(dist)[:knn_max].squeeze()
        return self.conditions[idx, :], self.datas[idx]

    def _get_scheduler(self, beta_start=0.0001, beta_end=0.02, scheduler_type="linear"):
        # Get the scheduler for the diffusion steps; alpha_t, beta_t
        t = np.arange(self.diffusion_steps)
        if scheduler_type == "linear":
            beta_t = np.linspace(beta_start, beta_end, self.diffusion_steps)
        elif scheduler_type == "quadratic":
            beta_t = np.linspace(beta_start**0.5, beta_end**0.5, self.diffusion_steps) ** 2
        elif scheduler_type == "squaredcos_cap_v2":
            beta_t = self._betas_for_alpha_bar(self.diffusion_steps)
        else:
            raise ValueError("Invalid scheduler type")
        alpha_t = 1 - beta_t
        alpha_bar_t = np.cumprod(alpha_t)
        sigma_t = (1 - alpha_bar_t[1:]) / (1 - alpha_bar_t[:-1]) * beta_t[1:]
        h_t = np.sqrt((1 - alpha_bar_t) / alpha_bar_t)
        return alpha_t, beta_t, alpha_bar_t, sigma_t, h_t

    def _betas_for_alpha_bar(self, num_diffusion_timesteps, max_beta=0.999, alpha_transform_type="cosine"):
        if alpha_transform_type == "cosine":

            def alpha_bar_fn(t):
                return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

        elif alpha_transform_type == "exp":

            def alpha_bar_fn(t):
                return math.exp(t * -12.0)

        else:
            raise ValueError(f"Unsupported alpha_tranform_type: {alpha_transform_type}")

        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
        return np.array(betas)


########################################## Policy part ##########################################
class DiffusionKernelRegressionPolicy(DiffusionKernelRegression):
    """Estimated diffusion kernel regression."""

    def __init__(self, states, actions, horizon: int = 8, n_obs_steps: int = 2, n_act_steps: int = 4, knn_max: int = 30, diffusion_steps: int = 100, scheduler_type: str = "linear", **kwargs):
        # Reshape the states and actions
        states = np.reshape(states, [states.shape[0], -1])
        actions = np.reshape(actions, [actions.shape[0], -1])
        super().__init__(datas=actions, conditions=states, knn_max=knn_max, diffusion_steps=diffusion_steps, scheduler_type=scheduler_type)
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_act_steps = n_act_steps

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], **kwargs):
        batch_size = kwargs.get("batch_size", 4)
        obs = obs_dict["state"].cpu().numpy()
        obs = np.reshape(obs, [obs.shape[0], -1])
        actions = self.conditional_sampling(condition=obs, batch_size=batch_size)
        # Reshape the actions
        actions = np.reshape(actions, [actions.shape[0], self.n_act_steps, -1])
        return actions


########################################## Image part ##########################################
from PIL import Image


class DiffusionKernelRegressionImage(DiffusionKernelRegression):
    """Diffusion kernel image generator."""

    def __init__(self, latents, vae: AutoencoderKL, image_processor: VaeImageProcessor, knn_max: int = 100, diffusion_steps: int = 100, scheduler_type: str = "linear", **kwargs):
        self.vae = vae
        self.image_processor = image_processor
        self.latent_size = latents.shape[1:]
        latents = latents.reshape([latents.shape[0], -1])
        super().__init__(datas=latents, knn_max=knn_max, diffusion_steps=diffusion_steps, scheduler_type=scheduler_type)

    def sample_image(self, batch_size: int = 4, **kwargs):
        check_knn = kwargs.get("check_knn", False)
        samples = self.conditional_sampling(batch_size=batch_size)
        samples = np.reshape(samples, (samples.shape[0],) + self.latent_size)
        # Decode the image
        image_list = []
        for i in range(samples.shape[0]):
            sample = samples[i][None, ...]
            scale_factor = 1.0
            sample = sample / scale_factor
            decode_pixel_values = self.vae.decode(torch.tensor(sample, dtype=torch.float32).to(self.vae.device), return_dict=False)[0]
            image = self.image_processor.postprocess(decode_pixel_values.detach(), do_denormalize=[True])[0]
            if check_knn:
                # Need to check the nearest image
                sample_key = np.reshape(sample, [sample.shape[0], -1])
                nearest_latents, _ = self._compute_neighbors(sample_key, 1)
                nearest_latents = np.reshape(nearest_latents, self.latent_size)
                nearest_latents = nearest_latents / scale_factor
                nearest_decode_pixel_values = self.vae.decode(torch.tensor(nearest_latents[None, ...], dtype=torch.float32).to(self.vae.device), return_dict=False)[0]
                nearest_image = self.image_processor.postprocess(nearest_decode_pixel_values.detach(), do_denormalize=[True])[0]
                # Concatenate the image
                concat_image = Image.new("RGB", (image.width * 2, image.height))
                concat_image.paste(image, (0, 0))
                concat_image.paste(nearest_image, (image.width, 0))
                image = concat_image
            image_list.append(image)
        return image_list

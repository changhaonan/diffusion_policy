"""Diffusion Trainer."""
import os
import math
import numpy as np

from diffusion_kernel_regression import DiffusionKernelRegression

class DiffusionTrainer:
    """Diffusion Trainer."""

    def __init__(self) -> None:
        # Params
        self.stats = {}
    
    ############################# UTILS #############################
    def _compute_stats(self, data, key):
        self.stats[key] = {}
        self.stats[key]["max"] = np.max(data, axis=0)
        self.stats[key]["min"] = np.min(data, axis=0)

    def _normalize(self, data, stats):
        return ((data - stats["min"]) / (stats["max"] - stats["min"])) * 2 - 1

    def _unnormalize(self, data, stats):
        return ((data + 1) / 2) * (stats["max"] - stats["min"]) + stats["min"]

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
        # h_t = (1 - alpha_bar_t) / alpha_bar_t
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
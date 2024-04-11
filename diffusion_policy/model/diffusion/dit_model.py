from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
from diffusion_policy.model.diffusion.conv1d_components import Downsample1d, Upsample1d, Conv1dBlock
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from timm.models.vision_transformer import PatchEmbed, Mlp

logger = logging.getLogger(__name__)


################################ DiT Block ################################
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    """Attention with mask support."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # Apply mask to attention scores
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head dimension
            attn = attn.masked_fill(mask, float("-inf"))  # Apply mask where mask is True

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c, mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


################################ DiT Model ################################
class DiTModel(ModuleAttrMixin):
    """Diffusion Transformer (DiT) model."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: int = None,
        cond_dim: int = 0,
        n_layer: int = 12,
        n_head: int = 12,
        n_emb: int = 768,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        causal_attn: bool = False,
        time_as_cond: bool = True,
        obs_as_cond: bool = False,
        n_cond_layers: int = 0,
    ) -> None:
        super().__init__()
        # Parameters
        self.use_final_conv = True
        self.long_skip = True
        # compute number of tokens for main trunk and condition encoder
        if n_obs_steps is None:
            n_obs_steps = horizon

        T = horizon
        T_cond = 1
        assert time_as_cond, "Time as condition is required"
        assert cond_dim > 0, "Conditioning dimension must be greater than 0"
        T_cond += n_obs_steps

        # input embedding stem
        self.input_emb = nn.Linear(input_dim, n_emb * T_cond)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb * T_cond))
        self.drop = nn.Dropout(p_drop_emb)

        # cond encoder
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = nn.Linear(cond_dim, n_emb)

        # decoder head
        self.ln_f = nn.LayerNorm(n_emb * T_cond)
        self.head = nn.Linear(n_emb * T_cond, output_dim)
        self.final_conv = nn.Conv1d(output_dim, output_dim, 1) if self.use_final_conv else None

        self.dit_blocks = nn.ModuleList()
        for i in range(n_layer):
            self.dit_blocks.insert(0, DiTBlock(hidden_size=n_emb * T_cond, num_heads=n_head, attn_drop=p_drop_attn, proj_drop=p_drop_attn))
        self.long_skip_proj = nn.ModuleList()
        for i in range(n_layer // 2):
            self.long_skip_proj.append(nn.Linear(2 * n_emb * T_cond, n_emb * T_cond))
        # init
        self.initialize_weights()

    def forward(self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int], cond: Optional[torch.Tensor] = None, **kwargs):
        # time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        # (B, 1, n_emb)

        # cond
        cond_embeddings = time_emb
        cond_obs_emb = self.cond_obs_emb(cond)
        cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)
        # (B, n_obs_steps, n_emb)

        x = self.input_emb(sample)
        pos_embedding = self.pos_emb[:, : x.shape[1]]

        # (B, T_pred, n_emb)
        h = []
        # Do DiT
        cond_embeddings = cond_embeddings.reshape(cond_embeddings.shape[0], -1)  # (B, n_emb * (n_obs_steps + 1))
        x = x + pos_embedding  # Add positional embedding
        for i in range(len(self.dit_blocks)):
            # Mask out the padding
            x = self.dit_blocks[i](x, cond_embeddings)
            if self.long_skip:
                if i < (len(self.dit_blocks) // 2):
                    h.append(x)
                elif i >= (len(self.dit_blocks) // 2) and h and i != len(self.dit_blocks) - 1:
                    # Long skip connection as in UViT
                    x = torch.cat([x, h.pop()], dim=-1)
                    x = self.long_skip_proj[i - len(self.dit_blocks) // 2](x)
        # head
        x = self.ln_f(x)
        x = self.head(x)
        # (B, T, n_out)
        if self.final_conv is not None:
            x = x.transpose(1, 2)
            x = self.final_conv(x)
            x = x.transpose(1, 2)
        return x

    def initialize_weights(self):
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.dit_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def get_optim_groups(self, weight_decay: float = 0.0):
        return [{"params": self.parameters()}]

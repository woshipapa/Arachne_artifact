import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.autoencoders.autoencoder_kl_cogvideox import (
    CogVideoXCausalConv3d as _CogVideoXCausalConv3d,
)
from diffusers.models.downsampling import CogVideoXDownsample3D
from diffusers.models.modeling_utils import ModelMixin
from torch.nn import functional as F
from einops import rearrange


class GuiderModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, guider_cfg):
        super().__init__()
        guider_cfg = guider_cfg.copy()
        guider_type = guider_cfg.pop("type")
        self.guider = globals()[guider_type](**guider_cfg)
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def forward(self, *args, **kwargs):
        if self.gradient_checkpointing:
            output = torch.utils.checkpoint.checkpoint(
                self.guider, *args, use_reentrant=False, **kwargs
            )
        else:
            output = self.guider(*args, **kwargs)
        return output


class GuiderNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=4,
        block_out_channels=(16, 32, 96, 256),
        init_zero=True,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=3, padding=1
        )
        self.blocks = nn.ModuleList([])
        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(
                nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            self.blocks.append(nn.SiLU())
            self.blocks.append(
                nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2)
            )
            self.blocks.append(nn.SiLU())
        self.conv_out = nn.Conv2d(
            block_out_channels[-1], out_channels, kernel_size=3, padding=1
        )
        if init_zero:
            self.conv_out = zero_module(self.conv_out)

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)
        for block in self.blocks:
            embedding = block(embedding)
        embedding = self.conv_out(embedding)
        return embedding


class GuiderNet2(nn.Module):
    def __init__(self, in_channels=3, mid_channels=4, out_channels=8):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(mid_channels, mid_channels, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(mid_channels, out_channels, 4, 2, 1),
        )

    def forward(self, x):
        return self.layers(x)


class PreNormattention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class PreAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        dots = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = self.attend(dots)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        heads=8,
        dim=2048,
        dim_head_k=256,
        dim_head_v=256,
        dropout_atte=0.05,
        mlp_dim=2048,
        dropout_ffn=0.05,
        depth=1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNormattention(
                            dim,
                            PreAttention(
                                dim,
                                heads=heads,
                                dim_head=dim_head_k,
                                dropout=dropout_atte,
                            ),
                        ),
                        FeedForward(dim, mlp_dim, dropout=dropout_ffn),
                    ]
                )
            )

        self.dwpose_embedding = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(dim * 4, dim * 4, 3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(dim * 4, dim, 3, stride=1, padding=1),
        )

    def forward(self, x):
        batch, c, f, h, w = x.shape
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = self.dwpose_embedding(x)
        x = rearrange(x, "(b f) c h w -> (b h w) f c", b=batch)
        for attn, ff in self.layers[:1]:
            x = attn(x)
            x = ff(x) + x
        if self.depth > 1:
            for attn, ff in self.layers[1:]:
                x = attn(x)
                x = ff(x) + x
        x = rearrange(x, "(b h w) f c -> b c f h w", b=batch, h=h)
        return x


class GuiderNet3DCogVideoX(nn.Module):
    def __init__(self, in_channels=3, mid_channels=4, out_channels=8):
        super().__init__()
        self.layers = nn.Sequential(
            CogVideoXDownsample3D(in_channels, mid_channels, 3, 2, compress_time=True),
            nn.SiLU(),
            CogVideoXDownsample3D(mid_channels, mid_channels, 3, 2, compress_time=True),
            nn.SiLU(),
            CogVideoXDownsample3D(mid_channels, out_channels, 3, 2),
        )

    def forward(self, x):
        return self.layers(x)


class GuiderNet3DCogVideoX2(nn.Module):
    def __init__(self, in_channels=16, mid_channels=16, out_channels=16):
        super().__init__()
        self.layers = nn.Sequential(
            CogVideoXCausalConv3d(in_channels, mid_channels, 3, 1),
            nn.SiLU(),
            CogVideoXCausalConv3d(mid_channels, mid_channels, 3, 1),
            nn.SiLU(),
            CogVideoXCausalConv3d(mid_channels, mid_channels, 3, 1),
            nn.SiLU(),
            CogVideoXCausalConv3d(mid_channels, mid_channels, 3, 1),
            nn.SiLU(),
            CogVideoXCausalConv3d(mid_channels, out_channels, 3, 1),
        )

    def forward(self, x):
        return self.layers(x)


class CogVideoXCausalConv3d(_CogVideoXCausalConv3d):
    def fake_context_parallel_forward(self, inputs):
        kernel_size = self.time_kernel_size
        if kernel_size > 1:
            cached_inputs = [inputs[:, :, :1]] * (kernel_size - 1)
            inputs = torch.cat(cached_inputs + [inputs], dim=2)
        return inputs


def pad_at_dim(t, pad, dim=-1):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), mode="constant")


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


import os, sys
def init_paths(project_name=None):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    python_paths = [
        "../../../../",
    ]
    if project_name is not None:
        python_paths.append(os.path.join(cur_dir, project_name))
    for python_path in python_paths:
        sys.path.insert(0, python_path)
        if "PYTHONPATH" in os.environ:
            os.environ["PYTHONPATH"] += ":{}".format(python_path)
        else:
            os.environ["PYTHONPATH"] = python_path
init_paths()
## dependencies
from typing import Any, Dict, Optional, Tuple, Union, List

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNorm
from diffusers.utils import is_torch_version, logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.model_loading_utils import load_state_dict
from einops import rearrange

from vast.utils.acceleration import (
    all_to_all,
    broadcast,
    gather_forward_split_backward,
    get_sequence_parallel_group,
    split_forward_gather_backward,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def _all_in_all_with_text(hidden_states, text_seq_length, sp_group, sp_size, mode):
    if mode == 1:
        hidden_states = all_to_all(hidden_states, scatter_dim=1, gather_dim=2, group=sp_group)
        hidden_states = rearrange(hidden_states, 'b h (p s) c -> b h p s c', p=sp_size)
        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(3) - text_seq_length], dim=3
        )
        encoder_hidden_states = rearrange(encoder_hidden_states, 'b h p s c -> b h (p s) c')
        hidden_states = rearrange(hidden_states, 'b h p s c -> b h (p s) c')
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=2)
    elif mode == 2:
        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(2) - text_seq_length], dim=2
        )
        encoder_hidden_states = rearrange(encoder_hidden_states, 'b h (p s) c -> b h p s c', p=sp_size)
        hidden_states = rearrange(hidden_states, 'b h (p s) c -> b h p s c', p=sp_size)
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=3)
        hidden_states = rearrange(hidden_states, 'b h p s c -> b h (p s) c')
        hidden_states = all_to_all(hidden_states, scatter_dim=2, gather_dim=1, group=sp_group)
    else:
        assert False
    return hidden_states

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

## VASTAttnProcessor2_0
class VASTAttnProcessor2_0:

    def __init__(self):
        if not hasattr(F, 'scaled_dot_product_attention'):
            raise ImportError('VASTAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.')

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        sp_group = get_sequence_parallel_group()
        if sp_group is not None:
            sp_size = dist.get_world_size(sp_group)
            query = _all_in_all_with_text(query, text_seq_length, sp_group, sp_size, mode=1)
            key = _all_in_all_with_text(key, text_seq_length, sp_group, sp_size, mode=1)
            value = _all_in_all_with_text(value, text_seq_length, sp_group, sp_size, mode=1)
            text_seq_length *= sp_size

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        if sp_group is not None:
            hidden_states = _all_in_all_with_text(hidden_states, text_seq_length, sp_group, sp_size, mode=2)
            text_seq_length = text_seq_length // sp_size

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states

## VASTPosEmbed
class VASTRotaryPosEmbed(nn.Module):
    def __init__(self, rope_dim: List[int], theta: float = None) -> None:
        super().__init__()

        self.rope_dim = rope_dim
        if theta is not None:
            self.theta = theta
        else:
            self.theta = 10000.0

    def forward(self, hidden_states: torch.Tensor, patch_size: Tuple = (1,1,1)) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = hidden_states.shape

        rope_sizes = [num_frames // patch_size[0], height // patch_size[1], width // patch_size[2]]

        grid_t = torch.arange(rope_sizes[0], device=hidden_states.device, dtype=torch.float32)
        grid_h = torch.arange(rope_sizes[1], device=hidden_states.device, dtype=torch.float32)
        grid_w = torch.arange(rope_sizes[2], device=hidden_states.device, dtype=torch.float32)

        freqs_t = get_1d_rotary_pos_embed(self.rope_dim[0], grid_t, theta=self.theta, use_real=True)
        # Spatial frequencies for height and width
        freqs_h = get_1d_rotary_pos_embed(self.rope_dim[1], grid_h, theta=self.theta, use_real=True)
        freqs_w = get_1d_rotary_pos_embed(self.rope_dim[2], grid_w, theta=self.theta, use_real=True)

        # BroadCast and concatenate temporal and spaial frequencie (height and width) into a 3d tensor
        def combine_time_height_width(freqs_t, freqs_h, freqs_w):
            freqs_t = freqs_t[:, None, None, :].expand(
                -1, rope_sizes[1], rope_sizes[2], -1
            )  # rope_sizes[0], rope_sizes[1], rope_sizes[2], dim_t
            freqs_h = freqs_h[None, :, None, :].expand(
                rope_sizes[0], -1, rope_sizes[2], -1
            )  # rope_sizes[0], rope_sizes[1], rope_sizes[2], dim_h
            freqs_w = freqs_w[None, None, :, :].expand(
                rope_sizes[0], rope_sizes[1], -1, -1
            )  # rope_sizes[0], rope_sizes[1], rope_sizes[2], dim_w

            freqs = torch.cat(
                [freqs_t, freqs_h, freqs_w], dim=-1
            )  # rope_sizes[0], rope_sizes[1], rope_sizes[2], (dim_t + dim_h + dim_w)
            freqs = freqs.view(
                rope_sizes[0] * rope_sizes[1] * rope_sizes[2], -1
            )  # (rope_sizes[0] * rope_sizes[1] * rope_sizes[2]), (dim_t + dim_h + dim_w)
            return freqs

        t_cos, t_sin = freqs_t  # both t_cos and t_sin has shape: temporal_size, dim_t
        h_cos, h_sin = freqs_h  # both h_cos and h_sin has shape: grid_size_h, dim_h
        w_cos, w_sin = freqs_w  # both w_cos and w_sin has shape: grid_size_w, dim_w

        cos = combine_time_height_width(t_cos, h_cos, w_cos)
        sin = combine_time_height_width(t_sin, h_sin, w_sin)
        return cos, sin

## VASTPatchEmbed
class VASTPatchEmbed(nn.Module):
    def __init__(
        self, 
        patch_size: List[Tuple] = [(1,2,2),(1,4,4)],          #patch_size=[(p_t,p_h,p_w),..]
        in_channels: int = 16,
        num_attention_heads: int = 48,
        attention_head_dim: int = 64,
        text_embed_dim: int = 4096,
        bias: bool = True,
        theta: float = None
    ) -> None:
        super().__init__()

        embed_dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size
        patch_layers = []
        p_t,p_h,p_w = self.patch_size[0]
        # for p_t,p_h,p_w in self.patch_size:
        #     patch_layers.append(nn.Conv3d(in_channels, embed_dim, kernel_size=(p_t,p_h,p_w), stride=(p_t,p_h,p_w), bias=bias))
        for i in range(len(self.patch_size)):
            patch_layers.append(nn.Conv3d(in_channels, embed_dim, kernel_size=(p_t,p_h,p_w), stride=(p_t,p_h,p_w), bias=bias))
        self.proj = nn.ModuleList(patch_layers)
        
        self.text_proj = nn.Linear(text_embed_dim, embed_dim)
        self.rope = VASTRotaryPosEmbed([attention_head_dim//4, attention_head_dim//8*3, attention_head_dim//8*3], theta=theta)

    def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor):
        r"""
        Args:
            text_embeds (`torch.Tensor`):
                Input text embeddings. Expected shape: (batch_size, seq_length, embedding_dim).
            image_embeds (`torch.Tensor`):
                Input image embeddings. Expected shape: (batch_size, num_frames, channels, height, width).
        """
        text_embeds = self.text_proj(text_embeds)

        batch_size, num_frames, channels, height, width = image_embeds.shape

        embeds = []
        ropes = []
        for i, (p_t, p_h, p_w) in enumerate(self.patch_size):
            image_embed_p = image_embeds.permute(0, 2, 1, 3, 4)
            if i==0:
                image_embed_p = self.proj[i](image_embed_p)
            elif i==1:
                image_embed_p = F.interpolate(image_embed_p, scale_factor=(self.patch_size[0][0]/(1.0*p_t), self.patch_size[0][1]/(1.0*p_h), self.patch_size[0][2]/(1.0*p_w)), mode='trilinear')
                image_embed_p = self.proj[i](image_embed_p)
            else:
                assert False
            image_embed_p = image_embed_p.flatten(2).transpose(1, 2)  # BCFHW -> BNC
            embeds.append(
                torch.cat([text_embeds, image_embed_p], dim=1).contiguous()
            )  # [batch, seq_length + num_frames x height x width, channels])
            ropes.append(self.rope(image_embeds, (p_t, p_h, p_w)))

        assert len(ropes) == len(embeds)

        return embeds, ropes

## VASTLayerNormZeroDS DoubleStream
class VASTLayerNormZeroDS(nn.Module):
    def __init__(
        self,
        conditioning_dim: int,
        embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_dim, 6 * embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hs_shift, hs_scale, hs_gate, enc_shift, enc_scale, enc_gate = self.linear(self.silu(temb)).chunk(6, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + hs_scale)[:, None, :] + hs_shift[:, None, :]
        encoder_hidden_states = self.norm(encoder_hidden_states) * (1 + enc_scale)[:, None, :] + enc_shift[:, None, :]
        return hidden_states, encoder_hidden_states, hs_gate[:, None, :], enc_gate[:, None, :]

## VASTLayerNormZeroSS SingleStream
class VASTLayerNormZeroSS(nn.Module):
    def __init__(
        self,
        conditioning_dim: int,
        embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_dim, 3 * embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_seq_length = encoder_hidden_states.shape[1]
        shift, scale, gate = self.linear(self.silu(temb)).chunk(3, dim=1)
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        norm_encoder_hidden_states, norm_hidden_states = (
            hidden_states[:, :text_seq_length, :],
            hidden_states[:, text_seq_length:, :],
        )
        return norm_hidden_states, norm_encoder_hidden_states, gate[:, None, :]

## VASTBlock DoubleStream
@maybe_allow_in_graph
class VASTBlockDS(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = 'gelu-approximate',
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        proj_out: bool = False,
    ):
        super().__init__()

        self.proj_out = proj_out
        # 1. Self Attention
        self.norm1 = VASTLayerNormZeroDS(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm='layer_norm' if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=VASTAttnProcessor2_0(),
        )

        # 2. Feed Forward
        self.norm2 = VASTLayerNormZeroDS(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        # 3. zero initilize project out
        if self.proj_out:
            self.prj_o = zero_module(nn.Linear(dim, dim, bias=True))

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, hs_gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        # attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + hs_gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, hs_gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + hs_gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        # project out
        if self.proj_out:
            prj_output = self.prj_o(torch.cat([encoder_hidden_states, hidden_states], dim=1))
            hidden_states = prj_output[:, text_seq_length:]
            encoder_hidden_states = prj_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states

## VASTBlock SingleStream
@maybe_allow_in_graph
class VASTBlockSS(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = 'gelu-approximate',
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        proj_out: bool = False,
    ):
        super().__init__()

        self.proj_out = proj_out
        # 1. Self Attention
        self.norm1 = VASTLayerNormZeroSS(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm='layer_norm' if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=VASTAttnProcessor2_0(),
        )

        # 2. Feed Forward
        self.norm2 = VASTLayerNormZeroSS(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        # 3. zero initilize project out
        if self.proj_out:
            self.prj_o = zero_module(nn.Linear(dim, dim, bias=True))

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        # attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + gate_msa * attn_encoder_hidden_states

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + gate_ff * ff_output[:, :text_seq_length]

        # project out
        if self.proj_out:
            prj_output = self.prj_o(torch.cat([encoder_hidden_states, hidden_states], dim=1))
            hidden_states = prj_output[:, text_seq_length:]
            encoder_hidden_states = prj_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states

## VASTTransformerModel
@maybe_allow_in_graph
class VASTTransformerModel(ModelMixin, ConfigMixin):
    
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 48,
        attention_head_dim: int = 64,
        in_channels: int = 32,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        text_embed_dim: int = 4096,
        num_front_layers: int = 24,
        num_middle_layers: int = 18,
        num_tail_layers: int = 0,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 1280,      #32的倍数
        sample_height: int = 736,      #32的倍数
        sample_frames: int = 97,       #8的倍数+1
        patch_size: List[Tuple] = [(1,2,2),(1,4,4)],          #patch_size=[(p_t,p_h,p_w),..]
        spatial_compression_ratio: int = 8,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = 'gelu-approximate',
        timestep_activation_fn: str = 'silu',
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        only_orgin: bool = False,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        self.only_orgin = only_orgin

        # 1. Patch embedding
        self.patch_embed = VASTPatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            text_embed_dim=text_embed_dim,
            bias=True,
            theta=10000.0,
        )
        self.embedding_dropout = nn.Dropout(dropout)

        # 2. Time embeddings
        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)

        # 3.1 Define spatio-temporal transformers blocks - num_front_layers
        self.transformer_blocks_front = nn.ModuleList(
            [
                VASTBlockDS(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_front_layers)
            ]
        )
        # 3.2 Define spatio-temporal transformers blocks - num_middle_layers
        ## use DoubleStream process low res latents
        self.transformer_blocks_midds = nn.ModuleList(
            [
                VASTBlockDS(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_middle_layers)
            ]
        )
        ## use SingleStream enhance text and visual latents fusion
        self.transformer_blocks_midss = nn.ModuleList(
            [
                VASTBlockSS(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    proj_out=True,
                )
                for _ in range(num_middle_layers)
            ]
        )
        ## use SingleStream process high res latents
        self.transformer_blocks_midms = nn.ModuleList(
            [
                VASTBlockSS(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    proj_out=True,
                )
                for _ in range(num_middle_layers)
            ]
        )
        ## interpolate
        up_scale = (patch_size[-1][0]*patch_size[-1][1]*patch_size[-1][2])//(patch_size[0][0]*patch_size[0][1]*patch_size[0][2]) 
        self.up_scale = nn.Linear(inner_dim//up_scale, inner_dim)

        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)
        self.norm_final_lr = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)

        # 4. Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )
        self.norm_out_lr = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )
        self.proj_out = nn.Linear(inner_dim, patch_size[0][0] * patch_size[0][1] * patch_size[0][2] * out_channels)
        self.proj_out_lr = nn.Linear(inner_dim, patch_size[0][0] * patch_size[0][1] * patch_size[0][2] * out_channels)
        # self.proj_out_lr = nn.Linear(inner_dim, patch_size[-1][0] * patch_size[-1][1] * patch_size[-1][2] * out_channels)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        if kwargs.pop("from_giga", False):
            from vast.models.dit import CogVideoXTransformer3DModel
            patch_size = kwargs.pop("patch_size", None)
            only_orgin = kwargs.pop("only_orgin", False)
            giga_model = CogVideoXTransformer3DModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
            giga_model_state_dict = giga_model.state_dict()
            if patch_size is not None:
                model = VASTTransformerModel(patch_size=patch_size, only_orgin=only_orgin)
            else:
                model = VASTTransformerModel(only_orgin=only_orgin)
            model_static = model.state_dict()
            giga_model_state_dict_new = {}
            for key in model_static.keys():
                if "patch_embed.proj" in key:
                    if "patch_embed.proj.0" in key:
                        old_key = key.replace("patch_embed.proj.0", "patch_embed.proj")
                    elif "patch_embed.proj.1" in key:
                        old_key = key.replace("patch_embed.proj.1", "patch_embed.proj")
                    else:
                        assert False
                    if old_key == "patch_embed.proj.bias":
                        if old_key in giga_model_state_dict:
                            if model_static[key].shape == giga_model_state_dict[old_key].shape:
                                giga_model_state_dict_new[key] = giga_model_state_dict[old_key]
                    elif old_key == "patch_embed.proj.weight":
                        if old_key in giga_model_state_dict:
                            # import pdb; pdb.set_trace()
                            if model_static[key].shape == giga_model_state_dict[old_key].unsqueeze(2).shape:
                                giga_model_state_dict_new[key] = giga_model_state_dict[old_key].unsqueeze(2)
                            else:
                                # giga_model_state_dict_new[key] = model_static[key]
                                # giga_model_state_dict_new[key].zero_()
                                # giga_model_state_dict_new[key][:,:,:1,:2,:2] = giga_model_state_dict[old_key].unsqueeze(2)
                                # giga_model_state_dict_new[key] = giga_model_state_dict[old_key].unsqueeze(2).repeat(1, 1, 1, 2, 2)/2.0
                                print("load partial parameters: " + key)
                                # print(giga_model_state_dict_new[key].shape)
                                print(model_static[key].shape)
                                # import pdb; pdb.set_trace()
                elif "proj_out" in key:
                    if "proj_out_lr" in key:
                        old_key = key.replace("proj_out_lr", "proj_out")
                    elif "proj_out" in key:
                        old_key = key
                    else:
                        assert False
                    if old_key == "proj_out.bias":
                        if old_key in giga_model_state_dict:
                            if model_static[key].shape == giga_model_state_dict[old_key].shape:
                                giga_model_state_dict_new[key] = giga_model_state_dict[old_key]
                            else:
                                # giga_model_state_dict_new[key] = model_static[key]
                                # giga_model_state_dict_new[key].zero_()
                                # giga_model_state_dict_new[key][:64] = giga_model_state_dict[old_key]
                                print("load partial parameters: " + key)
                                # print(giga_model_state_dict_new[key].shape)
                                print(model_static[key].shape)
                    elif old_key == "proj_out.weight":
                        if old_key in giga_model_state_dict:
                            # import pdb; pdb.set_trace()
                            if model_static[key].shape == giga_model_state_dict[old_key].shape:
                                giga_model_state_dict_new[key] = giga_model_state_dict[old_key]
                            else:
                                # giga_model_state_dict_new[key] = model_static[key]
                                # giga_model_state_dict_new[key].zero_()
                                # giga_model_state_dict_new[key][:64,:] = giga_model_state_dict[old_key]
                                print("load partial parameters: " + key)
                                # print(giga_model_state_dict_new[key].shape)
                                print(model_static[key].shape)
                                # import pdb; pdb.set_trace()
                elif "norm_final" in key:
                    if "norm_final_lr" in key:
                        old_key = key.replace("norm_final_lr", "norm_final")
                    elif "norm_final" in key:
                        old_key = key
                    else:
                        assert False
                    if model_static[key].shape == giga_model_state_dict[old_key].shape:
                        giga_model_state_dict_new[key] = giga_model_state_dict[old_key]
                elif "norm_out" in key:
                    if "norm_out_lr" in key:
                        old_key = key.replace("norm_out_lr", "norm_out")
                    elif "norm_out" in key:
                        old_key = key
                    else:
                        assert False
                    if model_static[key].shape == giga_model_state_dict[old_key].shape:
                        giga_model_state_dict_new[key] = giga_model_state_dict[old_key]
                elif "transformer_blocks" in key:
                    for i in range(24):
                        if "transformer_blocks_front."+str(i) in key:
                            old_key = key.replace("transformer_blocks_front."+str(i), "transformer_blocks."+str(i))
                            if old_key in giga_model_state_dict:
                                if model_static[key].shape == giga_model_state_dict[old_key].shape:
                                    giga_model_state_dict_new[key] = giga_model_state_dict[old_key]
                        elif "transformer_blocks_midds."+str(i) in key:
                            old_key = key.replace("transformer_blocks_midds."+str(i), "transformer_blocks."+str(i+24))
                            if old_key in giga_model_state_dict:
                                if model_static[key].shape == giga_model_state_dict[old_key].shape:
                                    giga_model_state_dict_new[key] = giga_model_state_dict[old_key]
                        elif "transformer_blocks_midss."+str(i) in key:
                            old_key = key.replace("transformer_blocks_midss."+str(i), "transformer_blocks."+str(i+24))
                            if old_key in giga_model_state_dict:
                                if model_static[key].shape == giga_model_state_dict[old_key].shape:
                                    giga_model_state_dict_new[key] = giga_model_state_dict[old_key]
                        elif "transformer_blocks_midms."+str(i) in key:  
                            old_key = key.replace("transformer_blocks_midms."+str(i), "transformer_blocks."+str(i+24))
                            if old_key in giga_model_state_dict:
                                if model_static[key].shape == giga_model_state_dict[old_key].shape:
                                    giga_model_state_dict_new[key] = giga_model_state_dict[old_key]
                        else:
                            old_key = key
                            if old_key in giga_model_state_dict:
                                if model_static[key].shape == giga_model_state_dict[old_key].shape:
                                    giga_model_state_dict_new[key] = giga_model_state_dict[old_key]
                else:
                    old_key = key
                    if old_key in giga_model_state_dict:
                        if model_static[key].shape == giga_model_state_dict[old_key].shape:
                            giga_model_state_dict_new[key] = giga_model_state_dict[old_key]
            check = False
            if check:
                with open("giga_model_state_dict_new.txt", "w") as f:
                    for key in giga_model_state_dict_new.keys():
                        f.write(key + "\n")
                with open("giga_model_state_dict.txt", "w") as f:
                    for key in giga_model_state_dict.keys():
                        f.write(key + "\n")
                with open("model_static.txt", "w") as f:
                    for key in model_static.keys():
                        f.write(key + "\n")
                # import pdb; pdb.set_trace()
            if isinstance(kwargs.get("torch_dtype", None), torch.dtype):
                missing_keys, unexpected_keys = model.load_state_dict(giga_model_state_dict_new, strict=False)
                model.to(kwargs.get("torch_dtype", None))
            else:
                missing_keys, unexpected_key = model.load_state_dict(giga_model_state_dict_new, strict=False)
            print("Missing Keys:", len(missing_keys))
            print("Unexpected Keys:", len(unexpected_keys))
            return model
        else:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        need_broadcast: bool = True,
        return_dict: bool = True,
    ):
        sp_group = get_sequence_parallel_group()
        if sp_group is not None:
            if need_broadcast:
                hidden_states = broadcast(hidden_states, group=sp_group)
                encoder_hidden_states = broadcast(encoder_hidden_states, group=sp_group)
                timestep = broadcast(timestep, group=sp_group)
                if timestep_cond is not None:
                    timestep_cond = broadcast(timestep_cond, group=sp_group)
            hidden_states = split_forward_gather_backward(hidden_states, dim=1, group=sp_group)
            encoder_hidden_states = split_forward_gather_backward(encoder_hidden_states, dim=1, group=sp_group)

        batch_size, num_frames, channels, height, width = hidden_states.shape
        p_all = self.patch_embed.patch_size

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # 2. Patch embedding
        hidden_states_all, ropes_all = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states_hr = hidden_states_all[0] #for other extra branch
        hidden_states = self.embedding_dropout(hidden_states_all[-1])
        image_rotary_emb_hr = ropes_all[0]      #for other extra branch
        image_rotary_emb = ropes_all[-1]
        # image_rotary_emb[0].zero_()
        ## rope has small differnece between diffusers version20240908 and the latest

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]
        hidden_states_hr = hidden_states_hr[:, text_seq_length:]

        # 3. Transformer blocks
        # process front blocks
        for i, block in enumerate(self.transformer_blocks_front):
            # import pdb; pdb.set_trace()
            if self.training and self.gradient_checkpointing:
                # import pdb; pdb.set_trace()
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {'use_reentrant': False} if is_torch_version('>=', '1.11.0') else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )
        # process middle blocks
        hidden_states_ds = hidden_states.clone()
        encoder_hidden_states_ds = encoder_hidden_states.clone()
        hidden_states_ss = hidden_states.clone()
        encoder_hidden_states_ss = encoder_hidden_states.clone()
        hidden_states_ms = hidden_states.reshape(batch_size, num_frames//p_all[-1][0], height//p_all[-1][1], width//p_all[-1][2], -1, p_all[-1][0]//p_all[0][0], p_all[-1][1]//p_all[0][1], p_all[-1][2]//p_all[0][2])
        hidden_states_ms = hidden_states_ms.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(2).transpose(1,2)
        hidden_states_ms = hidden_states_hr + self.up_scale(hidden_states_ms)
        encoder_hidden_states_ms = encoder_hidden_states.clone()
        for i, (block_midds, block_midss, block_midms) in enumerate(zip(self.transformer_blocks_midds, 
                                                                        self.transformer_blocks_midss, 
                                                                        self.transformer_blocks_midms)):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {'use_reentrant': False} if is_torch_version('>=', '1.11.0') else {}
                hidden_states_ds, encoder_hidden_states_ds = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block_midds),
                    hidden_states_ds,
                    encoder_hidden_states_ds,
                    emb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
                if not self.only_orgin:
                    hidden_states_ss, encoder_hidden_states_ss = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block_midss),
                        hidden_states_ss,
                        encoder_hidden_states_ss,
                        emb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                    hidden_states_ms, encoder_hidden_states_ms = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block_midms),
                        hidden_states_ms,
                        encoder_hidden_states_ms,
                        emb,
                        image_rotary_emb_hr,
                        **ckpt_kwargs,
                )
            else:
                # import torch.profiler as profiler
                # with profiler.profile(
                #     activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
                #     on_trace_ready=profiler.tensorboard_trace_handler('./log'),
                #     record_shapes=True,
                #     with_stack=True
                # ) as prof:
                # 运行你的代码
                hidden_states_ds, encoder_hidden_states_ds = block_midds(
                    hidden_states=hidden_states_ds,
                    encoder_hidden_states=encoder_hidden_states_ds,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )
                if not self.only_orgin:
                    hidden_states_ss, encoder_hidden_states_ss = block_midss(
                        hidden_states=hidden_states_ss,
                        encoder_hidden_states=encoder_hidden_states_ss,
                        temb=emb,
                        image_rotary_emb=image_rotary_emb,
                    )
                    hidden_states_ms, encoder_hidden_states_ms = block_midms(
                        hidden_states=hidden_states_ms,
                        encoder_hidden_states=encoder_hidden_states_ms,
                        temb=emb,
                        image_rotary_emb=image_rotary_emb_hr,
                    )
            if not self.only_orgin:
                hidden_states_lr = hidden_states_ds + hidden_states_ss
                hidden_states_lr = hidden_states_lr.reshape(batch_size, num_frames//p_all[-1][0], height//p_all[-1][1], width//p_all[-1][2], -1, p_all[-1][0]//p_all[0][0], p_all[-1][1]//p_all[0][1], p_all[-1][2]//p_all[0][2])
                hidden_states_lr = hidden_states_lr.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(2).transpose(1,2)
                hidden_states_ms = hidden_states_ms + self.up_scale(hidden_states_lr)
                encoder_hidden_states_ms = encoder_hidden_states_ms + encoder_hidden_states_ds + encoder_hidden_states_ss
                # import pdb; pdb.set_trace()
        # process tail blocks

        hidden_states = torch.cat([encoder_hidden_states_ms, hidden_states_ms], dim=1)
        hidden_states = self.norm_final(hidden_states)
        hidden_states = hidden_states[:, text_seq_length:]
        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        # import pdb; pdb.set_trace()
        hidden_states = self.proj_out(hidden_states)
        # 5. Unpatchify
        output = hidden_states.reshape(batch_size, num_frames//p_all[0][0], height//p_all[0][1], width//p_all[0][2], -1, p_all[0][0], p_all[0][1], p_all[0][2])
        output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        hidden_states_lr = torch.cat([encoder_hidden_states_ds, hidden_states_ds], dim=1)
        hidden_states_lr = self.norm_final_lr(hidden_states_lr)
        hidden_states_lr = hidden_states_lr[:, text_seq_length:]
        # 4. Final block
        hidden_states_lr = self.norm_out_lr(hidden_states_lr, temb=emb)
        # import pdb; pdb.set_trace()

        # debug
        # hidden_states_lr = self.proj_out_lr(hidden_states_lr)
        # # 5. Unpatchify
        # output_lr = hidden_states_lr.reshape(batch_size, num_frames//p_all[-1][0], height//p_all[-1][1], width//p_all[-1][2], -1, p_all[-1][0], p_all[-1][1], p_all[-1][2])
        # output_lr = output_lr.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        # import pdb; pdb.set_trace()
        hidden_states_lr = self.proj_out_lr(hidden_states_lr) 
        output_lr = hidden_states_lr.reshape(batch_size, num_frames//p_all[-1][0], height//p_all[-1][1], width//p_all[-1][2], -1, p_all[0][0], p_all[0][1], p_all[0][2])
        output_lr = output_lr.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        output_lr = F.interpolate(output_lr, scale_factor=(p_all[-1][0]/(1.0*p_all[0][0]), p_all[-1][1]/(1.0*p_all[0][1]), p_all[-1][2]/(1.0*p_all[0][2])), mode='trilinear')
        output_lr = output_lr.permute(0, 2, 1, 3, 4)

        if sp_group is not None:
            output = gather_forward_split_backward(output, dim=1, group=sp_group)
            output_lr = gather_forward_split_backward(output_lr, dim=1, group=sp_group)

        # import pdb; pdb.set_trace()
        if not return_dict:
            if self.only_orgin:
                return (output_lr,)
            else:
                return (output,output_lr)
        return Transformer2DModelOutput(sample=output)


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    device = torch.device("cuda")
    if False:
        transformer = VASTTransformerModel().to(device, dtype=torch.bfloat16) 
        print(type(transformer)) 
        num_params = sum(p.numel() for p in transformer.parameters())
        total_bilion = num_params / (1024 * 1024 * 1024)
        print(f"Total number of parameters: {num_params}")
        print(f"Total model size: {total_bilion:.2f} B")
        text_tensor = torch.randn(2, 226, 4096).to(device, dtype=torch.bfloat16) 
        img_tensor = torch.randn(2, 26, 32, 92, 160).to(device, dtype=torch.bfloat16)       
        timestep = torch.tensor([0, 999]).to(device, dtype=torch.bfloat16) 
        # 统计 FLOPs 和参数
        # with torch.no_grad():
        #     flops = FlopCountAnalysis(transformer, (img_tensor, text_tensor, timestep))
        #     print(f"FLOPs: {flops.total()/1e9}")
        #     # print(parameter_count_table(transformer))
        with torch.no_grad():
            noise = transformer(img_tensor, text_tensor, timestep, return_dict=False)[0]
        import pdb; pdb.set_trace()

    if False:
        text_tensor = torch.randn(2, 226, 4096).to(device) 
        img_tensor = torch.randn(2, 26, 16, 92, 160).to(device) 
        patch_embeder = VASTPatchEmbed().to(device) 
        embeds, ropes = patch_embeder(text_tensor, img_tensor)
        import pdb; pdb.set_trace()

    if True:
        transformer = VASTTransformerModel.from_pretrained(
            "/root/code/vast/pretrained/models--THUDM--CogVideoX-5b/transformer", 
            torch_dtype=torch.bfloat16,
            from_giga=True,
            # patch_size=[(1,2,2)],
            ).to(device)
        # with torch.n[o_grad():
        #     img_tensor = torch.load('img_tensor.pth')
        #     text_tensor = torch.load('text_tensor.pth')
        #     timestep = torch.load('timestep.pth')
        #     noise = transformer(img_tensor, text_tensor, timestep, return_dict=False)[0]
        #     noise_cogvx = torch.load('noise_cogvx.pth')
        #     max_error = torch.max(torch.abs(noise_cogvx - noise))
        #     print("Ma]x Error:", max_error)
        import pdb; pdb.set_trace()

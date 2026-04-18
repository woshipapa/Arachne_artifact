
from typing import Any, Dict, Optional, Tuple, Union
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.transformer_config import TransformerConfig


import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F



from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import (
    AttentionProcessor,
    FusedCogVideoXAttnProcessor2_0,
)
from diffusers.models.embeddings import (
    TimestepEmbedding,
    Timesteps,
    get_3d_sincos_pos_embed,
)
import os
from megatron.core import mpu, tensor_parallel
from diffusers.models.normalization import AdaLayerNorm, CogVideoXLayerNormZero
from diffusers.utils import is_torch_version, logging
from diffusers.utils.torch_utils import maybe_allow_in_graph

from megatron.core import mpu
from megatron.core.context_parallel import set_origin_length, set_target_length, pad_for_context_parallel, remove_pad_for_context_parallel
from diffusers.models.modeling_outputs import Transformer2DModelOutput

class CogVideoXParams:
    """
    Parameters for initializing the CogVideoXTransformer3DModel.
    """
    # Attention block parameters
    num_attention_heads: int = 30
    attention_head_dim: int = 64
    attention_bias: bool = True

    # Model architecture parameters
    num_layers: int = 30
    in_channels: int = 16
    out_channels: Optional[int] = 16
    patch_size: int = 2
    dropout: float = 0.0

    # Timestep and text embedding parameters
    time_embed_dim: int = 512
    text_embed_dim: int = 4096
    flip_sin_to_cos: bool = True
    freq_shift: int = 0

    # Input data dimension parameters
    sample_width: int = 90
    sample_height: int = 60
    sample_frames: int = 49 # Note: See original docstring for details on this default value
    temporal_compression_ratio: int = 4
    max_text_seq_length: int = 226

    # Activation functions
    activation_fn: str = "gelu-approximate"
    timestep_activation_fn: str = "silu"

    # Normalization parameters
    norm_elementwise_affine: bool = True
    norm_eps: float = 1e-5

    # Positional embedding parameters
    spatial_interpolation_scale: float = 1.875
    temporal_interpolation_scale: float = 1.0
    use_rotary_positional_embeddings: bool = False


def _get_activation_fn(name: str):
    """Helper to convert string activation name to function object."""
    if name == "gelu-approximate":
        return lambda x: F.gelu(x, approximate="tanh")
    elif name == "gelu":
        return F.gelu
    elif name == "silu":
        return F.silu
    # Add other activations if needed
    raise ValueError(f"Unsupported activation function: {name}")


class CogVideoXAttnProcessor2_0:
    r"""Processor for implementing scaled dot-product attention for the
    CogVideoX model.

    It applies a rotary embedding on query and key vectors, but does not include spatial normalization.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

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
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

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

        images_query = query[:, :, text_seq_length:]
        images_key = key[:, :, text_seq_length:]
        images_value = value[:, :, text_seq_length:]

        txt_query = query[:, :, :text_seq_length]
        txt_key = key[:, :, :text_seq_length]
        txt_value = value[:, :, :text_seq_length]


        if mpu.get_context_parallel_world_size() > 1:

            from yunchang.comm.all_to_all import SeqAllToAll4D
            from megatron.core.tensor_parallel.mappings import (
                split_forward_gather_backward,
                gather_forward_split_backward,
            )
            print(f"images_q shape is {images_query.shape}, txt_query shape is {txt_query.shape}")
            images_query = SeqAllToAll4D.apply(mpu.get_context_parallel_group(), images_query,  1, 2)
            images_key = SeqAllToAll4D.apply(mpu.get_context_parallel_group(),images_key, 1, 2)
            images_value = SeqAllToAll4D.apply(mpu.get_context_parallel_group(),images_value, 1, 2)

            # 注意力头维度切分 ，序列维度完整txt
            txt_query = split_forward_gather_backward(txt_query,mpu.get_context_parallel_group(), dim=1)
            txt_key = split_forward_gather_backward(txt_key,mpu.get_context_parallel_group(), dim=1)
            txt_value= split_forward_gather_backward(txt_value,mpu.get_context_parallel_group(), dim=1)


            # 现在的images qkv都是在seq_length都是带有padding的
            # 这里需要remove padding

            images_query, images_key, images_value = map(
                lambda x: remove_pad_for_context_parallel(x, dim=2),
                (images_query, images_key, images_value),
            )


        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            images_query = apply_rotary_emb(
                images_query, image_rotary_emb
            )
            if not attn.is_cross_attention:
                images_key = apply_rotary_emb(
                    images_key, image_rotary_emb
                )
         # bhsd 
        query = torch.cat([txt_query, images_query], dim=2).contiguous() 
        key = torch.cat([txt_key, images_key], dim=2).contiguous()
        value = torch.cat([txt_value, images_value], dim=2).contiguous()

        # if mpu.get_context_parallel_world_size() > 1:
        #     query, key ,value = map(
        #         lambda x: remove_pad_with_encoder_for_context_parallel(x, text_seq_length, dim=0),
        #         [query, key, value]
        #     )

        print(f"[Rank {torch.distributed.get_rank()}]  query shape is {query.shape}")
        core_attn_out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        core_attn_out = core_attn_out.to(query.dtype)
        hidden_states = core_attn_out[:,:, text_seq_length:, :]
        encoder_hidden_states = core_attn_out[:,:, :text_seq_length, :]

        # b s, head, dim
        hidden_states = hidden_states.transpose(1,2)
        encoder_hidden_states = encoder_hidden_states.transpose(1,2)
        if mpu.get_context_parallel_world_size() > 1:
            hidden_states = pad_for_context_parallel(hidden_states, dim=1)
            hidden_states = SeqAllToAll4D.apply(mpu.get_context_parallel_group(),hidden_states, 1, 2)
            encoder_hidden_states = gather_forward_split_backward(encoder_hidden_states, mpu.get_context_parallel_group(), 2) # b txt_seq n d
        hidden_states = hidden_states.reshape(
            batch_size, -1, attn.heads * head_dim
        )
        encoder_hidden_states = encoder_hidden_states.reshape(
            batch_size, -1, attn.heads * head_dim
        )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


class CogVideoXPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 2,
        patch_size_t: Optional[int] = None,
        in_channels: int = 16,
        embed_dim: int = 1920,
        text_embed_dim: int = 4096,
        bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_positional_embeddings: bool = True,
        use_learned_positional_embeddings: bool = True,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.embed_dim = embed_dim
        self.sample_height = sample_height
        self.sample_width = sample_width
        self.sample_frames = sample_frames
        self.temporal_compression_ratio = temporal_compression_ratio
        self.max_text_seq_length = max_text_seq_length
        self.spatial_interpolation_scale = spatial_interpolation_scale
        self.temporal_interpolation_scale = temporal_interpolation_scale
        self.use_positional_embeddings = use_positional_embeddings
        self.use_learned_positional_embeddings = use_learned_positional_embeddings

        if patch_size_t is None:
            # CogVideoX 1.0 checkpoints
            self.proj = nn.Conv2d(
                in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
            )
        else:
            # CogVideoX 1.5 checkpoints
            self.proj = nn.Linear(in_channels * patch_size * patch_size * patch_size_t, embed_dim)

        self.text_proj = nn.Linear(text_embed_dim, embed_dim)

        if use_positional_embeddings or use_learned_positional_embeddings:
            persistent = use_learned_positional_embeddings
            pos_embedding = self._get_positional_embeddings(sample_height, sample_width, sample_frames)
            self.register_buffer("pos_embedding", pos_embedding, persistent=persistent)

    def _get_positional_embeddings(
        self, sample_height: int, sample_width: int, sample_frames: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        post_patch_height = sample_height // self.patch_size
        post_patch_width = sample_width // self.patch_size
        post_time_compression_frames = (sample_frames - 1) // self.temporal_compression_ratio + 1
        num_patches = post_patch_height * post_patch_width * post_time_compression_frames

        pos_embedding = get_3d_sincos_pos_embed(
            self.embed_dim,
            (post_patch_width, post_patch_height),
            post_time_compression_frames,
            self.spatial_interpolation_scale,
            self.temporal_interpolation_scale,
            device=device,
            output_type="pt",
        )
        pos_embedding = pos_embedding.flatten(0, 1)
        joint_pos_embedding = pos_embedding.new_zeros(
            1, self.max_text_seq_length + num_patches, self.embed_dim, requires_grad=False
        )
        joint_pos_embedding.data[:, self.max_text_seq_length :].copy_(pos_embedding)

        return joint_pos_embedding

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

        self.sample_height = height
        self.sample_width = width
        

        if self.patch_size_t is None:
            image_embeds = image_embeds.reshape(-1, channels, height, width)
            image_embeds = self.proj(image_embeds)
            image_embeds = image_embeds.view(batch_size, num_frames, *image_embeds.shape[1:])
            image_embeds = image_embeds.flatten(3).transpose(2, 3)  # [batch, num_frames, height x width, channels]
            image_embeds = image_embeds.flatten(1, 2)  # [batch, num_frames x height x width, channels]
        else:
            p = self.patch_size
            p_t = self.patch_size_t

            image_embeds = image_embeds.permute(0, 1, 3, 4, 2)
            image_embeds = image_embeds.reshape(
                batch_size, num_frames // p_t, p_t, height // p, p, width // p, p, channels
            )
            image_embeds = image_embeds.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(4, 7).flatten(1, 3)
            image_embeds = self.proj(image_embeds)

        embeds = torch.cat(
            [text_embeds, image_embeds], dim=1
        ).contiguous()  # [batch, seq_length + num_frames x height x width, channels]
        print(f"txt_emb = {text_embeds.shape}, image_embeds = {image_embeds.shape}, self.sample_h_hw = ({self.sample_height},{self.sample_width})")
        if self.use_positional_embeddings or self.use_learned_positional_embeddings:
            if self.use_learned_positional_embeddings and (self.sample_width != width or self.sample_height != height):
                raise ValueError(
                    "It is currently not possible to generate videos at a different resolution that the defaults. This should only be the case with 'THUDM/CogVideoX-5b-I2V'."
                    "If you think this is incorrect, please open an issue at https://github.com/huggingface/diffusers/issues."
                )

            pre_time_compression_frames = (num_frames - 1) * self.temporal_compression_ratio + 1
            # self.sample_frames = pre_time_compression_frames
            if (
                self.sample_height != height
                or self.sample_width != width
                or self.sample_frames != pre_time_compression_frames
            ):
                pos_embedding = self._get_positional_embeddings(
                    height, width, pre_time_compression_frames, device=embeds.device
                )
                # print(f"pos_embedding shape is {pos_embedding.shape}")
            else:
                pos_embedding = self.pos_embedding

            pos_embedding = pos_embedding.to(dtype=embeds.dtype)
            embeds = embeds + pos_embedding

        return embeds


@maybe_allow_in_graph
class CogVideoXBlock(nn.Module):
    r"""
    Transformer block used in [CogVideoX](https://github.com/THUDM/CogVideo) model.

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        time_embed_dim (`int`):
            The number of channels in timestep embedding.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):
            Whether or not to use bias in attention projection layers.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(
            time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True
        )

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(
            time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True
        )

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        # norm & modulate

        # temb [bs, dim] 
        # hidden_states [bs, seq_padd/n, dim]
        # return gate_msa shape [bs, 1, dim]  enc_gate_msa shape [bs, 1, dim]
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = (
            self.norm1(hidden_states, encoder_hidden_states, temb)
        )

        # attention
        # return attn_hidden_states [bs, seq_padd/n, dim] 
        #  attn_encoder_hidden_states [bs, txt_seq, dim]
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = (
            encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states
        )

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = (
            self.norm2(hidden_states, encoder_hidden_states, temb)
        )

        # feed-forward
        norm_hidden_states = torch.cat(
            [norm_encoder_hidden_states, norm_hidden_states], dim=1
        )
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = (
            encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]
        )

        return hidden_states, encoder_hidden_states




class CogVideoXTransformer3DModel(VisionModule):
    """
    A Transformer model for video-like data, refactored to inherit from Megatron's
    VisionModule and strictly follow the WanModel initialization pattern.
    """

    def __init__(self, params: CogVideoXParams, config: TransformerConfig):

        # --- 1. Unpack parameters from the detailed `params` object ---
        self.num_attention_heads = params.num_attention_heads
        self.attention_head_dim = params.attention_head_dim
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        self.flip_sin_to_cos = params.flip_sin_to_cos
        self.freq_shift = params.freq_shift
        self.time_embed_dim = params.time_embed_dim
        self.text_embed_dim = params.text_embed_dim
        self.num_layers = params.num_layers
        self.dropout = params.dropout
        self.attention_bias = params.attention_bias
        self.sample_width = params.sample_width
        self.sample_height = params.sample_height
        self.sample_frames = params.sample_frames
        self.patch_size = params.patch_size
        self.temporal_compression_ratio = params.temporal_compression_ratio
        self.max_text_seq_length = params.max_text_seq_length
        self.activation_fn_str = params.activation_fn # Keep string for CogVideoXBlock
        self.timestep_activation_fn = params.timestep_activation_fn
        self.norm_elementwise_affine = params.norm_elementwise_affine
        self.norm_eps = params.norm_eps
        self.spatial_interpolation_scale = params.spatial_interpolation_scale
        self.temporal_interpolation_scale = params.temporal_interpolation_scale
        self.use_rotary_positional_embeddings = params.use_rotary_positional_embeddings


        if os.environ.get("NUM_COG_LAYERS"):
            num_layers = eval(os.environ.get("NUM_COG_LAYERS"))
            assert isinstance(num_layers, int)
            self.num_layers = num_layers
        # --- 2. Derive dependent parameters ---
        self.inner_dim = self.num_attention_heads * self.attention_head_dim
        
        # --- 3. Populate the generic `TransformerConfig` object ---
        config.hidden_size = self.inner_dim
        config.num_attention_heads = self.num_attention_heads
        config.num_query_groups = self.num_attention_heads # CogVideoX uses MHA, not GQA
        config.use_cpu_initialization = True # As per WanModel
        config.activation_func = _get_activation_fn(self.activation_fn_str) # Convert string to func
        config.hidden_dropout = self.dropout
        config.attention_dropout = self.dropout
        config.layernorm_epsilon = self.norm_eps
        config.add_qkv_bias = self.attention_bias
        config.rotary_interleaved = True # As per WanModel

        # --- 4. Call the parent class __init__ with the populated config ---
        super().__init__(config)

        # --- 5. Initialize CogVideoX-specific sub-modules ---
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.inner_dim,
            text_embed_dim=self.text_embed_dim,
            bias=True,
            sample_width=self.sample_width,
            sample_height=self.sample_height,
            sample_frames=self.sample_frames,
            temporal_compression_ratio=self.temporal_compression_ratio,
            max_text_seq_length=self.max_text_seq_length,
            spatial_interpolation_scale=self.spatial_interpolation_scale,
            temporal_interpolation_scale=self.temporal_interpolation_scale,
            use_positional_embeddings=not self.use_rotary_positional_embeddings,
        )
        self.embedding_dropout = nn.Dropout(self.dropout)

        self.time_proj = Timesteps(self.inner_dim, self.flip_sin_to_cos, self.freq_shift)
        self.time_embedding = TimestepEmbedding(
            self.inner_dim, self.time_embed_dim, self.timestep_activation_fn
        )

        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.num_attention_heads,
                    attention_head_dim=self.attention_head_dim,
                    time_embed_dim=self.time_embed_dim,
                    dropout=self.dropout,
                    activation_fn=self.activation_fn_str, # CogVideoXBlock expects the string name
                    attention_bias=self.attention_bias,
                    norm_elementwise_affine=self.norm_elementwise_affine,
                    norm_eps=self.norm_eps,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.norm_final = nn.LayerNorm(self.inner_dim, self.norm_eps, self.norm_elementwise_affine)

        self.norm_out = AdaLayerNorm(
            embedding_dim=self.time_embed_dim,
            output_dim=2 * self.inner_dim,
            norm_elementwise_affine=self.norm_elementwise_affine,
            norm_eps=self.norm_eps,
            chunk_dim=1,
        )
        self.proj_out = nn.Linear(self.inner_dim, self.patch_size * self.patch_size * self.out_channels)

        self.gradient_checkpointing = False

    def _get_block(self, dit_type: str, layer_number: int) -> nn.Module:
            """Helper method to retrieve a specific transformer block by its index."""
            if dit_type == "cog_block":
                return self.transformer_blocks[layer_number]
    

    def _checkpointed_forward(
            self, 
            dit_type: str,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            # img_and_txt: tuple,``
            *args
    ):
        "Forward method with activation checkpointing."
        if dit_type == "double_stream" or dit_type == "wan_block" or dit_type == "cog_block":
            # num_layers = len(self.transformer_blocks)
            recompute_layers = self.num_layers
            # print("recompute_layers_double",recompute_layers)
        elif dit_type == "single_stream":
            # num_layers = len(self.single_transformer_blocks)
            recompute_layers = self.num_single_layers
            # print("recompute_layers_single",recompute_layers)
        else:
            raise NotImplementedError(f"dit type: {dit_type} is not implemented! ")
        
        def custom(start, end):
            def custom_forward(*args):
                for index in range(start, end):
                    layer = self._get_block(dit_type, index)
                    x_ = layer(*args)
                return x_
            return custom_forward
        
        if self.config.recompute_method == "uniform":
            # Uniformly divide the total number of Transformer layers and
            # checkpoint the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            _layer_num = 0
            while _layer_num < self.num_layers:
                hidden_states, encoder_hidden_states = tensor_parallel.checkpoint(
                    custom(_layer_num, _layer_num + recompute_layers),
                    self.config.distribute_saved_activations,
                    hidden_states,
                    encoder_hidden_states,
                    *args
                )
                _layer_num += recompute_layers

        elif self.config.recompute_method == "block":
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            if os.environ.get("PROFILE_MEMORY"):
                from my_utils import ProfilerWrapper
                profiler = ProfilerWrapper(is_st=False, enable_record_cuda_mm=True)

            from my_utils import global_timer
            for _layer_num in range(recompute_layers):
                layer_str = f"layer_{_layer_num}_bs_{hidden_states.shape[0]}_f_{os.environ.get('frames')}_h_{os.environ.get('height')}_w_{os.environ.get('width')}_sp{mpu.get_context_parallel_world_size()}"
                mem_before_bytes = torch.cuda.memory_allocated()
                # print(layer_str)
                if os.environ.get("ENABLE_PROFILE_DIT_LAYER") == "1":
                
                    global_timer.start(layer_str)
                if _layer_num < recompute_layers:
                    # logger.info(f'[{dit_type}] layer_num {_layer_num}, hidden_states {hidden_states.shape}, encoder_hidden_states = {encoder_hidden_states.shape}')
                    hidden_states, encoder_hidden_states = tensor_parallel.checkpoint(
                        custom(_layer_num, _layer_num + 1),
                        self.config.distribute_saved_activations,
                        hidden_states,
                        encoder_hidden_states,
                        *args
                    )
                else:
                    block = self._get_block(dit_type, _layer_num)
                    hidden_states, encoder_hidden_states = block(*hidden_states, *encoder_hidden_states, *args)
                if os.environ.get("ENABLE_PROFILE_DIT_LAYER") == "1":    
                    global_timer.stop(layer_str)    
                mem_allocated_bytes = torch.cuda.memory_allocated()
                mem_reserved_bytes = torch.cuda.memory_reserved()
                
                # Calculate the change in allocated memory
                delta_bytes = mem_allocated_bytes - mem_before_bytes
                
                # Convert all values to Gigabytes (GB)
                delta_gb = delta_bytes / (1024 ** 3)
                allocated_gb = mem_allocated_bytes / (1024 ** 3)
                reserved_gb = mem_reserved_bytes / (1024 ** 3)

                # Log the detailed memory information
                log_msg = (
                    f"[Rank {torch.distributed.get_rank()}] After layer {_layer_num}: "
                    f"Delta: {delta_gb:+.3f}GB | "
                    f"Allocated: {allocated_gb:.3f}GB | "
                    f"Reserved: {reserved_gb:.3f}GB"
                )
                # global_timer.logger.info(log_msg)

                
                if os.environ.get("PROFILE_MEMORY"):
                    profiler.record()
        else:
            raise ValueError(f"Invalid activation recompute method {self.recompute_method}.")
        
        return  hidden_states, encoder_hidden_states


    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        need_broadcast: bool = True,
        return_dict: bool = True,
    ):
        # sp_group = get_sequence_parallel_group()
        # if sp_group is not None:
        #     if need_broadcast:
        #         hidden_states = broadcast(hidden_states, group=sp_group)
        #         encoder_hidden_states = broadcast(encoder_hidden_states, group=sp_group)
        #         timestep = broadcast(timestep, group=sp_group)
        #         if timestep_cond is not None:
        #             timestep_cond = broadcast(timestep_cond, group=sp_group)
        #     hidden_states = split_forward_gather_backward(
        #         hidden_states, dim=1, group=sp_group
        #     )
        #     encoder_hidden_states = split_forward_gather_backward(
        #         encoder_hidden_states, dim=1, group=sp_group
        #     )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)
        # import pdb; pdb.set_trace()

        # set encoder_hidden_states length to 226
        
        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        if mpu.get_context_parallel_world_size() > 1:
            from megatron.core.tensor_parallel.mappings import (
                split_forward_gather_backward,
                gather_forward_split_backward,
            )
            length = hidden_states.shape[1]
            set_origin_length(length)
            seq_parallel_world_size = mpu.get_context_parallel_world_size()
            if length % seq_parallel_world_size != 0:
                pad_size = seq_parallel_world_size - (length % seq_parallel_world_size)
                length = length + pad_size
            set_target_length(length)
            hidden_states = pad_for_context_parallel(hidden_states, 1)

            hidden_states = split_forward_gather_backward(
                hidden_states,
                mpu.get_context_parallel_group(),
                dim=1,
                grad_scale="none"
            )

        # 3. Transformer blocks
        # for i, block in enumerate(self.transformer_blocks):
        #     if self.training and self.gradient_checkpointing:

        #         def create_custom_forward(module):
        #             def custom_forward(*inputs):
        #                 return module(*inputs)

        #             return custom_forward

        #         ckpt_kwargs: Dict[str, Any] = (
        #             {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
        #         )
        #         hidden_states, encoder_hidden_states = (
        #             torch.utils.checkpoint.checkpoint(
        #                 create_custom_forward(block),
        #                 hidden_states,
        #                 encoder_hidden_states,
        #                 emb,
        #                 image_rotary_emb,
        #                 **ckpt_kwargs,
        #             )
        #         )
        #     else:
        #         hidden_states, encoder_hidden_states = block(
        #             hidden_states=hidden_states,
        #             encoder_hidden_states=encoder_hidden_states,
        #             temb=emb,
        #             image_rotary_emb=image_rotary_emb,
        #         )
        if self.config.recompute_granularity == "full":
            hidden_states, encoder_hidden_states = self._checkpointed_forward(
                "cog_block",
                hidden_states,
                encoder_hidden_states,
                emb,
                image_rotary_emb,
            )

        if mpu.get_context_parallel_world_size() > 1:
            hidden_states = gather_forward_split_backward(
                hidden_states,
                mpu.get_context_parallel_group(),
                dim=1,
                grad_scale="up",
            )
            hidden_states = remove_pad_for_context_parallel(hidden_states, 1)

        if not self.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        p = self.patch_size
        output = hidden_states.reshape(
            batch_size, num_frames, height // p, width // p, -1, p, p
        )
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

        # if sp_group is not None:
        #     output = gather_forward_split_backward(output, dim=1, group=sp_group)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
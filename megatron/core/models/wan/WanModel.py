import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Callable
from einops import rearrange
import torch.nn.functional as F
# from megatron.core import mpu

from megatron.core import mpu, tensor_parallel
from megatron.core.context_parallel import set_origin_length, set_target_length, pad_for_context_parallel, remove_pad_for_context_parallel
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    
T5_CONTEXT_TOKEN_NUMBER = 512   

class WanParams:
    # 1.3B
    hidden_size: int = 1536
    num_attention_heads: int = 12
    num_layers: int = 30
    ffn_dim: int = 8960

    # 35 layer
    # hidden_size: int = 5120
    # num_attention_heads: int = 40
    # num_layers: int = 40
    # ffn_dim: int = 13824


    attention_head_dim: int = 128
    in_channels: int = 16
    out_channels: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    
    
    eps: float = 1e-6
    patch_size: Tuple[int, int, int] = (1,2,2)
    activation_func: Callable = F.gelu
    add_qkv_bias: bool = True
    has_image_input: bool = False
    has_image_pos_emb: bool = False

def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads, type: str):
        super().__init__()
        self.num_heads = num_heads
        self.attn_type = type
        
    def forward(self, q, k, v):
        if self.attn_type == 'cross':
            x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        else:
            x = self.forward_attn(q,k,v,self.num_heads)
        return x
    def forward_attn(self, q, k, v, num_heads):
        cp_group = mpu.get_context_parallel_group()
        # print(f"[wan dit block] q shape is {q.shape}, cp_group world_size = {mpu.get_context_parallel_world_size()}")
        
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        from yunchang.comm.all_to_all import SeqAllToAll4D
        # qkv: b s/CP n d
        q = SeqAllToAll4D.apply(cp_group, q, 2, 1)
        k = SeqAllToAll4D.apply(cp_group, k, 2, 1)
        v = SeqAllToAll4D.apply(cp_group, v, 2, 1)

        # qkv: b s n/CP d
        q,k,v = map(
            lambda x: remove_pad_for_context_parallel(x, 1),
            [q,k,v]
        )

        if FLASH_ATTN_3_AVAILABLE:
            x = flash_attn_interface.flash_attn_func(q, k, v)[0]
            x = x.transpose(1, 2).contiguous()
        else:
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            x = F.scaled_dot_product_attention(q, k, v)

        x = pad_for_context_parallel(x,dim=2)
        x = SeqAllToAll4D.apply(
            cp_group, x, 2, 1
        )  # b img_seq sub_n d
        # torch.cuda.empty_cache()
        # x: b n s/CP d
        x = x.transpose(1, 2).flatten(2, 3).contiguous()
        # x: b s h

        return x

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads, "self")

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads, "cross")
        self.attn2 = AttentionModule(self.num_heads, "cross")

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            image_context_length = y.shape[1] - T5_CONTEXT_TOKEN_NUMBER
            img = y[:, :image_context_length]
            ctx = y[:, image_context_length:]
        else:
            ctx = y
            
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = self.attn2(q, k_img, v_img)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

from .layer import GateWithGradReduce, ModulateWithCPGradReduce

class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()
        self.gate2 = GateModule()

    def gate_with_cp_grad_reduce(self, x, gate, residual):
        return GateWithGradReduce.apply(x, gate, residual)
    
    def modulate_with_cp_grad_reduce(self, x, shift, scale):
        return ModulateWithCPGradReduce.apply(x, shift, scale)

    def forward(self, x, context, t_mod, freqs):
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        from my_utils import print_tensor_info
        # print_tensor_info(x, "transformer layer x")
        # print_tensor_info(freqs, "transformer layer freqs")
        modulation = self.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        modulation = modulation + t_mod
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=1)

        normed_x1 = self.norm1(x)
        modulated_x1 = self.modulate_with_cp_grad_reduce(normed_x1, shift_msa, scale_msa)
        attn_output = self.self_attn(modulated_x1, freqs)
        gated_x1 = self.gate_with_cp_grad_reduce(x, gate_msa, attn_output)

        normed_x3 = self.norm3(gated_x1)
        cross_attn_output = self.cross_attn(normed_x3, context)
        x = gated_x1 + cross_attn_output

        normed_x2 = self.norm2(x)
        modulated_x2 = self.modulate_with_cp_grad_reduce(normed_x2, shift_mlp, scale_mlp)
        ffn_output = self.ffn(modulated_x2)
        x = self.gate_with_cp_grad_reduce(x, gate_mlp, ffn_output)

        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x
import os
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.transformer_config import TransformerConfig
class WanModel(VisionModule):
    def __init__(self, wan_config: WanParams ,config: TransformerConfig):
        # wan_config

        self.in_dim = wan_config.in_channels
        self.ffn_dim = wan_config.ffn_dim
        self.out_dim = wan_config.out_channels
        self.text_dim = wan_config.text_dim
        self.freq_dim = wan_config.freq_dim
        self.eps = wan_config.eps
        self.patch_size = wan_config.patch_size
        self.has_image_input = wan_config.has_image_input
        self.has_image_pos_emb = wan_config.has_image_pos_emb
        self.num_heads = wan_config.num_attention_heads
        self.attention_head_dim = wan_config.attention_head_dim
        self.num_layers = wan_config.num_layers
        if os.environ.get("NUM_WAN_LAYERS"):
            num_layers = eval(os.environ.get("NUM_WAN_LAYERS"))
            assert isinstance(num_layers, int)
            self.num_layers = num_layers


        self.dim = self.num_heads * self.attention_head_dim
        self.hidden_size = self.dim
        #VisionModule __init__ Transformerconfig
        config.hidden_size = self.hidden_size
        config.num_attention_heads = self.num_heads
        config.num_query_groups = config.num_attention_heads
        config.use_cpu_initialization = True
        config.activation_func = wan_config.activation_func
        config.hidden_dropout=0
        config.attention_dropout=0
        config.layernorm_epsilon=1e-6
        config.add_qkv_bias = wan_config.add_qkv_bias
        config.rotary_interleaved=True
        config.attention_dropout = config.attention_dropout[0] if isinstance(config.attention_dropout, tuple) else config.attention_dropout
        transformer_config=config

        super().__init__(transformer_config)


        # tuple(1,2,2)  => HunyuanVideoPatchEmbed
        self.patch_embedding = nn.Conv3d(
            self.in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size)
        # text: 4096 => hidden_states
        self.text_embedding = nn.Sequential(
            nn.Linear(self.text_dim, self.dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(self.dim, self.dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(self.dim, self.dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(self.has_image_input, self.dim, self.num_heads, self.ffn_dim, self.eps)
            for _ in range(self.num_layers)
        ])

        # print(len(self.blocks))
        self.head = Head(self.dim, self.out_dim, self.patch_size, self.eps)
        head_dim = self.dim // self.num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if self.has_image_input:
            self.img_emb = MLP(1280, self.dim, has_pos_emb=self.has_image_pos_emb)  # clip_feature_dim = 1280

    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )
    def _get_block(
                self,
                dit_type: str,
                layer_number: int
        ):
            if dit_type == "wan_block":
                return self.blocks[layer_number]
            # elif dit_type == "single_stream":
            #     return self.single_transformer_blocks[layer_number]
            else:
                raise NotImplementedError(f"dit type: {dit_type} is not implemented! ")
            # Add this new method to your model class
        

    def _checkpointed_forward(
            self, 
            dit_type: str,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            # img_and_txt: tuple,``
            *args
    ):
        "Forward method with activation checkpointing."
        if dit_type == "double_stream" or dit_type == "wan_block":
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
                    hidden_states = tensor_parallel.checkpoint(
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
        
        return  hidden_states

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                cn_images=None, 
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))

        from my_utils import print_tensor_info
        # print_tensor_info(x, "x")
        # print_tensor_info(timestep, "time")
        # print_tensor_info(context, "context")
        context = self.text_embedding(context)
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        if cn_images is not None:
            x = torch.cat([x, cn_images], dim=1)  # (b, c_x + c_y, f, h, w)
        
        x, (f, h, w) = self.patchify(x)
        
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        
        x = x.contiguous()
        encoder_hidden_states = context.contiguous()
        temb = timestep.contiguous()

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        if mpu.get_context_parallel_world_size() > 1:
            from megatron.core.tensor_parallel.mappings import (
                split_forward_gather_backward,
                gather_forward_split_backward,
            )
            length = x.shape[1]
            set_origin_length(length)
            seq_parallel_world_size = mpu.get_context_parallel_world_size()
            if length % seq_parallel_world_size != 0:
                pad_size = seq_parallel_world_size - (length % seq_parallel_world_size)
                length = length + pad_size
            set_target_length(length)
            x = pad_for_context_parallel(x, 1)
            freqs = pad_for_context_parallel(freqs, 0)
            
            x = split_forward_gather_backward(
                x, 
                mpu.get_context_parallel_group(),
                dim=1,
                grad_scale="none"
            ) # b s n ds
            freqs = split_forward_gather_backward(
                freqs,
                mpu.get_context_parallel_group(),
                dim=0,
                grad_scale="none"
            )

        # def create_custom_forward(module):
        #     def custom_forward(*inputs):
        #         return module(*inputs)
        #     return custom_forward
        # for block in self.blocks:
        #     if self.training and use_gradient_checkpointing:
        #         if use_gradient_checkpointing_offload:
        #             with torch.autograd.graph.save_on_cpu():
        #                 x = torch.utils.checkpoint.checkpoint(
        #                     create_custom_forward(block),
        #                     x, context, t_mod, freqs,
        #                     use_reentrant=False,
        #                 )
        #         else:
        #             x = torch.utils.checkpoint.checkpoint(
        #                 create_custom_forward(block),
        #                 x, context, t_mod, freqs,
        #                 use_reentrant=False,
        #             )
        #     else:
        #         x = block(x, context, t_mod, freqs)

        if self.config.recompute_granularity == "full":
            x = self._checkpointed_forward(
                "wan_block",
                x,
                context,
                t_mod,
                freqs
            )
            # for block in self.blocks:
            #     x = torch.utils.checkpoint.checkpoint(
            #         create_custom_forward(block),
            #                 hidden_states, context, t_mod, freqs,
            #                 use_reentrant=False,
            #     )

        if mpu.get_context_parallel_world_size() > 1:


            with torch.cuda.nvtx.range("gather_forward"):
                x = gather_forward_split_backward(
                    x, 
                    mpu.get_context_parallel_group(),
                    dim=1,
                    grad_scale="none"
            )
                x = remove_pad_for_context_parallel(x, 1)
        
        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x


import numpy as np
from collections import defaultdict
import collections
def estimate_hunyuan_vae_encoder_flops(
    batch_size: int,
    input_t: int, 
    input_h: int, 
    input_w: int,
    in_channels: int = 3,
    out_channels: int = 3,
    latent_channels: int = 16,
    block_out_channels: tuple = (128, 256, 512, 512),
    layers_per_block: int = 2,
    temporal_compression_ratio: int = 4,
    spatial_compression_ratio: int = 8,
    mid_block_add_attention: bool = True,
    double_z: bool = True
):
    total_flops = 0.0
    
    # --- Helper: 3D Convolution FLOPs ---
    def get_conv3d_flops(t, h, w, cin, cout, k=3):
        # 2 * Batch * Output_Vol * Cin * Cout * Kernel_Vol
        return 2.0 * batch_size * t * h * w * cin * cout * (k**3)

    curr_t, curr_h, curr_w = input_t, input_h, input_w
    
    # =========================================================================
    # 1. Stem Layer
    # =========================================================================
    stem_channels = block_out_channels[0]
    total_flops += get_conv3d_flops(curr_t, curr_h, curr_w, in_channels, stem_channels)

    # =========================================================================
    # 2. Down Blocks
    # =========================================================================
    loop_output_channel = block_out_channels[0] 
    
    num_spatial_down = int(np.log2(spatial_compression_ratio))
    num_time_down = int(np.log2(temporal_compression_ratio))
    
    for i, target_out_channel in enumerate(block_out_channels):
        loop_input_channel = loop_output_channel
        loop_output_channel = target_out_channel
        is_final_block = (i == len(block_out_channels) - 1)
        
        if temporal_compression_ratio == 4:
            add_spatial = (i < num_spatial_down)
            add_time = (i >= (len(block_out_channels) - 1 - num_time_down) and not is_final_block)
        elif temporal_compression_ratio == 8:
            add_spatial = (i < num_spatial_down)
            add_time = (i < num_time_down)
        else:
            add_spatial, add_time = False, False

        stride_h = 2 if add_spatial else 1
        stride_w = 2 if add_spatial else 1
        stride_t = 2 if add_time else 1
        add_downsample = (add_spatial or add_time)

        # ResNet Layers
        block_flops = 0
        curr_resnet_in = loop_input_channel
        
        for _ in range(layers_per_block):
            curr_resnet_out = loop_output_channel
            
            # Conv 1 & Conv 2
            block_flops += get_conv3d_flops(curr_t, curr_h, curr_w, curr_resnet_in, curr_resnet_out)
            block_flops += get_conv3d_flops(curr_t, curr_h, curr_w, curr_resnet_out, curr_resnet_out)
            
            # Shortcut
            if curr_resnet_in != curr_resnet_out:
                block_flops += get_conv3d_flops(curr_t, curr_h, curr_w, curr_resnet_in, curr_resnet_out, k=1)
            
            curr_resnet_in = curr_resnet_out

        # Downsample Layer
        if add_downsample:
            next_t = curr_t // stride_t
            next_h = curr_h // stride_h
            next_w = curr_w // stride_w
            
            ds_flops = 2.0 * batch_size * next_t * next_h * next_w * loop_output_channel * loop_output_channel * (3**3)
            block_flops += ds_flops
            
            curr_t, curr_h, curr_w = next_t, next_h, next_w
            
        total_flops += block_flops

    # =========================================================================
    # 3. Mid Block
    # =========================================================================
    mid_block_layers = 1
    mid_channels = block_out_channels[-1]
    mid_flops = 0
    
    # Initial ResNet
    mid_flops += 2 * get_conv3d_flops(curr_t, curr_h, curr_w, mid_channels, mid_channels)

    # Loop
    for _ in range(mid_block_layers):
        # Attention
        if mid_block_add_attention:
            seq_len = curr_t * curr_h * curr_w
            
            # 4 * B * S * C^2
            mid_flops += 4 * 2 * batch_size * seq_len * (mid_channels**2)
            
            # 4 * B * S^2 * C
            mid_flops += 4 * batch_size * (seq_len**2) * mid_channels 
        
        # Follow-up ResNet
        mid_flops += 2 * get_conv3d_flops(curr_t, curr_h, curr_w, mid_channels, mid_channels)
    
    total_flops += mid_flops

    # =========================================================================
    # 4. Conv Out
    # =========================================================================
    final_in = block_out_channels[-1]
    final_out = 2 * latent_channels if double_z else latent_channels
    
    flops_out = get_conv3d_flops(curr_t, curr_h, curr_w, final_in, final_out)
    total_flops += flops_out
    
    return total_flops



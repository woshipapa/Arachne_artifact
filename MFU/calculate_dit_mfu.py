def estimate_hunyuan_dit_train_flops(
    B, T, H, W,
    patch_size,          # tuple (pt, ph, pw)
    hidden_dim,          # h (e.g., 3072)
    num_layers,          # total layers
    in_channels,
    out_channels,
    mlp_ratio=4.0,       # r
    ffn_dim=None,             # FFN hidden dim (e.g., 12288)
    sp_size=1,
    text_length=256,   
    use_checkpointing=False
):
    
        
    pt, ph, pw = patch_size
    s_global = (T // pt) * (H // ph) * (W // pw) + text_length

    print("s_global:", s_global)
    V_patch = pt * ph * pw
    
    # -------------------------------------------------------
    # -------------------------------------------------------
    
    # --- Part A: Input/Output Projections ---
    flops_io_global = 2 * B * s_global * hidden_dim * (in_channels * V_patch + out_channels * V_patch)

    # --- Part B: Transformer Backbone ---
    
    # Term 1: Linear Projections & MLP
    if ffn_dim is not None:
        flops_linear_global = (8 + 4 * ffn_dim / hidden_dim) * B * s_global * (hidden_dim**2)
    else:
        flops_linear_global = (8 + 4 * mlp_ratio) * B * s_global * (hidden_dim**2)

    # Term 2: Attention Matrix Ops (Attention Score + Aggregation)
    flops_attn_matrix_global = 4 * B * (s_global**2) * hidden_dim

    flops_per_layer_global = flops_linear_global + flops_attn_matrix_global
    
    flops_backbone_global = num_layers * flops_per_layer_global
    
    # --- Part C: Global Total Forward ---
    total_forward_flops_global = flops_io_global + flops_backbone_global
    
    # -------------------------------------------------------
    # -------------------------------------------------------
    forward_flops_per_gpu = total_forward_flops_global / sp_size
    
    # -------------------------------------------------------
    # -------------------------------------------------------
    # Cost = 1x Forward + 1x Re-Forward + 1x Backward (2x Fwd) = 4x Forward
    # Cost = 1x Forward + 1x Backward (2x Fwd) = 3x Forward
    train_factor = 4 if use_checkpointing else 3
    
    total_train_flops_per_gpu = forward_flops_per_gpu * train_factor
    
    return total_train_flops_per_gpu
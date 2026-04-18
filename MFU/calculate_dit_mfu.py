def estimate_hunyuan_dit_train_flops(
    B, T, H, W,          # 视频维度 (Latent Space)
    patch_size,          # tuple (pt, ph, pw)
    hidden_dim,          # h (e.g., 3072)
    num_layers,          # total layers
    in_channels,
    out_channels,
    mlp_ratio=4.0,       # r
    ffn_dim=None,             # FFN hidden dim (e.g., 12288)
    sp_size=1,           # 新增：Sequence Parallel Size (Context Parallel World Size)
    text_length=256,   
    use_checkpointing=False # 是否开启重计算 (决定系数是 3 还是 4)
):
    """
    计算 HunyuanVideo 在开启 SP (Ulysses) 后的【单卡】FLOPs
    """
    
        
    # 1. 计算 Global Sequence Length (s_global)
    pt, ph, pw = patch_size
    s_global = (T // pt) * (H // ph) * (W // pw) + text_length

    print("s_global:", s_global)
    V_patch = pt * ph * pw
    
    # -------------------------------------------------------
    # 核心逻辑：先算 Global FLOPs，再除以并行度 N
    # -------------------------------------------------------
    
    # --- Part A: Input/Output Projections ---
    # 这部分通常不算在 SP 核心区，但计算量占比很小，
    # 假设数据加载时已经切分好了 Batch 或 Sequence，
    # 简单起见，我们也认为它被均匀分摊到了各卡上。
    flops_io_global = 2 * B * s_global * hidden_dim * (in_channels * V_patch + out_channels * V_patch)

    # --- Part B: Transformer Backbone ---
    
    # Term 1: Linear Projections & MLP
    # 公式: 24 * B * s * h^2
    # 16Bsh^2 来自MLP
    if ffn_dim is not None:
        flops_linear_global = (8 + 4 * ffn_dim / hidden_dim) * B * s_global * (hidden_dim**2)
    else:
        flops_linear_global = (8 + 4 * mlp_ratio) * B * s_global * (hidden_dim**2)

    # Term 2: Attention Matrix Ops (Attention Score + Aggregation)
    # 公式: 4 * B * s^2 * h
    flops_attn_matrix_global = 4 * B * (s_global**2) * hidden_dim

    # 单层 Global 总和
    flops_per_layer_global = flops_linear_global + flops_attn_matrix_global
    
    # Backbone Global 总和
    flops_backbone_global = num_layers * flops_per_layer_global
    
    # --- Part C: Global Total Forward ---
    total_forward_flops_global = flops_io_global + flops_backbone_global
    
    # -------------------------------------------------------
    # 2. 转换为 Per-GPU FLOPs
    # -------------------------------------------------------
    # Ulysses 保证了计算负载均衡，每张卡承担 1/N
    forward_flops_per_gpu = total_forward_flops_global / sp_size
    
    # -------------------------------------------------------
    # 3. 考虑 Backward (Train FLOPs)
    # -------------------------------------------------------
    # 如果开启 Checkpointing (Full Recompute):
    # Cost = 1x Forward + 1x Re-Forward + 1x Backward (2x Fwd) = 4x Forward
    # 如果未开启:
    # Cost = 1x Forward + 1x Backward (2x Fwd) = 3x Forward
    train_factor = 4 if use_checkpointing else 3
    
    total_train_flops_per_gpu = forward_flops_per_gpu * train_factor
    
    return total_train_flops_per_gpu
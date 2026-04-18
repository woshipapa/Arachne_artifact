import numpy as np
import math
from collections import defaultdict, Counter

from .calculate_dit_mfu import estimate_hunyuan_dit_train_flops
from .calculate_vae_mfu import estimate_hunyuan_vae_encoder_flops


# =========================================================================
# Part 2: 修正后的 Pipeline 模拟器 (Temporal Loop -> Spatial Tiling)
# =========================================================================
class PipelineFlopsEstimatorTemporalLoop:
    def __init__(self):
        # 默认配置
        self.vae_config = {
            "in_channels": 33, "latent_channels": 16,
            "block_out_channels": (128, 256, 512, 512), "layers_per_block": 2,
            "temporal_compression_ratio": 4, "spatial_compression_ratio": 8,
            "mid_block_add_attention": True, "double_z": True
        }
        self.vae_tile_config = {
            "tile_h": 256, "h_stride": 192,
            "tile_w": 256, "w_stride": 192,
            "tile_frame": 16, "frame_stride": 12 # tile_sample_min_num_frames
        }
        self.dit_config = {
            "hidden_dim": 3072, "num_layers": 60, "patch_size": (1, 2, 2),
            "in_channels": 16, "out_channels": 16
        }

        self.wan_dit_config = {
            "hidden_dim": 5120, "num_layers": 40, "patch_size": (1, 2, 2),
            "in_channels": 16, "out_channels": 16, "ffn_dim": 13824
        }

    def simulate_pipeline_per_rank(self, bs: int, t: int, h: int, w: int, sp_degree: int, model_type: str = "hunyuan"):
        print(f"=== Pipeline Simulation (Temporal Serial -> Spatial SP): B={bs}, T={t}, H={h}, W={w}, SP={sp_degree} ===")

        rank_vae_flops = defaultdict(float)
        # 初始化
        for r in range(sp_degree):
            rank_vae_flops[r] = 0.0

        tc = self.vae_tile_config
        
        # ===========================================================
        # Step 1: VAE Temporal Loop (串行)
        # ===========================================================
        # 代码逻辑: for i in range(0, num_frames, self.tile_sample_stride_num_frames):
        
        temporal_chunks_count = 0
        
        # 遍历时间维度
        for f_start in range(0, t, tc['frame_stride']):
            temporal_chunks_count += 1
            
            # 计算当前时间块的输入 T 大小
            # 代码: tile = x[:, :, i : i + self.tile_sample_min_num_frames + 1, :, :]
            # 注意这个 +1，通常是为了重叠或 Padding，FLOPs 需要算上这个输入尺寸
            f_end_slice = min(f_start + tc['tile_frame'] + 1, t)
            
            # 当前时间块的实际 T 长度
            current_t_chunk = f_end_slice - f_start
            
            # ===========================================================
            # Step 2: VAE Spatial Tiling (并行分发)
            # ===========================================================
            # 对于每一个时间块，重新进行 Spatial Tiling 和 Round-Robin 分发
            
            # 2.1 生成 Spatial Tiles (只看 H, W)
            spatial_tile_coords = []
            for h_s in range(0, h, tc['h_stride']):
                for w_s in range(0, w, tc['w_stride']):
                    spatial_tile_coords.append((h_s, w_s))
            
            # 2.2 按 Shape 分组 (Spatial Tiles)
            tiles_by_shape = defaultdict(list)
            for (h_s, w_s) in spatial_tile_coords:
                h_e = min(h_s + tc['tile_h'], h)
                w_e = min(w_s + tc['tile_w'], w)
                
                # Shape 包含: (当前时间块T, 空间H, 空间W)
                shape = (current_t_chunk, h_e - h_s, w_e - w_s)
                tiles_by_shape[shape].append(shape)
            
            # 2.3 分发给 SP Ranks
            # 注意: 这里是每一次 Temporal Loop 都重新分发
            # Rank 0 每次都会拿到每组 Shape 的第一个 Tile
            for shape, tiles_list in sorted(tiles_by_shape.items()):
                count = len(tiles_list)
                
                for r in range(sp_degree):
                    # 计算该 Rank 在【当前时间块】的【当前Shape组】中分到的数量
                    # 逻辑: tiled_encode 内部独立调用，Round Robin 从 0 开始
                    num_tiles_local = len(range(r, count, sp_degree))
                    
                    if num_tiles_local > 0:
                        tile_flops = estimate_hunyuan_vae_encoder_flops(
                            batch_size=bs, # Batch 作用于 Tensor 内部
                            input_t=shape[0], input_h=shape[1], input_w=shape[2],
                            **self.vae_config
                        )
                        rank_vae_flops[r] += tile_flops * num_tiles_local

        print(f"[VAE] Processed {temporal_chunks_count} temporal chunks serially.")
        
        # 打印 VAE 负载情况
        vae_values = [rank_vae_flops[r] for r in range(sp_degree)]
        print(f"[VAE] Rank loads: Min={min(vae_values)/1e12:.2f}T, Max={max(vae_values)/1e12:.2f}T")
        print(f"[VAE] Load Imbalance Penalty: {(1 - (sum(vae_values)/sp_degree)/max(vae_values))*100:.2f}%")

        # ===========================================================
        # Step 3: DiT 计算 (Global)
        # ===========================================================
        lat_t = math.ceil(t / self.vae_config['temporal_compression_ratio'])
        lat_h = math.ceil(h / self.vae_config['spatial_compression_ratio'])
        lat_w = math.ceil(w / self.vae_config['spatial_compression_ratio'])
        if model_type == "wan":
            global_dit_flops = estimate_hunyuan_dit_train_flops(
                B=bs, T=lat_t, H=lat_h, W=lat_w,
                **self.wan_dit_config
            )
        elif model_type == "hunyuan":
            global_dit_flops = estimate_hunyuan_dit_train_flops(
                B=bs, T=lat_t, H=lat_h, W=lat_w,
                **self.dit_config
            )
        
        per_rank_dit_flops = global_dit_flops / sp_degree
        print(f"[DiT] Latent: {lat_t}x{lat_h}x{lat_w}, Adding {per_rank_dit_flops/1e12:.2f}T per rank.")

        # ===========================================================
        # Step 4: 总和
        # ===========================================================
        final_rank_flops = {}
        for r in range(sp_degree):
            final_rank_flops[r] = rank_vae_flops[r] + per_rank_dit_flops
            
        return final_rank_flops

# =========================================================================
# 运行
# =========================================================================
if __name__ == "__main__":
    estimator = PipelineFlopsEstimatorTemporalLoop()
    
    # 模拟参数
    BS, T, H, W, SP = 1, 81, 1056, 1932, 1
    MODEL_TYPE = "wan"
    final_flops = estimator.simulate_pipeline_per_rank(BS, T, H, W, SP, MODEL_TYPE)
    
    print("-" * 60)
    for r in sorted(final_flops.keys()):
        print(f"Rank {r}: {final_flops[r]/1e12:.4f} TFLOPs")
    
    max_flops = max(final_flops.values())
    print("-" * 60)
    print(f"Bottleneck (Max Rank): {max_flops/1e12:.4f} TFLOPs")
from typing import Tuple


class WanParams:
    hidden_states: int = 5120
    in_channels: int = 36
    out_channels: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 13824
    eps: float = 1e-6
    patch_size: Tuple[int,int,int] = (1, 2, 2)
    num_layers: int = 40
    has_image_input: bool = True
    has_image_pos_emb: bool = False

    
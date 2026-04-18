from functools import partial

import numpy as np
import torch
import torch.nn as nn
from vast.utils.acceleration import broadcast, get_sequence_parallel_group


class CogVideoXLoss(nn.Module):
    def __init__(
        self,
        num_timesteps=1000,
        shift_scale=1.0,
        uniform_sampling=True,
        offset_noise_level=0.0,
        min_snr_value=None,
        type="l2",
        world_size=1,
        rank=0,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.uniform_sampling = uniform_sampling
        self.offset_noise_level = offset_noise_level
        self.min_snr_value = min_snr_value
        self.type = type
        discretizer = ZeroSNRDDPMDiscretization(
            num_timesteps=num_timesteps, shift_scale=shift_scale
        )
        self.sigmas = discretizer(num_timesteps, do_append_zero=False, flip=True)
        if self.uniform_sampling:
            self.group_num = world_size
            self.group_width = (
                world_size // self.group_num
            )  # the number of rank in one group
            self.sigma_interval = self.num_timesteps // self.group_num
            self.world_size = world_size
            self.rank = rank

    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss

    def add_noise(self, input_):
        self.sigmas = self.sigmas.to(input_.device)

        if self.uniform_sampling:
            group_index = self.rank // self.group_width
            start_idx = group_index * self.sigma_interval
            end_idx = (group_index + 1) * self.sigma_interval
            if self.rank == self.world_size - 1:
                end_idx = self.num_timesteps
            timesteps = torch.randint(start_idx, end_idx, (input_.shape[0],)).to(
                input_.device
            )
        else:
            timesteps = torch.randint(0, self.num_timesteps, (input_.shape[0],)).to(
                input_.device
            )

        noise = torch.randn_like(input_)
        if self.offset_noise_level > 0.0:
            offset_noise = append_dims(
                torch.randn(input_.shape[0]).to(input_.device), input_.ndim
            )
            noise = noise + offset_noise * self.offset_noise_level

        sp_group = get_sequence_parallel_group()
        if sp_group is not None:
            input_ = broadcast(input_, group=sp_group)
            timesteps = broadcast(timesteps, group=sp_group)
            noise = broadcast(noise, group=sp_group)

        alphas_cumprod_sqrt = append_dims(self.sigmas[timesteps], input_.ndim)

        noised_input = (
            input_.float() * alphas_cumprod_sqrt
            + noise * (1 - alphas_cumprod_sqrt**2) ** 0.5
        )

        w = 1 / (1 - alphas_cumprod_sqrt**2)  # v-pred
        if self.min_snr_value is not None:
            w = min(w, self.min_snr_value)

        self.c_skip = alphas_cumprod_sqrt
        self.c_out = -((1 - alphas_cumprod_sqrt**2) ** 0.5)
        self.input = input_
        self.noised_input = noised_input
        self.w = w

        return self.noised_input, timesteps

    def forward(self, model_pred):
        denoised = self.noised_input * self.c_skip + model_pred * self.c_out
        return self.get_loss(denoised, self.input, self.w).mean()


class ZeroSNRDDPMDiscretization:
    def __init__(
        self,
        linear_start=0.00085,
        linear_end=0.0120,
        num_timesteps=1000,
        shift_scale=1.0,  # noise schedule t_n -> t_m: logSNR(t_m) = logSNR(t_n) - log(shift_scale)
        keep_start=False,
        post_shift=False,
    ):
        super().__init__()
        if keep_start and not post_shift:
            linear_start = linear_start / (
                shift_scale + (1 - shift_scale) * linear_start
            )
        self.num_timesteps = num_timesteps
        betas = make_beta_schedule(
            "linear", num_timesteps, linear_start=linear_start, linear_end=linear_end
        )
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.to_torch = partial(torch.tensor, dtype=torch.float32)

        # SNR shift
        if not post_shift:
            self.alphas_cumprod = self.alphas_cumprod / (
                shift_scale + (1 - shift_scale) * self.alphas_cumprod
            )

        self.post_shift = post_shift
        self.shift_scale = shift_scale

    def get_sigmas(self, n, device="cpu", return_idx=False):
        if n < self.num_timesteps:
            timesteps = generate_roughly_equally_spaced_steps(n, self.num_timesteps)
            alphas_cumprod = self.alphas_cumprod[timesteps]
        elif n == self.num_timesteps:
            alphas_cumprod = self.alphas_cumprod
        else:
            raise ValueError

        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        alphas_cumprod = to_torch(alphas_cumprod)
        alphas_cumprod_sqrt = alphas_cumprod.sqrt()
        alphas_cumprod_sqrt_0 = alphas_cumprod_sqrt[0].clone()
        alphas_cumprod_sqrt_T = alphas_cumprod_sqrt[-1].clone()

        alphas_cumprod_sqrt -= alphas_cumprod_sqrt_T
        alphas_cumprod_sqrt *= alphas_cumprod_sqrt_0 / (
            alphas_cumprod_sqrt_0 - alphas_cumprod_sqrt_T
        )

        if self.post_shift:
            alphas_cumprod_sqrt = (
                alphas_cumprod_sqrt**2
                / (self.shift_scale + (1 - self.shift_scale) * alphas_cumprod_sqrt**2)
            ) ** 0.5

        if return_idx:
            return torch.flip(alphas_cumprod_sqrt, (0,)), timesteps
        else:
            return torch.flip(alphas_cumprod_sqrt, (0,))  # sqrt(alpha_t): 0 -> 0.99

    def __call__(
        self, n, do_append_zero=True, device="cpu", flip=False, return_idx=False
    ):
        if return_idx:
            sigmas, idx = self.get_sigmas(n, device=device, return_idx=return_idx)
        else:
            sigmas = self.get_sigmas(n, device=device, return_idx=return_idx)
        sigmas = append_zero(sigmas) if do_append_zero else sigmas
        if return_idx:
            return sigmas if not flip else torch.flip(sigmas, (0,)), idx
        else:
            return sigmas if not flip else torch.flip(sigmas, (0,))


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims
    dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def make_beta_schedule(
    schedule,
    n_timestep,
    linear_start=1e-4,
    linear_end=2e-2,
):
    if schedule == "linear":
        betas = (
            torch.linspace(
                linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64
            )
            ** 2
        )
    return betas.numpy()


def generate_roughly_equally_spaced_steps(
    num_substeps: int, max_step: int
) -> np.ndarray:
    return np.linspace(max_step - 1, 0, num_substeps, endpoint=False).astype(int)[::-1]

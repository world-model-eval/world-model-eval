import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import einops


class Diffusion(nn.Module):
    def __init__(
        self,
        timesteps: int = 1_000,
        sampling_timesteps: int = 10,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        alphas = 1 - self.sigmoid_beta_schedule(self.timesteps)
        # an unfortunate variable name, but it's the standard one
        self.register_buffer("alphas_cumprod", alphas.cumprod(dim=0), persistent=False)
        self.stabilization_level = 15
        self.device = torch.device(device) if device is not None else torch.device("cpu")

    def sigmoid_beta_schedule(self, timesteps: int, start: float = -3, end: float = 3, tau: float = 1) -> torch.Tensor:
        # https://arxiv.org/abs/2212.11972
        t = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64) / timesteps
        v_start = torch.tensor(start / tau).sigmoid()
        v_end = torch.tensor(end / tau).sigmoid()
        alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999).float()

    def q_sample(self, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        B, T = t.shape

        alphas_cumprod = self.alphas_cumprod[t.reshape(-1)].view(B, T, 1, 1, 1)

        return alphas_cumprod.sqrt() * x + (1 - alphas_cumprod).sqrt() * noise

    def loss_fn(self, model: nn.Module, x: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        B, T, D = actions.shape

        t = torch.randint(0, self.timesteps, (B, T), device=x.device, dtype=torch.long)
        noise = torch.randn_like(x)

        x_t = self.q_sample(x, t, noise)
        pred_v = model(x_t, t, actions)

        # Build target v
        alphas_cumprod = self.alphas_cumprod[t.reshape(-1)].view(B, T, 1, 1, 1)
        target_v = alphas_cumprod.sqrt() * noise - (1 - alphas_cumprod).sqrt() * x

        loss = F.mse_loss(pred_v, target_v)
        return loss

    def ddim_sample_step(
        self,
        model: nn.Module,
        x: torch.Tensor,
        actions: torch.Tensor,
        t_idx: torch.Tensor,
        t_next_idx: torch.Tensor,
    ) -> torch.Tensor:
        # Derived from
        # https://github.com/buoyancy99/diffusion-forcing/blob/475e0bcab87545e48b24b39fb46a81fe59d80594/algorithms/diffusion_forcing/models/diffusion.py#L383
        B, T, H, W, C = x.shape
        B, T, D = actions.shape
        B, T = t_idx.shape

        sampling_noise_steps = torch.linspace(
            -1,
            self.timesteps - 1,
            steps=self.sampling_timesteps + 1,
            device=x.device,
            dtype=torch.long,
        )
        t = sampling_noise_steps[t_idx]
        t_next = sampling_noise_steps[t_next_idx]

        clipped_t = torch.where(t < 0, torch.full_like(t, self.stabilization_level - 1, dtype=torch.long), t)
        orig_x = x.clone().detach()
        scaled_context = self.q_sample(x, clipped_t, torch.zeros_like(x))
        x = torch.where(t.reshape(B, T, 1, 1, 1) < 0, scaled_context, x)

        alphas_cumprod = self.alphas_cumprod[t.reshape(-1)].view(B, T, 1, 1, 1)
        alphas_next_cumprod = torch.where(
            t_next < 0,
            torch.ones_like(t_next),
            self.alphas_cumprod[t_next.reshape(-1)].view(B, T),
        ).view(B, T, 1, 1, 1)
        c = (1 - alphas_next_cumprod).sqrt()

        v_pred = model(x, clipped_t, actions)
        x_start = alphas_cumprod.sqrt() * x - (1 - alphas_cumprod).sqrt() * v_pred
        pred_noise = ((1 / alphas_cumprod).sqrt() * x - x_start) / ((1 / alphas_cumprod) - 1).sqrt()
        x_pred = alphas_next_cumprod.sqrt() * x_start + c * pred_noise
        x_pred = torch.where(
            (t == t_next).view(B, T, 1, 1, 1),
            orig_x,
            x_pred,
        )
        return x_pred

    def generate_pyramid_scheduling_matrix(self, horizon: int) -> torch.Tensor:
        height = self.sampling_timesteps + horizon
        scheduling_matrix = torch.zeros((height, horizon), dtype=torch.long)
        for m in range(height):
            for t in range(horizon):
                scheduling_matrix[m, t] = self.sampling_timesteps + t - m
        return torch.clip(scheduling_matrix, 0, self.sampling_timesteps)

    def generate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        actions: torch.Tensor,
        n_context_frames: int = 1,
        n_frames: int = 1,
        horizon: int = 1,
        window_len: int | None = None,
    ) -> torch.Tensor:
        B, T, H, W, C = x.shape
        curr_frame = 0
        x_pred = x[:, :n_context_frames]
        curr_frame += n_context_frames

        pbar = tqdm(total=n_frames, initial=curr_frame, desc="Sampling")
        while curr_frame < n_frames:
            horizon = min(n_frames - curr_frame, horizon)
            scheduling_matrix = self.generate_pyramid_scheduling_matrix(horizon)

            chunk = torch.randn((B, horizon, *x.shape[-3:]), device=self.device)
            x_pred = torch.cat([x_pred, chunk], dim=1)

            # Adjust context length
            start_frame = max(0, curr_frame + horizon - (window_len or model.max_frames))

            pbar.set_postfix(
                {
                    "start": start_frame,
                    "end": curr_frame + horizon,
                }
            )

            for m in range(scheduling_matrix.shape[0] - 1):
                t, t_next = scheduling_matrix[m], scheduling_matrix[m + 1]
                t, t_next = map(lambda x: einops.repeat(x, "t -> b t", b=B), (t, t_next))
                t, t_next = map(
                    lambda x: torch.cat((torch.zeros((B, curr_frame), dtype=torch.long), x), dim=1),
                    (t, t_next),
                )

                x_pred[:, start_frame:] = self.ddim_sample_step(
                    model,
                    x_pred[:, start_frame:],
                    actions[:, start_frame : curr_frame + horizon],
                    t[:, start_frame:],
                    t_next[:, start_frame:],
                )

            curr_frame += horizon
            pbar.update(horizon)

        return x_pred

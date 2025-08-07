import fire
from pathlib import Path
import imageio
import os
import datetime
import logging
from copy import deepcopy
from collections import OrderedDict
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.distributed as dist
from dataset import OpenXMP4VideoDataset
from model import DiT
from vae import VAE
from diffusion import Diffusion
from tensorboardX import SummaryWriter


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    # https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/train.py#L40
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model: torch.nn.Module, flag: bool = True) -> None:
    """Set the requires_grad flag for all parameters of ``model``."""
    for p in model.parameters():
        p.requires_grad = flag


def init_distributed() -> tuple[int, int, int, bool]:
    """Initialize torch.distributed if available.

    Returns a tuple of (local_rank, global_rank, world_size, is_distributed).
    """
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return local_rank, global_rank, world_size, True
    return 0, 0, 1, False


def main(
    dataset_dir: Path = Path("sample_data"),
    checkpoint_dir: Path | None = None,
    # Dataset
    input_h: int = 256,
    input_w: int = 256,
    n_frames: int = 10,
    frame_skip: int = 1,
    subset_names: str = "bridge",
    action_dim: int = 10,
    num_workers: int = 16,
    # Training
    batch_size: int = 4,
    timesteps: int = 1_000,
    lr: float = 3e-5,
    ema_decay: float = 0.999,
    max_train_steps: int = 500_000,
    action_dropout_prob: float = 0.0,
    # Architecture
    patch_size: int = 2,
    model_dim: int = 1024,
    layers: int = 16,
    heads: int = 16,
    # Logging
    validate_every: int = 20_000,
    log_every: int = 100,
    # Sampling
    sampling_timesteps: int = 10,
    window_len: int | None = None,
    horizon: int = 1,
    cfg: float = 0.0,
) -> None:
    assert torch.cuda.is_available(), "CUDA device required for training"

    local_rank, rank, world_size, distributed = init_distributed()
    device = f"cuda:{local_rank}" if distributed else "cuda"
    device = torch.device(device)
    train_dataset = OpenXMP4VideoDataset(
        save_dir=dataset_dir,
        input_h=input_h,
        input_w=input_w,
        n_frames=n_frames,
        frame_skip=frame_skip,
        action_dim=action_dim,
        subset_names=subset_names,
        split="train",
    )
    val_dataset = OpenXMP4VideoDataset(
        save_dir=dataset_dir,
        input_h=input_h,
        input_w=input_w,
        n_frames=n_frames,
        frame_skip=frame_skip,
        action_dim=action_dim,
        subset_names=subset_names,
        split="test",
    )

    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    )
    val_sampler = (
        DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    vae = VAE().to(device)
    model = DiT(
        in_channels=vae.vae.config.latent_channels,
        patch_size=patch_size,
        dim=model_dim,
        num_layers=layers,
        num_heads=heads,
        action_dim=action_dim,
        max_frames=n_frames,
        action_dropout_prob=action_dropout_prob,
    ).to(device)

    diffusion = Diffusion(
        timesteps=timesteps,
        sampling_timesteps=sampling_timesteps,
        device=device,
    ).to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        model_no_ddp = model.module
    else:
        model_no_ddp = model

    # Exponential Moving Average of model parameters
    ema = deepcopy(model_no_ddp).to(device)
    requires_grad(ema, False)
    update_ema(ema, model_no_ddp, ema_decay)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02, betas=(0.9, 0.99))

    if checkpoint_dir is None:
        run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = Path("outputs") / run_name
        logging.info(
            "No checkpoint_dir specified, using autogenerated directory %s",
            checkpoint_dir,
        )
    else:
        checkpoint_dir = Path(checkpoint_dir)
        logging.info("Using provided checkpoint_dir %s", checkpoint_dir)

    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    summary_writer = SummaryWriter(checkpoint_dir)

    # resume from latest checkpoint if available
    ckpts = sorted(checkpoint_dir.glob("ckpt_*.pt"))
    train_steps = 0
    if ckpts:
        latest = max(ckpts, key=lambda p: int(p.stem.split("_")[1]))
        data = torch.load(latest, map_location=device)
        state_dict = data["model"]
        model_no_ddp.load_state_dict(state_dict)
        optimizer.load_state_dict(data["optimizer"])
        if "ema" in data:
            ema.load_state_dict(data["ema"])
        else:
            update_ema(ema, model_no_ddp, 0.0)
        train_steps = int(data.get("step", 0))
        logging.info("Loaded checkpoint %s (step %d)", latest, train_steps)

    running_loss = torch.tensor(0.0)
    num_batches = 0
    loss_history: list[torch.Tensor] = []
    mse_history: list[torch.Tensor] = []
    pbar = tqdm(total=max_train_steps, desc="Training") if rank == 0 else None
    if pbar is not None:
        pbar.n = train_steps
        pbar.refresh()
    while train_steps < max_train_steps:
        try:
            x, actions = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, actions = next(train_iter)

        x = x.to(device)
        actions = actions.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x = vae.encode(x)
            loss = diffusion.loss_fn(model, x, actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_ema(ema, model_no_ddp, ema_decay)

        running_loss += loss.detach().cpu()
        num_batches += 1

        if train_steps == 0 or (train_steps + 1) % log_every == 0:
            avg_loss = running_loss / num_batches
            if distributed:
                avg_loss = avg_loss.to(device)
                dist.all_reduce(avg_loss)
                avg_loss /= world_size
                avg_loss_cpu = avg_loss.detach().cpu()
            else:
                avg_loss_cpu = avg_loss.detach().cpu()

            if rank == 0:
                loss_history.append(avg_loss_cpu)
                summary_writer.add_scalar("train/loss", avg_loss_cpu, train_steps)
                if pbar is not None:
                    pbar.set_postfix({"loss": avg_loss_cpu.item()})
                plt.figure()
                plt.plot(
                    [i * log_every for i in range(len(loss_history))],
                    [loss_tensor.cpu().numpy() for loss_tensor in loss_history],
                )
                plt.xlabel("step")
                plt.ylabel("loss")
                plt.tight_layout()
                plt.savefig(checkpoint_dir / "loss.png")
                plt.close()

            running_loss.zero_()
            num_batches = 0

        if train_steps == 0 or train_steps % validate_every == 0 and rank == 0:
            model.eval()
            with torch.no_grad():
                try:
                    val_x, val_actions = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    val_x, val_actions = next(val_iter)

                val_x = val_x.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    val_latent = vae.encode(val_x)
                    val_actions = val_actions.to(device)
                    ema.eval()
                    samples = diffusion.generate(
                        ema,
                        val_latent,
                        val_actions,
                        n_context_frames=1,
                        n_frames=val_latent.shape[1],
                        window_len=window_len,
                        horizon=horizon,
                        cfg=cfg,
                    )
                    samples = vae.decode(samples)
                mse = F.mse_loss(samples, val_x).detach().cpu()
                summary_writer.add_scalar("val/mse", mse, train_steps)
                mse_history.append(mse)
                plt.figure()
                plt.plot(
                    [i * validate_every for i in range(len(mse_history))],
                    [m.cpu().numpy() for m in mse_history],
                )
                plt.xlabel("step")
                plt.ylabel("mse")
                plt.tight_layout()
                plt.savefig(checkpoint_dir / "mse.png")
                plt.close()

            video_np = samples.float().clamp(0, 1).cpu().numpy()
            gt_np = val_x.float().clamp(0, 1).cpu().numpy()
            summary_writer.add_video(f"val/generation_cfg{cfg}", video_np, train_steps, dataformats="NTHWC")
            summary_writer.add_video("val/gt", gt_np, train_steps, dataformats="NTHWC")
            summary_writer.flush()
            step_str = f"{train_steps:09d}"
            video_path = checkpoint_dir / f"gen_{step_str}.gif"
            imageio.mimsave(video_path, (video_np[0] * 255).astype(np.uint8), fps=8)
            torch.save(
                {
                    "model": model_no_ddp.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": train_steps,
                },
                checkpoint_dir / f"ckpt_{step_str}.pt",
            )
            model.train()

        train_steps += 1
        if pbar is not None:
            pbar.update(1)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)

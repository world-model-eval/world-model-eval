from pathlib import Path
from typing import Sequence
from tqdm import tqdm

import numpy as np
import torch
import einops
from pytorchvideo.data.encoded_video import EncodedVideo
from torch.utils.data import Dataset
from torchvision import transforms


class OpenXMP4VideoDataset(Dataset):
    def __init__(
        self,
        save_dir: str | Path,
        input_h: int,
        input_w: int,
        n_frames: int,
        *,
        frame_skip: int = 1,
        action_dim: int = 10,
        split: str = "train",
        subset_names: Sequence[str] | str | None = None,
        max_videos: int | None = None,
    ) -> None:
        super().__init__()

        if split not in {"train", "test"}:
            raise ValueError(f"Unknown split: {split}")

        if isinstance(subset_names, str):
            subset_names = subset_names.split(",")
        self.save_dir = Path(save_dir)
        if subset_names is None:
            subset_names = [p.name for p in self.save_dir.iterdir() if p.is_dir()]
        self.subset_names = list(subset_names)

        self.n_frames = int(n_frames)
        self.frame_skip = int(frame_skip)
        self.clip_len = self.n_frames * self.frame_skip
        self.action_dim = int(action_dim)

        self.transform = transforms.Resize((int(input_h), int(input_w)))

        self.video_paths: list[Path] = []
        self.video_lengths: list[int] = []
        for name in self.subset_names:
            subset_dir = self.save_dir / name / split
            mp4_files = sorted(subset_dir.glob("*.mp4"))
            if max_videos is not None:
                mp4_files = mp4_files[:max_videos]
            for mp4 in tqdm(mp4_files, desc=f"Loading {name} {split} videos"):
                action_path = mp4.with_suffix(".npz")
                if not action_path.exists():
                    continue
                try:
                    length = int(np.load(action_path)["arr_0"].shape[0])
                except Exception:
                    continue
                if length >= self.clip_len:
                    self.video_paths.append(mp4)
                    self.video_lengths.append(length)

        if not self.video_paths:
            raise RuntimeError(f"No valid videos found in {self.save_dir} for subsets {self.subset_names}")

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        video_path = self.video_paths[idx]
        length = self.video_lengths[idx]
        action_path = video_path.with_suffix(".npz")

        start = np.random.randint(0, length - self.clip_len + 1)

        video = EncodedVideo.from_path(video_path, decode_audio=False)
        fps = video._container.streams.video[0].guessed_rate
        start_sec = start / fps
        end_sec = (start + self.clip_len) / fps
        clip = video.get_clip(start_sec=start_sec, end_sec=end_sec)["video"]
        clip = einops.rearrange(clip, "c t h w -> t h w c")

        actions = np.load(action_path)["arr_0"][start : start + self.clip_len]
        assert actions.shape[1] == self.action_dim, f"Unexpected action dim: {actions.shape[1]} != {self.action_dim}"

        clip = clip[:: self.frame_skip]
        actions = actions[:: self.frame_skip]
        assert len(clip) == self.n_frames
        assert len(actions) == self.n_frames

        clip = clip.float() / 255.0
        clip = einops.rearrange(clip, "t h w c -> t c h w")
        clip = self.transform(clip)
        clip = einops.rearrange(clip, "t c h w -> t h w c")
        actions = torch.from_numpy(actions).float()
        return clip, actions

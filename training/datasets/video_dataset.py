
"""
NeuroWorld AI
File: training/datasets/video_dataset.py

Video dataset for:
- VQ-VAE tokenizer training
- Video prediction
- World-model representation learning
- Temporal consistency learning

Supported inputs:
- Video files: MP4, AVI, MOV, MKV, WEBM
- Directories containing image frames
- Individual image files

Output:
    context_frames: [C, 3, H, W]
    target_frames:  [P, 3, H, W]
    frames:         [C + P, 3, H, W]
    timestamps:     [C + P]
    frame_mask:     [C + P]
    file_index:     scalar
    start_index:    scalar

Dependencies:
    torch
    numpy
    pillow
    opencv-python
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov",
    ".mkv", ".webm",
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp",
}


@dataclass
class VideoDatasetConfig:
    path: str

    image_size: int = 128

    context_frames: int = 8
    prediction_frames: int = 4

    stride: int = 1
    frame_step: int = 1

    max_videos: int | None = None

    normalize: bool = True
    recursive: bool = True

    def __post_init__(self) -> None:
        for name in (
            "image_size",
            "context_frames",
            "prediction_frames",
            "stride",
            "frame_step",
        ):
            if getattr(self, name) < 1:
                raise ValueError(
                    f"{name} must be >= 1"
                )

        if (
            self.max_videos is not None
            and self.max_videos < 1
        ):
            raise ValueError(
                "max_videos must be >= 1"
            )


class VideoDataset(Dataset):
    """
    Video sequence dataset.

    Each sample contains context and future
    frames from the same source recording.

    No sequence crosses recording boundaries.

    Image folders are treated as separate
    recordings. A directory containing only
    images is treated as one recording.
    """

    def __init__(
        self,
        config: VideoDatasetConfig | str,
        transform: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        if isinstance(config, str):
            config = VideoDatasetConfig(
                path=config
            )

        self.config = config
        self.transform = transform

        self.root = Path(config.path)

        if not self.root.exists():
            raise FileNotFoundError(
                f"Video dataset path not found: "
                f"{self.root}"
            )

        self.sequence_length = (
            config.context_frames
            + config.prediction_frames
        )

        self.sources = self._discover_sources()

        if not self.sources:
            raise ValueError(
                "No supported videos or "
                "image sequences found"
            )

        self.frame_counts = [
            self._count_frames(source)
            for source in self.sources
        ]

        self.windows: list[
            tuple[int, int]
        ] = []

        self._build_windows()

        if not self.windows:
            raise ValueError(
                "No valid video sequences. "
                "Reduce context_frames or "
                "prediction_frames."
            )

    def _discover_sources(
        self,
    ) -> list[dict]:
        sources = []

        if self.root.is_file():
            suffix = self.root.suffix.lower()

            if suffix in VIDEO_EXTENSIONS:
                sources.append({
                    "type": "video",
                    "path": self.root,
                })

            elif suffix in IMAGE_EXTENSIONS:
                sources.append({
                    "type": "images",
                    "paths": [self.root],
                })

            return sources

        iterator = (
            self.root.rglob("*")
            if self.config.recursive
            else self.root.glob("*")
        )

        files = sorted(
            path
            for path in iterator
            if path.is_file()
        )

        for path in files:
            if (
                path.suffix.lower()
                in VIDEO_EXTENSIONS
            ):
                sources.append({
                    "type": "video",
                    "path": path,
                })

        # Group image files by their parent
        # directory to preserve sequence order.
        image_groups: dict[
            Path, list[Path]
        ] = {}

        for path in files:
            if (
                path.suffix.lower()
                in IMAGE_EXTENSIONS
            ):
                image_groups.setdefault(
                    path.parent,
                    [],
                ).append(path)

        for directory, paths in sorted(
            image_groups.items()
        ):
            sources.append({
                "type": "images",
                "path": directory,
                "paths": sorted(paths),
            })

        if self.config.max_videos is not None:
            sources = sources[
                :self.config.max_videos
            ]

        return sources

    @staticmethod
    def _count_frames(
        source: dict,
    ) -> int:
        if source["type"] == "images":
            return len(
                source["paths"]
            )

        capture = cv2.VideoCapture(
            str(source["path"])
        )

        if not capture.isOpened():
            capture.release()
            return 0

        count = int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        capture.release()

        return max(count, 0)

    def _build_windows(self) -> None:
        required_span = (
            (self.sequence_length - 1)
            * self.config.frame_step
            + 1
        )

        for file_index, count in enumerate(
            self.frame_counts
        ):
            if count < required_span:
                continue

            for start in range(
                0,
                count - required_span + 1,
                self.config.stride,
            ):
                self.windows.append(
                    (file_index, start)
                )

    def __len__(self) -> int:
        return len(self.windows)

    def _load_video_frames(
        self,
        path: Path,
        indices: list[int],
    ) -> list[np.ndarray]:
        capture = cv2.VideoCapture(
            str(path)
        )

        if not capture.isOpened():
            raise RuntimeError(
                f"Cannot open video: {path}"
            )

        frames = []

        try:
            for index in indices:
                capture.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    index,
                )

                success, frame = (
                    capture.read()
                )

                if not success:
                    raise RuntimeError(
                        f"Failed reading frame "
                        f"{index} from {path}"
                    )

                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                frames.append(frame)

        finally:
            capture.release()

        return frames

    @staticmethod
    def _load_image_frames(
        paths: list[Path],
        indices: list[int],
    ) -> list[np.ndarray]:
        frames = []

        for index in indices:
            with Image.open(
                paths[index]
            ) as image:
                frame = np.asarray(
                    image.convert("RGB")
                )

            frames.append(frame)

        return frames

    def _process_frame(
        self,
        frame: np.ndarray,
    ) -> Tensor:
        image = Image.fromarray(
            frame.astype(np.uint8)
        )

        image = image.resize(
            (
                self.config.image_size,
                self.config.image_size,
            ),
            resample=Image.Resampling.BILINEAR,
        )

        array = np.asarray(
            image,
            dtype=np.float32,
        ).copy()

        if self.config.normalize:
            array /= 255.0

        tensor = torch.from_numpy(
            array
        ).permute(
            2, 0, 1
        ).contiguous()

        return tensor

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Tensor]:
        file_index, start = (
            self.windows[index]
        )

        source = self.sources[
            file_index
        ]

        indices = [
            start + i * self.config.frame_step
            for i in range(
                self.sequence_length
            )
        ]

        if source["type"] == "video":
            frames = self._load_video_frames(
                source["path"],
                indices,
            )

        else:
            frames = self._load_image_frames(
                source["paths"],
                indices,
            )

        tensors = [
            self._process_frame(frame)
            for frame in frames
        ]

        video = torch.stack(
            tensors,
            dim=0,
        )

        if self.transform is not None:
            video = self.transform(
                video
            )

            if (
                not isinstance(video, Tensor)
                or video.shape
                != (
                    self.sequence_length,
                    3,
                    self.config.image_size,
                    self.config.image_size,
                )
            ):
                raise ValueError(
                    "Video transform must preserve "
                    "[T, 3, H, W] shape"
                )

        context_length = (
            self.config.context_frames
        )

        context = video[
            :context_length
        ]

        target = video[
            context_length:
        ]

        # Frame indices are provided as
        # timestamps in units of frames.
        # Convert using source FPS when
        # actual seconds are required.
        timestamps = torch.tensor(
            indices,
            dtype=torch.float32,
        )

        frame_mask = torch.ones(
            self.sequence_length,
            dtype=torch.bool,
        )

        return {
            "context_frames": context,
            "target_frames": target,
            "frames": video,
            "timestamps": timestamps,
            "frame_mask": frame_mask,
            "file_index": torch.tensor(
                file_index,
                dtype=torch.long,
            ),
            "start_index": torch.tensor(
                start,
                dtype=torch.long,
            ),
        }

    def summary(self) -> dict:
        return {
            "path": str(self.root),
            "recordings": len(self.sources),
            "sequences": len(self),
            "image_size": (
                self.config.image_size
            ),
            "context_frames": (
                self.config.context_frames
            ),
            "prediction_frames": (
                self.config.prediction_frames
            ),
            "frame_step": (
                self.config.frame_step
            ),
        }


def create_video_dataloader(
    config: VideoDatasetConfig,
    *,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 42,
    transform: Callable[[Tensor], Tensor] | None = None,
) -> DataLoader:
    dataset = VideoDataset(
        config,
        transform=transform,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


if __name__ == "__main__":
    # Create a demonstration image sequence.
    demo_dir = Path(
        "data/processed/videos/demo_sequence"
    )

    demo_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(42)

    for i in range(24):
        frame = rng.integers(
            0,
            256,
            size=(128, 128, 3),
            dtype=np.uint8,
        )

        Image.fromarray(
            frame
        ).save(
            demo_dir / f"frame_{i:04d}.png"
        )

    config = VideoDatasetConfig(
        path="data/processed/videos",
        image_size=128,
        context_frames=8,
        prediction_frames=4,
        stride=2,
    )

    loader = create_video_dataloader(
        config,
        batch_size=2,
    )

    print(
        "Dataset:",
        loader.dataset.summary(),
    )

    batch = next(iter(loader))

    for key, value in batch.items():
        print(
            key,
            tuple(value.shape),
        )

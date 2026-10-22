
```python
#!/usr/bin/env python3
"""
NeuroWorld AI
File: scripts/preprocess_videos.py

Video preprocessing pipeline for world-model training.

Features:
    - MP4, AVI, MOV, MKV, WEBM support
    - Video metadata inspection
    - Configurable FPS sampling
    - Aspect-ratio-preserving resize
    - Center crop or letterbox padding
    - RGB frame conversion
    - Optional float normalization
    - Fixed-length temporal sequences
    - Sliding-window sequence extraction
    - Train/validation/test split by source video
    - Compressed NPZ exports
    - JSON manifest and dataset summary
    - Corrupt video handling
    - Reproducible random seeds
    - CLI configuration

Dependencies:
    pip install opencv-python numpy

Example:
    python scripts/preprocess_videos.py \
        --input data/raw/videos \
        --output data/processed/videos \
        --width 128 \
        --height 128 \
        --fps 10 \
        --sequence-length 16 \
        --stride 8

Output:
    data/processed/videos/
        train/
        val/
        test/
        manifest.json
        summary.json

Each NPZ contains:
    frames: [T, H, W, 3]
    timestamps: [T]
    source_video: string
    sequence_index: integer

For normalized output:
    frames.dtype = float32
    frames range = [0, 1]

For non-normalized output:
    frames.dtype = uint8
    frames range = [0, 255]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

import cv2
import numpy as np


LOGGER = logging.getLogger(
    "neuroworld.video_preprocessing"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".m4v",
}


@dataclass
class VideoPreprocessConfig:
    input_dir: str = "data/raw/videos"
    output_dir: str = "data/processed/videos"

    width: int = 128
    height: int = 128

    target_fps: float = 10.0

    sequence_length: int = 16
    stride: int = 8

    resize_mode: str = "letterbox"

    normalize: bool = True

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    seed: int = 42

    max_videos: Optional[int] = None
    max_frames_per_video: Optional[int] = None

    recursive: bool = True
    overwrite: bool = False

    compression: bool = True

    min_valid_frames: int = 16


@dataclass
class VideoMetadata:
    source_path: str

    width: int
    height: int

    source_fps: float
    frame_count: int

    duration_seconds: float

    readable: bool
    error: Optional[str] = None


@dataclass
class SequenceRecord:
    sequence_id: str
    source_video: str
    split: str

    sequence_index: int

    frame_count: int

    start_time: float
    end_time: float

    output_path: str

    shape: list[int]
    dtype: str


@dataclass
class VideoProcessingResult:
    video_path: str
    split: str

    success: bool

    sampled_frames: int = 0
    sequences_created: int = 0

    processing_seconds: float = 0.0

    error: Optional[str] = None


class VideoPreprocessor:
    """
    Preprocess video files for temporal
    machine-learning datasets.

    Source videos are assigned to splits
    before sequence generation to avoid
    train/test leakage between windows
    of the same video.
    """

    def __init__(
        self,
        config: Optional[
            VideoPreprocessConfig
        ] = None,
    ) -> None:
        self.config = (
            config or VideoPreprocessConfig()
        )

        self._validate_config()

        self.input_dir = self._resolve_path(
            self.config.input_dir
        )

        self.output_dir = self._resolve_path(
            self.config.output_dir
        )

        self.records: list[SequenceRecord] = []
        self.results: list[VideoProcessingResult] = []

    @staticmethod
    def _resolve_path(
        value: str,
    ) -> Path:
        path = Path(value).expanduser()

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        return path.resolve()

    def _validate_config(self) -> None:
        c = self.config

        if c.width <= 0 or c.height <= 0:
            raise ValueError(
                "Frame dimensions must be positive"
            )

        if (
            not math.isfinite(c.target_fps)
            or c.target_fps <= 0
        ):
            raise ValueError(
                "target_fps must be positive and finite"
            )

        if c.sequence_length <= 0:
            raise ValueError(
                "sequence_length must be positive"
            )

        if c.stride <= 0:
            raise ValueError(
                "stride must be positive"
            )

        if c.min_valid_frames <= 0:
            raise ValueError(
                "min_valid_frames must be positive"
            )

        if c.resize_mode not in {
            "stretch",
            "center_crop",
            "letterbox",
        }:
            raise ValueError(
                "resize_mode must be stretch, "
                "center_crop or letterbox"
            )

        ratios = (
            c.train_ratio,
            c.val_ratio,
            c.test_ratio,
        )

        if any(
            not math.isfinite(value)
            or value < 0
            for value in ratios
        ):
            raise ValueError(
                "Split ratios must be nonnegative"
            )

        if not math.isclose(
            sum(ratios),
            1.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "Split ratios must sum to 1"
            )

        for value in (
            c.max_videos,
            c.max_frames_per_video,
        ):
            if value is not None and value <= 0:
                raise ValueError(
                    "Optional limits must be positive"
                )

    def discover_videos(self) -> list[Path]:
        if not self.input_dir.is_dir():
            raise FileNotFoundError(
                f"Input directory not found: "
                f"{self.input_dir}"
            )

        iterator = (
            self.input_dir.rglob("*")
            if self.config.recursive
            else self.input_dir.glob("*")
        )

        videos = sorted(
            path
            for path in iterator
            if (
                path.is_file()
                and path.suffix.lower()
                in VIDEO_EXTENSIONS
            )
        )

        if self.config.max_videos is not None:
            videos = videos[
                :self.config.max_videos
            ]

        return videos

    def inspect_video(
        self,
        path: Path,
    ) -> VideoMetadata:
        capture = cv2.VideoCapture(
            str(path)
        )

        if not capture.isOpened():
            capture.release()

            return VideoMetadata(
                source_path=str(path),
                width=0,
                height=0,
                source_fps=0.0,
                frame_count=0,
                duration_seconds=0.0,
                readable=False,
                error="Unable to open video",
            )

        try:
            width = int(
                capture.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            height = int(
                capture.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

            fps = float(
                capture.get(
                    cv2.CAP_PROP_FPS
                )
            )

            count = int(
                capture.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            valid = (
                width > 0
                and height > 0
                and math.isfinite(fps)
                and fps > 0
            )

            duration = (
                count / fps
                if valid and count > 0
                else 0.0
            )

            return VideoMetadata(
                source_path=str(path),
                width=width,
                height=height,
                source_fps=fps,
                frame_count=count,
                duration_seconds=duration,
                readable=valid,
                error=(
                    None
                    if valid
                    else "Invalid video metadata"
                ),
            )

        finally:
            capture.release()

    def resize_frame(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Convert BGR frame to RGB and resize.

        Modes:
            stretch:
                Direct resize.

            center_crop:
                Scale to cover and crop center.

            letterbox:
                Scale to fit and pad borders.
        """

        c = self.config

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        source_h, source_w = rgb.shape[:2]

        if source_h <= 0 or source_w <= 0:
            raise ValueError(
                "Invalid source frame dimensions"
            )

        target_w = c.width
        target_h = c.height

        if c.resize_mode == "stretch":
            return cv2.resize(
                rgb,
                (target_w, target_h),
                interpolation=cv2.INTER_AREA,
            )

        if c.resize_mode == "center_crop":
            scale = max(
                target_w / source_w,
                target_h / source_h,
            )

            new_w = max(
                target_w,
                round(source_w * scale),
            )

            new_h = max(
                target_h,
                round(source_h * scale),
            )

            resized = cv2.resize(
                rgb,
                (new_w, new_h),
                interpolation=(
                    cv2.INTER_AREA
                    if scale < 1
                    else cv2.INTER_LINEAR
                ),
            )

            x = (new_w - target_w) // 2
            y = (new_h - target_h) // 2

            return resized[
                y:y + target_h,
                x:x + target_w,
            ]

        scale = min(
            target_w / source_w,
            target_h / source_h,
        )

        new_w = max(
            1,
            min(
                target_w,
                round(source_w * scale),
            ),
        )

        new_h = max(
            1,
            min(
                target_h,
                round(source_h * scale),
            ),
        )

        resized = cv2.resize(
            rgb,
            (new_w, new_h),
            interpolation=(
                cv2.INTER_AREA
                if scale < 1
                else cv2.INTER_LINEAR
            ),
        )

        canvas = np.zeros(
            (target_h, target_w, 3),
            dtype=np.uint8,
        )

        x = (target_w - new_w) // 2
        y = (target_h - new_h) // 2

        canvas[
            y:y + new_h,
            x:x + new_w,
        ] = resized

        return canvas

    def sample_frames(
        self,
        path: Path,
        metadata: VideoMetadata,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Decode frames sequentially.

        Select the first decoded frame
        at or after each target timestamp.

        This uses source FPS metadata for
        timestamps, so it is most suitable
        for constant-frame-rate videos.
        """

        capture = cv2.VideoCapture(
            str(path)
        )

        if not capture.isOpened():
            raise RuntimeError(
                "Could not open video"
            )

        frames: list[np.ndarray] = []
        timestamps: list[float] = []

        source_index = 0
        next_sample_time = 0.0

        interval = (
            1.0 / self.config.target_fps
        )

        try:
            while True:
                success, frame = capture.read()

                if not success:
                    break

                timestamp = (
                    source_index
                    / metadata.source_fps
                )

                source_index += 1

                if (
                    timestamp + 1e-8
                    < next_sample_time
                ):
                    continue

                processed = self.resize_frame(
                    frame
                )

                frames.append(processed)
                timestamps.append(timestamp)

                next_sample_time += interval

                # Skip target timestamps that
                # fall before the next source
                # frame, avoiding duplicate
                # source frames.
                while (
                    next_sample_time
                    <= timestamp + 1e-8
                ):
                    next_sample_time += interval

                if (
                    self.config.max_frames_per_video
                    is not None
                    and len(frames)
                    >= self.config.max_frames_per_video
                ):
                    break

        finally:
            capture.release()

        if not frames:
            return (
                np.empty(
                    (
                        0,
                        self.config.height,
                        self.config.width,
                        3,
                    ),
                    dtype=np.uint8,
                ),
                np.empty(
                    (0,),
                    dtype=np.float64,
                ),
            )

        return (
            np.stack(
                frames,
                axis=0,
            ),
            np.asarray(
                timestamps,
                dtype=np.float64,
            ),
        )

    def generate_sequences(
        self,
        frames: np.ndarray,
        timestamps: np.ndarray,
    ) -> Iterator[
        tuple[int, np.ndarray, np.ndarray]
    ]:
        """
        Generate complete fixed-length
        windows without padding.
        """

        sequence_length = (
            self.config.sequence_length
        )

        stride = self.config.stride

        if len(frames) < sequence_length:
            return

        for index, start in enumerate(
            range(
                0,
                len(frames)
                - sequence_length + 1,
                stride,
            )
        ):
            end = start + sequence_length

            yield (
                index,
                frames[start:end],
                timestamps[start:end],
            )

    def split_videos(
        self,
        videos: list[Path],
    ) -> dict[str, list[Path]]:
        """
        Split by original video to avoid
        sharing overlapping windows between
        training and evaluation datasets.
        """

        rng = np.random.default_rng(
            self.config.seed
        )

        indices = rng.permutation(
            len(videos)
        )

        shuffled = [
            videos[int(index)]
            for index in indices
        ]

        count = len(shuffled)

        train_count = int(
            count * self.config.train_ratio
        )

        val_count = int(
            count * self.config.val_ratio
        )

        return {
            "train": shuffled[
                :train_count
            ],
            "val": shuffled[
                train_count:
                train_count + val_count
            ],
            "test": shuffled[
                train_count + val_count:
            ],
        }

    def _sequence_path(
        self,
        split: str,
        video_index: int,
        sequence_index: int,
    ) -> Path:
        directory = (
            self.output_dir / split
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return directory / (
            f"video_{video_index:06d}_"
            f"sequence_{sequence_index:06d}.npz"
        )

    def process_video(
        self,
        path: Path,
        split: str,
        video_index: int,
    ) -> VideoProcessingResult:
        started = time.perf_counter()

        result = VideoProcessingResult(
            video_path=str(path),
            split=split,
            success=False,
        )

        try:
            metadata = self.inspect_video(
                path
            )

            if not metadata.readable:
                raise RuntimeError(
                    metadata.error
                    or "Unreadable video"
                )

            frames, timestamps = (
                self.sample_frames(
                    path,
                    metadata,
                )
            )

            result.sampled_frames = len(
                frames
            )

            if (
                len(frames)
                < max(
                    self.config.min_valid_frames,
                    self.config.sequence_length,
                )
            ):
                raise RuntimeError(
                    "Video contains insufficient "
                    "sampled frames"
                )

            source_name = str(
                path.relative_to(
                    self.input_dir
                )
            )

            for (
                sequence_index,
                sequence,
                sequence_timestamps,
            ) in self.generate_sequences(
                frames,
                timestamps,
            ):
                output_path = self._sequence_path(
                    split,
                    video_index,
                    sequence_index,
                )

                if (
                    output_path.exists()
                    and not self.config.overwrite
                ):
                    raise FileExistsError(
                        f"Output already exists: "
                        f"{output_path}"
                    )

                if self.config.normalize:
                    sequence_data = (
                        sequence.astype(
                            np.float32
                        ) / 255.0
                    )
                else:
                    sequence_data = sequence

                payload = {
                    "frames": sequence_data,
                    "timestamps": (
                        sequence_timestamps
                    ),
                    "source_video": np.asarray(
                        source_name
                    ),
                    "sequence_index": np.asarray(
                        sequence_index,
                        dtype=np.int32,
                    ),
                }

                if self.config.compression:
                    np.savez_compressed(
                        output_path,
                        **payload,
                    )
                else:
                    np.savez(
                        output_path,
                        **payload,
                    )

                self.records.append(
                    SequenceRecord(
                        sequence_id=(
                            f"{split}_"
                            f"{video_index:06d}_"
                            f"{sequence_index:06d}"
                        ),
                        source_video=source_name,
                        split=split,
                        sequence_index=(
                            sequence_index
                        ),
                        frame_count=(
                            len(sequence_data)
                        ),
                        start_time=float(
                            sequence_timestamps[0]
                        ),
                        end_time=float(
                            sequence_timestamps[-1]
                        ),
                        output_path=str(
                            output_path.relative_to(
                                self.output_dir
                            )
                        ),
                        shape=list(
                            sequence_data.shape
                        ),
                        dtype=str(
                            sequence_data.dtype
                        ),
                    )
                )

                result.sequences_created += 1

            result.success = (
                result.sequences_created > 0
            )

            if not result.success:
                result.error = (
                    "No complete sequences generated"
                )

        except Exception as exc:
            result.error = str(exc)

            LOGGER.warning(
                "Failed processing %s: %s",
                path,
                exc,
            )

        result.processing_seconds = (
            time.perf_counter() - started
        )

        self.results.append(result)

        return result

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()

        self.records.clear()
        self.results.clear()

        videos = self.discover_videos()

        if not videos:
            raise RuntimeError(
                f"No videos found in {self.input_dir}"
            )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        splits = self.split_videos(
            videos
        )

        video_index = 0

        for split, split_videos in splits.items():
            LOGGER.info(
                "Processing %s: %d videos",
                split,
                len(split_videos),
            )

            for video in split_videos:
                result = self.process_video(
                    video,
                    split,
                    video_index,
                )

                LOGGER.info(
                    "%s | frames=%d | sequences=%d "
                    "| success=%s",
                    video.name,
                    result.sampled_frames,
                    result.sequences_created,
                    result.success,
                )

                video_index += 1

        manifest = {
            "project": "NeuroWorld AI",
            "dataset_type": (
                "video_temporal_sequences"
            ),
            "created_at": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "configuration": asdict(
                self.config
            ),
            "sequences": [
                asdict(record)
                for record in self.records
            ],
        }

        (
            self.output_dir / "manifest.json"
        ).write_text(
            json.dumps(
                manifest,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        successful = sum(
            result.success
            for result in self.results
        )

        split_counts = dict(
            Counter(
                record.split
                for record in self.records
            )
        )

        summary = {
            "total_videos": len(videos),
            "successful_videos": successful,
            "failed_videos": (
                len(videos) - successful
            ),
            "total_sequences": len(
                self.records
            ),
            "sequences_by_split": (
                split_counts
            ),
            "output_dir": str(
                self.output_dir
            ),
            "processing_seconds": round(
                time.perf_counter() - started,
                3,
            ),
            "video_results": [
                asdict(result)
                for result in self.results
            ],
        }

        (
            self.output_dir / "summary.json"
        ).write_text(
            json.dumps(
                summary,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        return summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "NeuroWorld AI video preprocessing"
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/videos",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/videos",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--resize-mode",
        choices=[
            "stretch",
            "center_crop",
            "letterbox",
        ],
        default="letterbox",
    )

    parser.add_argument(
        "--no-normalize",
        action="store_true",
    )

    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--no-compression",
        action="store_true",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    config = VideoPreprocessConfig(
        input_dir=args.input,
        output_dir=args.output,
        width=args.width,
        height=args.height,
        target_fps=args.fps,
        sequence_length=args.sequence_length,
        stride=args.stride,
        resize_mode=args.resize_mode,
        normalize=not args.no_normalize,
        max_videos=args.max_videos,
        max_frames_per_video=args.max_frames,
        overwrite=args.overwrite,
        compression=not args.no_compression,
        seed=args.seed,
    )

    processor = VideoPreprocessor(
        config
    )

    try:
        summary = processor.run()

    except Exception:
        LOGGER.exception(
            "Video preprocessing failed"
        )
        return 1

    print(
        json.dumps(
            summary,
            indent=2,
            allow_nan=False,
        )
    )

    return (
        0
        if summary["successful_videos"] > 0
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
```

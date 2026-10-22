
"""
NeuroWorld AI
File: neuroworld/perception/optical_flow.py

Optical Flow & Visual Motion Intelligence

Features:
    - Dense optical flow (Farneback)
    - Sparse optical flow (Lucas-Kanade)
    - Shi-Tomasi feature detection
    - Forward-backward tracking validation
    - Flow magnitude and direction
    - Moving-region detection
    - Motion bounding boxes
    - Global motion statistics
    - HSV optical-flow visualization
    - Motion heatmap generation
    - Frame warping using optical flow
    - PyTorch flow feature encoding
    - Video optical-flow processing
    - Frame-pair motion analysis
    - Structured motion reports
    - NeuroWorld motion-encoder integration

Dependencies:
    pip install opencv-python numpy torch

Run:
    python -m neuroworld.perception.optical_flow

Coordinate conventions:
    Flow[..., 0] = horizontal displacement (dx)
    Flow[..., 1] = vertical displacement (dy)

    Positive x points right.
    Positive y points down.

    Flow is measured in pixels per frame pair.

Important:
    Optical flow measures apparent image motion.
    It is not directly equivalent to physical
    object velocity or 3D scene motion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


# ============================================================
# EXCEPTIONS
# ============================================================

class OpticalFlowError(Exception):
    pass


class OpticalFlowValidationError(OpticalFlowError):
    pass


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class OpticalFlowConfig:
    # Farneback parameters
    pyramid_scale: float = 0.5
    levels: int = 3
    window_size: int = 15
    iterations: int = 3
    polynomial_n: int = 5
    polynomial_sigma: float = 1.2

    # Sparse feature tracking
    max_corners: int = 300
    quality_level: float = 0.01
    min_distance: float = 7.0

    lk_window_size: int = 21
    lk_max_level: int = 3

    forward_backward_threshold: float = 1.5

    # Motion detection
    motion_threshold: float = 1.5
    minimum_region_area: int = 100

    # Visualization
    visualization_max_magnitude: float = 20.0

    # Feature encoder
    embedding_dim: int = 128

    device: str = "cpu"

    def __post_init__(self) -> None:
        if not 0 < self.pyramid_scale < 1:
            raise ValueError(
                "pyramid_scale must be in (0,1)"
            )

        for name in (
            "levels",
            "window_size",
            "iterations",
            "polynomial_n",
            "max_corners",
            "lk_window_size",
            "embedding_dim",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"{name} must be positive"
                )

        if self.polynomial_n not in (5, 7):
            raise ValueError(
                "polynomial_n must be 5 or 7"
            )

        if not 0 < self.quality_level <= 1:
            raise ValueError(
                "quality_level must be in (0,1]"
            )

        if self.min_distance < 0:
            raise ValueError(
                "min_distance must be nonnegative"
            )

        if self.motion_threshold < 0:
            raise ValueError(
                "motion_threshold must be nonnegative"
            )

        if self.minimum_region_area <= 0:
            raise ValueError(
                "minimum_region_area must be positive"
            )

        if self.forward_backward_threshold <= 0:
            raise ValueError(
                "forward_backward_threshold must be positive"
            )

        if self.visualization_max_magnitude <= 0:
            raise ValueError(
                "visualization_max_magnitude must be positive"
            )


# ============================================================
# FRAME VALIDATION
# ============================================================

def validate_frame(
    frame: np.ndarray,
    name: str = "frame",
) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
        raise OpticalFlowValidationError(
            f"{name} must be a NumPy array"
        )

    if frame.dtype != np.uint8:
        raise OpticalFlowValidationError(
            f"{name} must have uint8 dtype"
        )

    if frame.ndim == 2:
        pass

    elif (
        frame.ndim == 3
        and frame.shape[2] in (3, 4)
    ):
        pass

    else:
        raise OpticalFlowValidationError(
            f"{name} must be grayscale, BGR, or BGRA"
        )

    if min(frame.shape[:2]) < 8:
        raise OpticalFlowValidationError(
            f"{name} is too small"
        )

    return frame


def to_grayscale(
    frame: np.ndarray,
) -> np.ndarray:
    validate_frame(frame)

    if frame.ndim == 2:
        return frame

    if frame.shape[2] == 4:
        return cv2.cvtColor(
            frame,
            cv2.COLOR_BGRA2GRAY,
        )

    return cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )


def validate_frame_pair(
    previous: np.ndarray,
    current: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    previous_gray = to_grayscale(
        previous
    )

    current_gray = to_grayscale(
        current
    )

    if (
        previous_gray.shape
        != current_gray.shape
    ):
        raise OpticalFlowValidationError(
            "Frames must have matching dimensions"
        )

    return previous_gray, current_gray


def validate_flow(
    flow: np.ndarray,
) -> np.ndarray:
    flow = np.asarray(
        flow,
        dtype=np.float32,
    )

    if (
        flow.ndim != 3
        or flow.shape[-1] != 2
    ):
        raise OpticalFlowValidationError(
            "Flow must have shape [H,W,2]"
        )

    if not np.isfinite(flow).all():
        raise OpticalFlowValidationError(
            "Flow contains non-finite values"
        )

    return flow


# ============================================================
# DENSE OPTICAL FLOW
# ============================================================

class DenseOpticalFlow:
    """
    Dense Farneback optical-flow estimator.

    Returns:
        [H,W,2] float32 array.

    Flow represents displacement from
    previous frame to current frame.
    """

    def __init__(
        self,
        config: OpticalFlowConfig | None = None,
    ) -> None:
        self.config = (
            config or OpticalFlowConfig()
        )

    def estimate(
        self,
        previous: np.ndarray,
        current: np.ndarray,
    ) -> np.ndarray:
        previous_gray, current_gray = (
            validate_frame_pair(
                previous,
                current,
            )
        )

        flow = cv2.calcOpticalFlowFarneback(
            previous_gray,
            current_gray,
            None,
            self.config.pyramid_scale,
            self.config.levels,
            self.config.window_size,
            self.config.iterations,
            self.config.polynomial_n,
            self.config.polynomial_sigma,
            0,
        )

        return validate_flow(flow)


# ============================================================
# SPARSE OPTICAL FLOW
# ============================================================

@dataclass
class SparseFlowResult:
    previous_points: np.ndarray
    current_points: np.ndarray

    displacements: np.ndarray
    tracking_errors: np.ndarray

    valid_tracks: int

    @property
    def mean_displacement(self) -> np.ndarray:
        if self.valid_tracks == 0:
            return np.zeros(
                2,
                dtype=np.float32,
            )

        return self.displacements.mean(
            axis=0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "previous_points": (
                self.previous_points.tolist()
            ),
            "current_points": (
                self.current_points.tolist()
            ),
            "displacements": (
                self.displacements.tolist()
            ),
            "tracking_errors": (
                self.tracking_errors.tolist()
            ),
            "valid_tracks": self.valid_tracks,
            "mean_displacement": (
                self.mean_displacement.tolist()
            ),
        }


class SparseOpticalFlow:
    """
    Shi-Tomasi corner detection and
    pyramidal Lucas-Kanade tracking.

    Forward-backward validation rejects
    inconsistent feature tracks.
    """

    def __init__(
        self,
        config: OpticalFlowConfig | None = None,
    ) -> None:
        self.config = (
            config or OpticalFlowConfig()
        )

    def estimate(
        self,
        previous: np.ndarray,
        current: np.ndarray,
    ) -> SparseFlowResult:
        previous_gray, current_gray = (
            validate_frame_pair(
                previous,
                current,
            )
        )

        points = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=self.config.max_corners,
            qualityLevel=self.config.quality_level,
            minDistance=self.config.min_distance,
        )

        if points is None:
            return self._empty_result()

        criteria = (
            cv2.TERM_CRITERIA_EPS
            | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        )

        next_points, status, errors = (
            cv2.calcOpticalFlowPyrLK(
                previous_gray,
                current_gray,
                points,
                None,
                winSize=(
                    self.config.lk_window_size,
                    self.config.lk_window_size,
                ),
                maxLevel=self.config.lk_max_level,
                criteria=criteria,
            )
        )

        if (
            next_points is None
            or status is None
        ):
            return self._empty_result()

        back_points, back_status, _ = (
            cv2.calcOpticalFlowPyrLK(
                current_gray,
                previous_gray,
                next_points,
                None,
                winSize=(
                    self.config.lk_window_size,
                    self.config.lk_window_size,
                ),
                maxLevel=self.config.lk_max_level,
                criteria=criteria,
            )
        )

        if (
            back_points is None
            or back_status is None
        ):
            return self._empty_result()

        original = points.reshape(-1, 2)
        tracked = next_points.reshape(-1, 2)
        returned = back_points.reshape(-1, 2)

        backward_error = np.linalg.norm(
            original - returned,
            axis=1,
        )

        valid = (
            (status.reshape(-1) == 1)
            & (back_status.reshape(-1) == 1)
            & np.isfinite(
                backward_error
            )
            & (
                backward_error
                <= self.config.forward_backward_threshold
            )
            & np.isfinite(
                tracked
            ).all(axis=1)
        )

        original = original[valid]
        tracked = tracked[valid]

        if errors is None:
            tracking_errors = np.zeros(
                len(original),
                dtype=np.float32,
            )
        else:
            tracking_errors = (
                errors.reshape(-1)[valid]
                .astype(np.float32)
            )

        return SparseFlowResult(
            previous_points=original.astype(
                np.float32
            ),
            current_points=tracked.astype(
                np.float32
            ),
            displacements=(
                tracked - original
            ).astype(np.float32),
            tracking_errors=tracking_errors,
            valid_tracks=len(original),
        )

    @staticmethod
    def _empty_result() -> SparseFlowResult:
        empty_points = np.empty(
            (0, 2),
            dtype=np.float32,
        )

        return SparseFlowResult(
            previous_points=empty_points.copy(),
            current_points=empty_points.copy(),
            displacements=empty_points.copy(),
            tracking_errors=np.empty(
                (0,),
                dtype=np.float32,
            ),
            valid_tracks=0,
        )


# ============================================================
# FLOW STATISTICS
# ============================================================

@dataclass
class FlowStatistics:
    mean_magnitude: float
    max_magnitude: float

    median_magnitude: float
    std_magnitude: float

    mean_dx: float
    mean_dy: float

    dominant_direction_radians: float

    moving_pixel_ratio: float

    def to_dict(self) -> dict[str, float]:
        return {
            "mean_magnitude": (
                self.mean_magnitude
            ),
            "max_magnitude": (
                self.max_magnitude
            ),
            "median_magnitude": (
                self.median_magnitude
            ),
            "std_magnitude": (
                self.std_magnitude
            ),
            "mean_dx": self.mean_dx,
            "mean_dy": self.mean_dy,
            "dominant_direction_radians": (
                self.dominant_direction_radians
            ),
            "moving_pixel_ratio": (
                self.moving_pixel_ratio
            ),
        }


class FlowStatisticsExtractor:
    def __init__(
        self,
        motion_threshold: float = 1.5,
    ) -> None:
        if motion_threshold < 0:
            raise ValueError(
                "motion_threshold must be nonnegative"
            )

        self.motion_threshold = motion_threshold

    def extract(
        self,
        flow: np.ndarray,
    ) -> FlowStatistics:
        flow = validate_flow(flow)

        magnitude = np.linalg.norm(
            flow,
            axis=-1,
        )

        mean_dx = float(
            flow[..., 0].mean()
        )

        mean_dy = float(
            flow[..., 1].mean()
        )

        return FlowStatistics(
            mean_magnitude=float(
                magnitude.mean()
            ),
            max_magnitude=float(
                magnitude.max()
            ),
            median_magnitude=float(
                np.median(magnitude)
            ),
            std_magnitude=float(
                magnitude.std()
            ),
            mean_dx=mean_dx,
            mean_dy=mean_dy,
            dominant_direction_radians=float(
                math.atan2(
                    mean_dy,
                    mean_dx,
                )
            ),
            moving_pixel_ratio=float(
                np.mean(
                    magnitude
                    >= self.motion_threshold
                )
            ),
        )


# ============================================================
# MOTION REGION DETECTION
# ============================================================

@dataclass
class MotionRegion:
    x: int
    y: int

    width: int
    height: int

    area: float

    mean_magnitude: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "area": self.area,
            "mean_magnitude": (
                self.mean_magnitude
            ),
        }


class MotionRegionDetector:
    """
    Detects regions with significant
    optical-flow magnitude.

    Note:
        Camera motion can cause large
        portions of the frame to appear
        as moving regions.
    """

    def __init__(
        self,
        config: OpticalFlowConfig,
    ) -> None:
        self.config = config

    def detect(
        self,
        flow: np.ndarray,
    ) -> tuple[
        np.ndarray,
        list[MotionRegion],
    ]:
        flow = validate_flow(flow)

        magnitude = np.linalg.norm(
            flow,
            axis=-1,
        )

        mask = (
            magnitude
            >= self.config.motion_threshold
        ).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        regions = []

        for contour in contours:
            area = float(
                cv2.contourArea(
                    contour
                )
            )

            if (
                area
                < self.config.minimum_region_area
            ):
                continue

            x, y, width, height = (
                cv2.boundingRect(
                    contour
                )
            )

            region_magnitude = magnitude[
                y:y + height,
                x:x + width,
            ]

            regions.append(
                MotionRegion(
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    area=area,
                    mean_magnitude=float(
                        region_magnitude.mean()
                    ),
                )
            )

        regions.sort(
            key=lambda item: item.area,
            reverse=True,
        )

        return mask, regions


# ============================================================
# OPTICAL FLOW VISUALIZATION
# ============================================================

class FlowVisualizer:
    """
    Converts optical flow to visual images.

    Hue:
        Motion direction

    Brightness:
        Motion magnitude
    """

    def __init__(
        self,
        max_magnitude: float = 20.0,
    ) -> None:
        if max_magnitude <= 0:
            raise ValueError(
                "max_magnitude must be positive"
            )

        self.max_magnitude = max_magnitude

    def to_color(
        self,
        flow: np.ndarray,
    ) -> np.ndarray:
        flow = validate_flow(flow)

        magnitude, angle = cv2.cartToPolar(
            flow[..., 0],
            flow[..., 1],
            angleInDegrees=True,
        )

        hsv = np.zeros(
            (
                flow.shape[0],
                flow.shape[1],
                3,
            ),
            dtype=np.uint8,
        )

        hsv[..., 0] = (
            angle / 2.0
        ).astype(np.uint8)

        hsv[..., 1] = 255

        hsv[..., 2] = np.clip(
            magnitude
            / self.max_magnitude
            * 255.0,
            0,
            255,
        ).astype(np.uint8)

        return cv2.cvtColor(
            hsv,
            cv2.COLOR_HSV2BGR,
        )

    def magnitude_heatmap(
        self,
        flow: np.ndarray,
    ) -> np.ndarray:
        flow = validate_flow(flow)

        magnitude = np.linalg.norm(
            flow,
            axis=-1,
        )

        normalized = np.clip(
            magnitude
            / self.max_magnitude
            * 255.0,
            0,
            255,
        ).astype(np.uint8)

        return cv2.applyColorMap(
            normalized,
            cv2.COLORMAP_TURBO,
        )

    @staticmethod
    def draw_regions(
        frame: np.ndarray,
        regions: list[MotionRegion],
    ) -> np.ndarray:
        validate_frame(frame)

        if frame.ndim == 2:
            output = cv2.cvtColor(
                frame,
                cv2.COLOR_GRAY2BGR,
            )
        elif frame.shape[2] == 4:
            output = cv2.cvtColor(
                frame,
                cv2.COLOR_BGRA2BGR,
            )
        else:
            output = frame.copy()

        for region in regions:
            cv2.rectangle(
                output,
                (
                    region.x,
                    region.y,
                ),
                (
                    region.x + region.width,
                    region.y + region.height,
                ),
                (0, 255, 0),
                2,
            )

        return output


# ============================================================
# FRAME WARPING
# ============================================================

class OpticalFlowWarper:
    """
    Backward image warping.

    For flow defined from frame A to B,
    use approximately -flow to sample
    frame A into frame B coordinates.

    This approximation is most reliable
    for small, smooth displacements.
    """

    @staticmethod
    def warp(
        frame: np.ndarray,
        backward_flow: np.ndarray,
    ) -> np.ndarray:
        validate_frame(frame)

        backward_flow = validate_flow(
            backward_flow
        )

        height, width = frame.shape[:2]

        if backward_flow.shape[:2] != (
            height,
            width,
        ):
            raise OpticalFlowValidationError(
                "Flow and frame sizes mismatch"
            )

        grid_x, grid_y = np.meshgrid(
            np.arange(width),
            np.arange(height),
        )

        map_x = (
            grid_x.astype(np.float32)
            + backward_flow[..., 0]
        )

        map_y = (
            grid_y.astype(np.float32)
            + backward_flow[..., 1]
        )

        return cv2.remap(
            frame,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )


# ============================================================
# MOTION ANALYSIS RESULT
# ============================================================

@dataclass
class OpticalFlowResult:
    flow: np.ndarray

    statistics: FlowStatistics

    motion_mask: np.ndarray

    regions: list[MotionRegion]

    sparse_tracks: SparseFlowResult | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def summary(self) -> dict[str, Any]:
        return {
            "flow_shape": list(
                self.flow.shape
            ),
            "statistics": (
                self.statistics.to_dict()
            ),
            "motion_regions": [
                region.to_dict()
                for region in self.regions
            ],
            "sparse_tracks": (
                self.sparse_tracks.valid_tracks
                if self.sparse_tracks is not None
                else None
            ),
            "metadata": self.metadata,
        }


# ============================================================
# OPTICAL FLOW ENGINE
# ============================================================

class OpticalFlowEngine:
    """
    High-level frame-pair analysis.

    Combines:
        Dense flow
        Sparse tracking
        Motion statistics
        Motion-region detection
        Flow visualization
    """

    def __init__(
        self,
        config: OpticalFlowConfig | None = None,
    ) -> None:
        self.config = (
            config or OpticalFlowConfig()
        )

        self.dense = DenseOpticalFlow(
            self.config
        )

        self.sparse = SparseOpticalFlow(
            self.config
        )

        self.statistics = FlowStatisticsExtractor(
            self.config.motion_threshold
        )

        self.regions = MotionRegionDetector(
            self.config
        )

        self.visualizer = FlowVisualizer(
            self.config.visualization_max_magnitude
        )

    def analyze(
        self,
        previous: np.ndarray,
        current: np.ndarray,
        *,
        include_sparse: bool = True,
    ) -> OpticalFlowResult:
        flow = self.dense.estimate(
            previous,
            current,
        )

        statistics = self.statistics.extract(
            flow
        )

        mask, regions = self.regions.detect(
            flow
        )

        sparse_tracks = None

        if include_sparse:
            sparse_tracks = (
                self.sparse.estimate(
                    previous,
                    current,
                )
            )

        return OpticalFlowResult(
            flow=flow,
            statistics=statistics,
            motion_mask=mask,
            regions=regions,
            sparse_tracks=sparse_tracks,
            metadata={
                "method": "farneback",
                "height": flow.shape[0],
                "width": flow.shape[1],
            },
        )


# ============================================================
# TORCH FLOW FEATURE ENCODER
# ============================================================

class OpticalFlowFeatureEncoder(nn.Module):
    """
    CNN encoder for optical-flow tensors.

    Input:
        [B,2,H,W]

    Output:
        [B,embedding_dim]

    Can feed motion features into:
        - motion_encoder.py
        - video_encoder.py
        - world_model.py
        - trajectory_prediction.py
    """

    def __init__(
        self,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                2,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            ),
            nn.Flatten(),
        )

        self.projection = nn.Sequential(
            nn.Linear(
                128,
                embedding_dim,
            ),
            nn.LayerNorm(
                embedding_dim
            ),
        )

    def forward(
        self,
        flow: Tensor,
    ) -> Tensor:
        if (
            flow.ndim != 4
            or flow.shape[1] != 2
        ):
            raise OpticalFlowValidationError(
                "Flow must be [B,2,H,W]"
            )

        if not torch.isfinite(
            flow
        ).all():
            raise OpticalFlowValidationError(
                "Flow contains non-finite values"
            )

        features = self.encoder(
            flow
        )

        return self.projection(
            features
        )


# ============================================================
# NUMPY TO TORCH ADAPTER
# ============================================================

class OpticalFlowTensorAdapter:
    """
    Converts OpenCV optical flow:

        [H,W,2]

    Into PyTorch:

        [1,2,H,W]
    """

    @staticmethod
    def to_tensor(
        flow: np.ndarray,
        *,
        device: str = "cpu",
        normalize_by: float | None = None,
    ) -> Tensor:
        flow = validate_flow(
            flow
        )

        tensor = torch.from_numpy(
            flow.copy()
        ).permute(
            2,
            0,
            1,
        ).unsqueeze(0)

        if normalize_by is not None:
            if normalize_by <= 0:
                raise ValueError(
                    "normalize_by must be positive"
                )

            tensor = (
                tensor / normalize_by
            )

        return tensor.to(
            device=device,
            dtype=torch.float32,
        )


# ============================================================
# VIDEO OPTICAL FLOW PROCESSOR
# ============================================================

class VideoOpticalFlowProcessor:
    """
    Processes video frame pairs.

    Each result describes apparent motion
    between consecutive decoded frames.

    For real-time speed estimation, use
    frame timestamps and camera calibration.
    """

    def __init__(
        self,
        engine: OpticalFlowEngine | None = None,
    ) -> None:
        self.engine = (
            engine or OpticalFlowEngine()
        )

    def process_video(
        self,
        video_path: str | Path,
        *,
        max_pairs: int | None = None,
        include_sparse: bool = False,
    ) -> Iterator[OpticalFlowResult]:
        path = Path(
            video_path
        )

        if not path.is_file():
            raise FileNotFoundError(
                str(path)
            )

        if (
            max_pairs is not None
            and max_pairs <= 0
        ):
            raise ValueError(
                "max_pairs must be positive"
            )

        capture = cv2.VideoCapture(
            str(path)
        )

        if not capture.isOpened():
            raise OpticalFlowError(
                f"Cannot open video: {path}"
            )

        try:
            success, previous = (
                capture.read()
            )

            if not success:
                return

            count = 0

            while True:
                if (
                    max_pairs is not None
                    and count >= max_pairs
                ):
                    break

                success, current = (
                    capture.read()
                )

                if not success:
                    break

                result = self.engine.analyze(
                    previous,
                    current,
                    include_sparse=include_sparse,
                )

                result.metadata[
                    "frame_pair_index"
                ] = count

                yield result

                previous = current
                count += 1

        finally:
            capture.release()


# ============================================================
# SYNTHETIC TEST FRAMES
# ============================================================

def create_synthetic_frames(
    width: int = 320,
    height: int = 240,
    dx: int = 12,
    dy: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates a moving textured rectangle
    for a reproducible optical-flow demo.
    """

    previous = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    current = previous.copy()

    x1, y1 = 60, 70
    x2, y2 = 130, 140

    cv2.rectangle(
        previous,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        -1,
    )

    cv2.rectangle(
        current,
        (x1 + dx, y1 + dy),
        (x2 + dx, y2 + dy),
        (255, 255, 255),
        -1,
    )

    for offset in range(10, 60, 12):
        cv2.circle(
            previous,
            (x1 + offset, y1 + offset),
            3,
            (0, 0, 0),
            -1,
        )

        cv2.circle(
            current,
            (
                x1 + offset + dx,
                y1 + offset + dy,
            ),
            3,
            (0, 0, 0),
            -1,
        )

    return previous, current


# ============================================================
# FACTORY
# ============================================================

def create_optical_flow_engine(
    config: OpticalFlowConfig | None = None,
) -> OpticalFlowEngine:
    return OpticalFlowEngine(
        config
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print("=" * 65)
    print(
        "NEUROWORLD AI - OPTICAL FLOW"
    )
    print("=" * 65)

    config = OpticalFlowConfig(
        motion_threshold=1.0,
        minimum_region_area=50,
        embedding_dim=128,
    )

    engine = OpticalFlowEngine(
        config
    )

    previous, current = (
        create_synthetic_frames()
    )

    result = engine.analyze(
        previous,
        current,
    )

    print("\nMotion statistics:")
    print(
        result.statistics.to_dict()
    )

    print(
        "\nMotion regions:",
        len(result.regions),
    )

    print(
        "Sparse tracks:",
        result.sparse_tracks.valid_tracks
        if result.sparse_tracks
        else 0,
    )

    print(
        "Flow shape:",
        result.flow.shape,
    )

    # Visualization
    color_flow = (
        engine.visualizer.to_color(
            result.flow
        )
    )

    heatmap = (
        engine.visualizer.magnitude_heatmap(
            result.flow
        )
    )

    annotated = (
        engine.visualizer.draw_regions(
            current,
            result.regions,
        )
    )

    output_dir = Path(
        "outputs/optical_flow"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(
            output_dir / "flow_color.png"
        ),
        color_flow,
    )

    cv2.imwrite(
        str(
            output_dir / "flow_heatmap.png"
        ),
        heatmap,
    )

    cv2.imwrite(
        str(
            output_dir / "motion_regions.png"
        ),
        annotated,
    )

    # Neural feature encoding
    flow_tensor = (
        OpticalFlowTensorAdapter.to_tensor(
            result.flow,
            normalize_by=20.0,
        )
    )

    encoder = OpticalFlowFeatureEncoder(
        embedding_dim=128
    )

    encoder.eval()

    with torch.no_grad():
        embedding = encoder(
            flow_tensor
        )

    print(
        "\nFlow tensor:",
        tuple(flow_tensor.shape),
    )

    print(
        "Motion embedding:",
        tuple(embedding.shape),
    )

    print(
        "\nSaved visualization files to:",
        output_dir,
    )

    print(
        "\nOptical flow demonstration completed"
    )


if __name__ == "__main__":
    main()

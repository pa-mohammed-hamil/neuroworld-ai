
"""
NeuroWorld AI
File: neuroworld/perception/depth_estimator.py

Monocular Depth Estimation & 3D Scene Perception

Features:
    - RGB image depth estimation
    - Hugging Face pretrained depth models
    - Depth Anything / DPT-compatible models
    - CPU and CUDA inference
    - Image and NumPy input
    - Depth map normalization
    - Depth map color visualization
    - Camera intrinsic parameters
    - 3D point cloud reconstruction
    - Region-based depth analysis
    - Depth uncertainty estimation
    - Obstacle proximity analysis
    - Batch image inference
    - Optional FastAPI integration
    - Synthetic fallback for unit tests

Important:
    Monocular models may output relative depth.
    Relative depth is not a reliable metric
    distance in meters without calibration.

Install:
    pip install torch transformers pillow numpy opencv-python

Run:
    python -m neuroworld.perception.depth_estimator
"""

from __future__ import annotations

import argparse
import math

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from PIL import Image


# ============================================================
# EXCEPTIONS
# ============================================================

class DepthEstimatorError(Exception):
    """Base depth-estimation error."""


class DepthValidationError(DepthEstimatorError):
    """Invalid image or camera configuration."""


class DepthModelLoadError(DepthEstimatorError):
    """Pretrained model loading failure."""


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class DepthEstimatorConfig:
    model_name: str = (
        "Intel/dpt-hybrid-midas"
    )

    device: str = "auto"

    min_depth: float = 1e-6

    max_visualization_depth: float = 100.0

    confidence_threshold: float = 0.5

    use_half_precision: bool = False

    trust_remote_code: bool = False

    local_files_only: bool = False

    def __post_init__(self) -> None:
        if self.min_depth <= 0:
            raise ValueError(
                "min_depth must be positive"
            )

        if self.max_visualization_depth <= 0:
            raise ValueError(
                "max_visualization_depth must be positive"
            )

        if not (
            0 <= self.confidence_threshold <= 1
        ):
            raise ValueError(
                "confidence_threshold must be in [0,1]"
            )


# ============================================================
# CAMERA INTRINSICS
# ============================================================

@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float

    cx: float
    cy: float

    width: int
    height: int

    def __post_init__(self) -> None:
        if (
            self.fx <= 0
            or self.fy <= 0
        ):
            raise DepthValidationError(
                "Focal lengths must be positive"
            )

        if (
            self.width <= 0
            or self.height <= 0
        ):
            raise DepthValidationError(
                "Invalid camera dimensions"
            )

        if not all(
            math.isfinite(value)
            for value in (
                self.fx,
                self.fy,
                self.cx,
                self.cy,
            )
        ):
            raise DepthValidationError(
                "Camera intrinsics must be finite"
            )

    @classmethod
    def from_fov(
        cls,
        width: int,
        height: int,
        horizontal_fov_degrees: float,
    ) -> CameraIntrinsics:
        if not (
            0 < horizontal_fov_degrees < 180
        ):
            raise DepthValidationError(
                "FOV must be between 0 and 180 degrees"
            )

        focal = (
            width / 2.0
        ) / math.tan(
            math.radians(
                horizontal_fov_degrees
            ) / 2.0
        )

        return cls(
            fx=focal,
            fy=focal,
            cx=(width - 1) / 2.0,
            cy=(height - 1) / 2.0,
            width=width,
            height=height,
        )


# ============================================================
# DEPTH RESULT
# ============================================================

@dataclass
class DepthEstimationResult:
    depth_map: np.ndarray

    width: int
    height: int

    is_metric: bool

    model_name: str

    confidence_map: np.ndarray | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.depth_map = np.asarray(
            self.depth_map,
            dtype=np.float32,
        )

        if self.depth_map.shape != (
            self.height,
            self.width,
        ):
            raise DepthValidationError(
                "Depth map shape mismatch"
            )

        if not np.isfinite(
            self.depth_map
        ).all():
            raise DepthValidationError(
                "Depth map contains non-finite values"
            )

        if self.confidence_map is not None:
            self.confidence_map = np.asarray(
                self.confidence_map,
                dtype=np.float32,
            )

            if (
                self.confidence_map.shape
                != self.depth_map.shape
            ):
                raise DepthValidationError(
                    "Confidence map shape mismatch"
                )

    @property
    def minimum_depth(self) -> float:
        return float(
            self.depth_map.min()
        )

    @property
    def maximum_depth(self) -> float:
        return float(
            self.depth_map.max()
        )

    @property
    def mean_depth(self) -> float:
        return float(
            self.depth_map.mean()
        )

    def to_dict(
        self,
        include_depth: bool = False,
    ) -> dict[str, Any]:
        output = {
            "width": self.width,
            "height": self.height,
            "is_metric": self.is_metric,
            "model_name": self.model_name,
            "minimum_depth": self.minimum_depth,
            "maximum_depth": self.maximum_depth,
            "mean_depth": self.mean_depth,
            "metadata": self.metadata,
        }

        if include_depth:
            output["depth_map"] = (
                self.depth_map.tolist()
            )

        return output


# ============================================================
# IMAGE UTILITIES
# ============================================================

class DepthImageProcessor:
    """
    Converts RGB inputs to PIL images.

    NumPy images must be RGB, not BGR.
    """

    @staticmethod
    def load_image(
        image: str | Path | Image.Image | np.ndarray,
    ) -> Image.Image:
        if isinstance(
            image,
            (str, Path),
        ):
            path = Path(image)

            if not path.is_file():
                raise FileNotFoundError(
                    str(path)
                )

            with Image.open(path) as source:
                return source.convert("RGB")

        if isinstance(
            image,
            Image.Image,
        ):
            return image.convert("RGB")

        if isinstance(
            image,
            np.ndarray,
        ):
            if (
                image.ndim != 3
                or image.shape[2] != 3
            ):
                raise DepthValidationError(
                    "NumPy image must be [H,W,3]"
                )

            if image.dtype != np.uint8:
                raise DepthValidationError(
                    "NumPy RGB image must be uint8"
                )

            return Image.fromarray(
                image,
                mode="RGB",
            )

        raise TypeError(
            "Unsupported image type"
        )

    @staticmethod
    def normalize_depth(
        depth: np.ndarray,
        *,
        invert: bool = False,
    ) -> np.ndarray:
        depth = np.asarray(
            depth,
            dtype=np.float32,
        )

        finite = np.isfinite(
            depth
        )

        output = np.zeros_like(
            depth
        )

        if not finite.any():
            return output

        values = depth[finite]

        minimum = float(
            values.min()
        )

        maximum = float(
            values.max()
        )

        if maximum - minimum > 1e-8:
            output[finite] = (
                depth[finite] - minimum
            ) / (
                maximum - minimum
            )

        if invert:
            output[finite] = (
                1.0 - output[finite]
            )

        return output

    @staticmethod
    def colorize(
        depth: np.ndarray,
        *,
        invert: bool = False,
    ) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "Install opencv-python"
            ) from exc

        normalized = (
            DepthImageProcessor.normalize_depth(
                depth,
                invert=invert,
            )
        )

        grayscale = (
            normalized * 255.0
        ).astype(
            np.uint8
        )

        colored_bgr = cv2.applyColorMap(
            grayscale,
            cv2.COLORMAP_TURBO,
        )

        return cv2.cvtColor(
            colored_bgr,
            cv2.COLOR_BGR2RGB,
        )

    @staticmethod
    def save_visualization(
        depth: np.ndarray,
        output_path: str | Path,
        *,
        invert: bool = False,
    ) -> None:
        colored = (
            DepthImageProcessor.colorize(
                depth,
                invert=invert,
            )
        )

        Image.fromarray(
            colored
        ).save(
            output_path
        )


# ============================================================
# DEPTH BACKEND PROTOCOL
# ============================================================

class DepthBackend(Protocol):
    model_name: str
    is_metric: bool

    def predict(
        self,
        image: Image.Image,
    ) -> np.ndarray:
        ...


# ============================================================
# HUGGING FACE DEPTH BACKEND
# ============================================================

class HuggingFaceDepthBackend:
    """
    Loads a pretrained depth-estimation model.

    Compatible with supported Transformers
    AutoImageProcessor and
    AutoModelForDepthEstimation checkpoints.

    Important:
        Different models may produce
        relative depth, inverse depth,
        or metric depth.

        This backend defaults to
        is_metric=False.
    """

    def __init__(
        self,
        config: DepthEstimatorConfig,
        *,
        is_metric: bool = False,
    ) -> None:
        try:
            from transformers import (
                AutoImageProcessor,
                AutoModelForDepthEstimation,
            )
        except ImportError as exc:
            raise ImportError(
                "Install transformers"
            ) from exc

        self.config = config
        self.model_name = config.model_name
        self.is_metric = is_metric

        if config.device == "auto":
            device_name = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            device_name = config.device

        self.device = torch.device(
            device_name
        )

        try:
            self.processor = (
                AutoImageProcessor.from_pretrained(
                    config.model_name,
                    trust_remote_code=(
                        config.trust_remote_code
                    ),
                    local_files_only=(
                        config.local_files_only
                    ),
                )
            )

            self.model = (
                AutoModelForDepthEstimation.from_pretrained(
                    config.model_name,
                    trust_remote_code=(
                        config.trust_remote_code
                    ),
                    local_files_only=(
                        config.local_files_only
                    ),
                )
            )

        except Exception as exc:
            raise DepthModelLoadError(
                f"Failed to load {config.model_name}: "
                f"{exc}"
            ) from exc

        self.model.to(
            self.device
        )

        if (
            config.use_half_precision
            and self.device.type == "cuda"
        ):
            self.model.half()

        self.model.eval()

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image,
    ) -> np.ndarray:
        inputs = self.processor(
            images=image,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(
                self.device
            )
            for key, value in inputs.items()
        }

        if (
            self.config.use_half_precision
            and self.device.type == "cuda"
        ):
            inputs = {
                key: (
                    value.half()
                    if value.is_floating_point()
                    else value
                )
                for key, value in inputs.items()
            }

        outputs = self.model(
            **inputs
        )

        predicted_depth = (
            outputs.predicted_depth
        )

        if predicted_depth.ndim == 3:
            predicted_depth = (
                predicted_depth.unsqueeze(1)
            )

        resized = (
            torch.nn.functional.interpolate(
                predicted_depth,
                size=(
                    image.height,
                    image.width,
                ),
                mode="bicubic",
                align_corners=False,
            )
        )

        depth = (
            resized[0, 0]
            .float()
            .cpu()
            .numpy()
        )

        if not np.isfinite(depth).all():
            raise DepthEstimatorError(
                "Model produced non-finite depth"
            )

        return depth.astype(
            np.float32
        )


# ============================================================
# SYNTHETIC DEPTH BACKEND
# ============================================================

class SyntheticDepthBackend:
    """
    Synthetic depth generator for testing.

    This does not infer real scene geometry.

    Produces a simple vertical gradient.
    """

    model_name = "synthetic-depth"
    is_metric = False

    def predict(
        self,
        image: Image.Image,
    ) -> np.ndarray:
        height = image.height
        width = image.width

        gradient = np.linspace(
            1.0,
            0.1,
            height,
            dtype=np.float32,
        )

        return np.broadcast_to(
            gradient[:, None],
            (height, width),
        ).copy()


# ============================================================
# DEPTH UNCERTAINTY
# ============================================================

class DepthUncertaintyEstimator:
    """
    Estimate uncertainty from repeated
    predictions or an ensemble.

    Input:
        depth_maps [K,H,W]

    Output:
        standard deviation [H,W]

    Repeated deterministic predictions
    from the same model usually provide
    no meaningful epistemic uncertainty.
    """

    @staticmethod
    def estimate(
        depth_maps: np.ndarray,
    ) -> np.ndarray:
        depth_maps = np.asarray(
            depth_maps,
            dtype=np.float32,
        )

        if (
            depth_maps.ndim != 3
            or depth_maps.shape[0] < 2
        ):
            raise DepthValidationError(
                "Expected at least two depth maps"
            )

        if not np.isfinite(
            depth_maps
        ).all():
            raise DepthValidationError(
                "Depth maps must be finite"
            )

        return np.std(
            depth_maps,
            axis=0,
        ).astype(
            np.float32
        )


# ============================================================
# DEPTH REGION ANALYSIS
# ============================================================

@dataclass
class DepthRegionStatistics:
    minimum: float
    maximum: float
    mean: float
    median: float
    standard_deviation: float

    pixel_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "median": self.median,
            "standard_deviation": (
                self.standard_deviation
            ),
            "pixel_count": self.pixel_count,
        }


class DepthRegionAnalyzer:
    @staticmethod
    def analyze(
        depth_map: np.ndarray,
        bounding_box: tuple[
            int,
            int,
            int,
            int,
        ],
    ) -> DepthRegionStatistics:
        depth = np.asarray(
            depth_map,
            dtype=np.float32,
        )

        if depth.ndim != 2:
            raise DepthValidationError(
                "Depth map must be 2D"
            )

        x1, y1, x2, y2 = (
            bounding_box
        )

        height, width = depth.shape

        x1 = max(
            0,
            min(x1, width),
        )

        x2 = max(
            0,
            min(x2, width),
        )

        y1 = max(
            0,
            min(y1, height),
        )

        y2 = max(
            0,
            min(y2, height),
        )

        if x2 <= x1 or y2 <= y1:
            raise DepthValidationError(
                "Empty depth region"
            )

        region = depth[
            y1:y2,
            x1:x2,
        ]

        values = region[
            np.isfinite(region)
        ]

        if values.size == 0:
            raise DepthValidationError(
                "No valid depth pixels"
            )

        return DepthRegionStatistics(
            minimum=float(
                values.min()
            ),
            maximum=float(
                values.max()
            ),
            mean=float(
                values.mean()
            ),
            median=float(
                np.median(values)
            ),
            standard_deviation=float(
                values.std()
            ),
            pixel_count=int(
                values.size
            ),
        )


# ============================================================
# POINT CLOUD RECONSTRUCTION
# ============================================================

class DepthPointCloudReconstructor:
    """
    Back-projects metric depth into 3D.

    Pinhole camera model:

        X = (u-cx)*Z/fx
        Y = (v-cy)*Z/fy
        Z = depth

    Coordinate convention:
        +X right
        +Y down
        +Z forward

    Requires depth in meters.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
    ) -> None:
        self.intrinsics = intrinsics

    def reconstruct(
        self,
        depth_map: np.ndarray,
        *,
        stride: int = 1,
        min_depth: float = 0.01,
        max_depth: float = 100.0,
    ) -> np.ndarray:
        if stride <= 0:
            raise DepthValidationError(
                "stride must be positive"
            )

        depth = np.asarray(
            depth_map,
            dtype=np.float32,
        )

        if depth.shape != (
            self.intrinsics.height,
            self.intrinsics.width,
        ):
            raise DepthValidationError(
                "Depth dimensions mismatch"
            )

        ys, xs = np.mgrid[
            0:self.intrinsics.height:stride,
            0:self.intrinsics.width:stride,
        ]

        sampled_depth = depth[
            ::stride,
            ::stride,
        ]

        valid = (
            np.isfinite(sampled_depth)
            & (sampled_depth >= min_depth)
            & (sampled_depth <= max_depth)
        )

        z = sampled_depth[valid]

        x = (
            xs[valid]
            - self.intrinsics.cx
        ) * z / self.intrinsics.fx

        y = (
            ys[valid]
            - self.intrinsics.cy
        ) * z / self.intrinsics.fy

        return np.stack(
            [x, y, z],
            axis=-1,
        ).astype(
            np.float32
        )


# ============================================================
# OBSTACLE PROXIMITY ANALYZER
# ============================================================

@dataclass
class DepthProximityResult:
    nearest_depth: float
    median_depth: float

    threshold: float

    close_pixel_fraction: float

    potential_obstacle: bool

    unit: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "nearest_depth": (
                self.nearest_depth
            ),
            "median_depth": (
                self.median_depth
            ),
            "threshold": self.threshold,
            "close_pixel_fraction": (
                self.close_pixel_fraction
            ),
            "potential_obstacle": (
                self.potential_obstacle
            ),
            "unit": self.unit,
        }


class DepthProximityAnalyzer:
    """
    Evaluates nearby depth pixels.

    Requires calibrated metric depth.

    This is a perception heuristic,
    not a certified collision detector.
    """

    def __init__(
        self,
        distance_threshold: float = 1.0,
        minimum_close_fraction: float = 0.05,
    ) -> None:
        if distance_threshold <= 0:
            raise ValueError(
                "distance_threshold must be positive"
            )

        if not (
            0 <= minimum_close_fraction <= 1
        ):
            raise ValueError(
                "Invalid close fraction"
            )

        self.distance_threshold = (
            distance_threshold
        )

        self.minimum_close_fraction = (
            minimum_close_fraction
        )

    def analyze(
        self,
        result: DepthEstimationResult,
    ) -> DepthProximityResult:
        if not result.is_metric:
            raise DepthValidationError(
                "Metric depth is required for "
                "distance-based obstacle analysis"
            )

        depth = result.depth_map

        valid = (
            np.isfinite(depth)
            & (depth > 0)
        )

        values = depth[valid]

        if values.size == 0:
            raise DepthValidationError(
                "No valid depth values"
            )

        close_fraction = float(
            np.mean(
                values
                < self.distance_threshold
            )
        )

        return DepthProximityResult(
            nearest_depth=float(
                values.min()
            ),
            median_depth=float(
                np.median(values)
            ),
            threshold=(
                self.distance_threshold
            ),
            close_pixel_fraction=(
                close_fraction
            ),
            potential_obstacle=(
                close_fraction
                >= self.minimum_close_fraction
            ),
            unit="meters",
        )


# ============================================================
# MAIN DEPTH ESTIMATOR
# ============================================================

class DepthEstimator:
    """
    High-level depth estimation interface.

    Default:
        Hugging Face pretrained model

    Optional:
        SyntheticDepthBackend for testing

    Note:
        is_metric must only be True when
        the selected model is known to
        output calibrated metric depth.
    """

    def __init__(
        self,
        config: DepthEstimatorConfig | None = None,
        *,
        backend: DepthBackend | None = None,
        is_metric: bool = False,
    ) -> None:
        self.config = (
            config or DepthEstimatorConfig()
        )

        if backend is not None:
            self.backend = backend

        else:
            self.backend = (
                HuggingFaceDepthBackend(
                    self.config,
                    is_metric=is_metric,
                )
            )

    def estimate(
        self,
        image: (
            str
            | Path
            | Image.Image
            | np.ndarray
        ),
    ) -> DepthEstimationResult:
        pil_image = (
            DepthImageProcessor.load_image(
                image
            )
        )

        depth = self.backend.predict(
            pil_image
        )

        depth = np.asarray(
            depth,
            dtype=np.float32,
        )

        if depth.shape != (
            pil_image.height,
            pil_image.width,
        ):
            raise DepthEstimatorError(
                "Backend returned incorrect depth size"
            )

        return DepthEstimationResult(
            depth_map=depth,
            width=pil_image.width,
            height=pil_image.height,
            is_metric=self.backend.is_metric,
            model_name=self.backend.model_name,
            metadata={
                "input_size": [
                    pil_image.width,
                    pil_image.height,
                ],
                "depth_type": (
                    "metric"
                    if self.backend.is_metric
                    else "relative_or_model_specific"
                ),
            },
        )

    def estimate_batch(
        self,
        images: list[
            str
            | Path
            | Image.Image
            | np.ndarray
        ],
    ) -> list[DepthEstimationResult]:
        return [
            self.estimate(image)
            for image in images
        ]

    def save_depth_map(
        self,
        result: DepthEstimationResult,
        output_path: str | Path,
    ) -> None:
        DepthImageProcessor.save_visualization(
            result.depth_map,
            output_path,
        )


# ============================================================
# FACTORY
# ============================================================

def create_depth_estimator(
    model_name: str = (
        "Intel/dpt-hybrid-midas"
    ),
    device: str = "auto",
) -> DepthEstimator:
    config = DepthEstimatorConfig(
        model_name=model_name,
        device=device,
    )

    return DepthEstimator(
        config
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "NeuroWorld AI Depth Estimator"
        )
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="depth_output.png",
    )

    parser.add_argument(
        "--synthetic",
        action="store_true",
        help=(
            "Use synthetic test backend "
            "instead of pretrained model"
        ),
    )

    args = parser.parse_args()

    if args.synthetic:
        estimator = DepthEstimator(
            backend=SyntheticDepthBackend()
        )

        image = Image.new(
            "RGB",
            (320, 240),
            color=(120, 160, 200),
        )

    else:
        if args.image is None:
            parser.error(
                "--image is required unless "
                "--synthetic is specified"
            )

        estimator = (
            create_depth_estimator()
        )

        image = args.image

    result = estimator.estimate(
        image
    )

    estimator.save_depth_map(
        result,
        args.output,
    )

    print("=" * 60)
    print("NEUROWORLD AI - DEPTH ESTIMATOR")
    print("=" * 60)

    print(
        "Model:",
        result.model_name,
    )

    print(
        "Depth shape:",
        result.depth_map.shape,
    )

    print(
        "Metric depth:",
        result.is_metric,
    )

    print(
        "Depth range:",
        result.minimum_depth,
        result.maximum_depth,
    )

    print(
        "Saved visualization:",
        args.output,
    )

    print(
        "\nDepth estimation completed"
    )


if __name__ == "__main__":
    main()

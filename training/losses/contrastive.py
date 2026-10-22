
"""
NeuroWorld AI
File: training/losses/contrastive.py

Contrastive learning losses for:
- World-model representation learning
- Video and image embeddings
- Robot state-action embeddings
- Multimodal representation alignment
- Self-supervised learning
- Teacher-student representation learning

Features:
- InfoNCE / NT-Xent loss
- Symmetric contrastive loss
- Supervised contrastive loss
- Triplet margin loss
- Cosine embedding loss
- BYOL-style consistency loss
- VICReg regularization
- Hard-negative mining
- Optional memory queue
- Distributed negative gathering
- Configurable temperature
- L2 normalization
- Loss component logging

Requires:
    torch >= 2.1
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.distributed as dist
import torch.nn.functional as F

from torch import Tensor, nn


@dataclass
class ContrastiveLossConfig:
    loss_type: str = "infonce"

    temperature: float = 0.07
    normalize_embeddings: bool = True

    symmetric: bool = True
    label_smoothing: float = 0.0

    margin: float = 0.2

    hard_negative_top_k: int | None = None

    use_memory_queue: bool = False
    queue_size: int = 4096

    distributed_negatives: bool = False

    vicreg_similarity_weight: float = 25.0
    vicreg_variance_weight: float = 25.0
    vicreg_covariance_weight: float = 1.0

    variance_target: float = 1.0
    variance_epsilon: float = 1e-4

    detect_nonfinite: bool = True

    def __post_init__(self) -> None:
        supported = {
            "infonce",
            "nt_xent",
            "supervised",
            "triplet",
            "cosine",
            "byol",
            "vicreg",
        }

        if self.loss_type not in supported:
            raise ValueError(
                f"Unsupported loss type: {self.loss_type}"
            )

        if (
            not math.isfinite(self.temperature)
            or self.temperature <= 0
        ):
            raise ValueError(
                "temperature must be positive"
            )

        if not 0 <= self.label_smoothing < 1:
            raise ValueError(
                "label_smoothing must be in [0, 1)"
            )

        if self.margin < 0:
            raise ValueError(
                "margin must be nonnegative"
            )

        if (
            self.hard_negative_top_k is not None
            and self.hard_negative_top_k < 1
        ):
            raise ValueError(
                "hard_negative_top_k must be >= 1"
            )

        if self.queue_size < 1:
            raise ValueError(
                "queue_size must be >= 1"
            )

        if self.variance_target <= 0:
            raise ValueError(
                "variance_target must be positive"
            )

        if self.variance_epsilon <= 0:
            raise ValueError(
                "variance_epsilon must be positive"
            )


class ContrastiveLoss(nn.Module):
    """
    Configurable contrastive learning module.

    Main usage:

        criterion = ContrastiveLoss()

        loss = criterion(
            embeddings_a,
            embeddings_b,
        )

    Supported shapes:
        embeddings_a: [B, D]
        embeddings_b: [B, D]

    Supervised contrastive:
        labels: [B]

    Triplet:
        negatives: [B, D]

    Memory queue:
        Stores detached embeddings from
        previous batches as extra negatives.

    Distributed negatives:
        Requires initialized torch.distributed.
        Each rank must have the same local
        batch size for this implementation.
    """

    def __init__(
        self,
        config: ContrastiveLossConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = (
            config or ContrastiveLossConfig()
        )

        self.register_buffer(
            "_queue",
            torch.empty(0, 0),
            persistent=False,
        )

        self.register_buffer(
            "_queue_ptr",
            torch.zeros(
                (),
                dtype=torch.long,
            ),
            persistent=False,
        )

        self.register_buffer(
            "_queue_count",
            torch.zeros(
                (),
                dtype=torch.long,
            ),
            persistent=False,
        )

        self.last_metrics: dict[
            str, float
        ] = {}

    def _validate_embeddings(
        self,
        a: Tensor,
        b: Tensor,
    ) -> None:
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(
                "Embeddings must have shape [B, D]"
            )

        if a.shape != b.shape:
            raise ValueError(
                "Embedding shapes must match"
            )

        if a.shape[0] == 0:
            raise ValueError(
                "Empty embedding batch"
            )

        if a.device != b.device:
            raise ValueError(
                "Embeddings must share a device"
            )

    def _prepare(
        self,
        embeddings: Tensor,
    ) -> Tensor:
        embeddings = embeddings.float()

        if self.config.normalize_embeddings:
            embeddings = F.normalize(
                embeddings,
                p=2,
                dim=-1,
            )

        return embeddings

    def _distributed_gather(
        self,
        embeddings: Tensor,
    ) -> tuple[Tensor, int]:
        if not self.config.distributed_negatives:
            return embeddings, 0

        if (
            not dist.is_available()
            or not dist.is_initialized()
        ):
            raise RuntimeError(
                "Distributed process group "
                "must be initialized"
            )

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        local_size = torch.tensor(
            [embeddings.shape[0]],
            device=embeddings.device,
            dtype=torch.long,
        )

        sizes = [
            torch.zeros_like(local_size)
            for _ in range(world_size)
        ]

        dist.all_gather(
            sizes,
            local_size,
        )

        batch_sizes = [
            int(size.item())
            for size in sizes
        ]

        if len(set(batch_sizes)) != 1:
            raise ValueError(
                "Distributed contrastive learning "
                "requires equal local batch sizes"
            )

        # Gather detached remote embeddings
        # while retaining local gradients.
        gathered = [
            torch.zeros_like(embeddings)
            for _ in range(world_size)
        ]

        dist.all_gather(
            gathered,
            embeddings.detach().contiguous(),
        )

        gathered[rank] = embeddings

        global_embeddings = torch.cat(
            gathered,
            dim=0,
        )

        offset = (
            rank * embeddings.shape[0]
        )

        return global_embeddings, offset

    def _initialize_queue(
        self,
        dimension: int,
        device: torch.device,
    ) -> None:
        if (
            self._queue.ndim == 2
            and self._queue.shape
            == (
                self.config.queue_size,
                dimension,
            )
            and self._queue.device == device
        ):
            return

        self._queue = torch.zeros(
            self.config.queue_size,
            dimension,
            device=device,
            dtype=torch.float32,
        )

        self._queue_ptr.zero_()
        self._queue_count.zero_()

    @torch.no_grad()
    def _enqueue(
        self,
        embeddings: Tensor,
    ) -> None:
        if not self.config.use_memory_queue:
            return

        embeddings = embeddings.detach().float()

        self._initialize_queue(
            embeddings.shape[-1],
            embeddings.device,
        )

        capacity = self.config.queue_size

        if embeddings.shape[0] >= capacity:
            self._queue.copy_(
                embeddings[-capacity:]
            )

            self._queue_ptr.zero_()
            self._queue_count.fill_(
                capacity
            )

            return

        count = embeddings.shape[0]
        pointer = int(
            self._queue_ptr.item()
        )

        first = min(
            count,
            capacity - pointer,
        )

        self._queue[
            pointer:pointer + first
        ] = embeddings[:first]

        remaining = count - first

        if remaining > 0:
            self._queue[
                :remaining
            ] = embeddings[first:]

        self._queue_ptr.fill_(
            (pointer + count) % capacity
        )

        self._queue_count.fill_(
            min(
                capacity,
                int(self._queue_count.item())
                + count,
            )
        )

    def _queued_negatives(
        self,
        dimension: int,
        device: torch.device,
    ) -> Tensor | None:
        if not self.config.use_memory_queue:
            return None

        self._initialize_queue(
            dimension,
            device,
        )

        count = int(
            self._queue_count.item()
        )

        if count == 0:
            return None

        return self._queue[:count].detach()

    def _contrastive_direction(
        self,
        anchors: Tensor,
        positives: Tensor,
        *,
        positive_offset: int = 0,
        queued_negatives: Tensor | None = None,
    ) -> Tensor:
        batch_size = anchors.shape[0]

        similarities = (
            anchors @ positives.T
        ) / self.config.temperature

        labels = (
            torch.arange(
                batch_size,
                device=anchors.device,
            )
            + positive_offset
        )

        if (
            self.config.hard_negative_top_k
            is not None
        ):
            positive_logits = similarities.gather(
                dim=1,
                index=labels.unsqueeze(1),
            )

            negative_mask = torch.ones_like(
                similarities,
                dtype=torch.bool,
            )

            negative_mask.scatter_(
                1,
                labels.unsqueeze(1),
                False,
            )

            negative_logits = (
                similarities.masked_fill(
                    ~negative_mask,
                    float("-inf"),
                )
            )

            k = min(
                self.config.hard_negative_top_k,
                similarities.shape[1] - 1,
            )

            if k > 0:
                hard_negatives = torch.topk(
                    negative_logits,
                    k=k,
                    dim=1,
                ).values

                similarities = torch.cat(
                    [
                        positive_logits,
                        hard_negatives,
                    ],
                    dim=1,
                )

            else:
                similarities = positive_logits

            labels = torch.zeros(
                batch_size,
                device=anchors.device,
                dtype=torch.long,
            )

        if queued_negatives is not None:
            queue_logits = (
                anchors
                @ queued_negatives.T
            ) / self.config.temperature

            similarities = torch.cat(
                [
                    similarities,
                    queue_logits,
                ],
                dim=1,
            )

        return F.cross_entropy(
            similarities,
            labels,
            label_smoothing=(
                self.config.label_smoothing
            ),
        )

    def info_nce_loss(
        self,
        embeddings_a: Tensor,
        embeddings_b: Tensor,
    ) -> Tensor:
        self._validate_embeddings(
            embeddings_a,
            embeddings_b,
        )

        a = self._prepare(
            embeddings_a
        )

        b = self._prepare(
            embeddings_b
        )

        global_a, offset_a = (
            self._distributed_gather(a)
        )

        global_b, offset_b = (
            self._distributed_gather(b)
        )

        queue = self._queued_negatives(
            a.shape[-1],
            a.device,
        )

        loss_ab = self._contrastive_direction(
            a,
            global_b,
            positive_offset=offset_b,
            queued_negatives=queue,
        )

        if self.config.symmetric:
            loss_ba = self._contrastive_direction(
                b,
                global_a,
                positive_offset=offset_a,
                queued_negatives=queue,
            )

            loss = 0.5 * (
                loss_ab + loss_ba
            )

        else:
            loss = loss_ab

        if self.training:
            with torch.no_grad():
                self._enqueue(
                    global_b
                )

        return loss

    def nt_xent_loss(
        self,
        embeddings_a: Tensor,
        embeddings_b: Tensor,
    ) -> Tensor:
        """
        SimCLR-style NT-Xent.

        Each sample has one positive
        in the opposite augmented view.
        """
        self._validate_embeddings(
            embeddings_a,
            embeddings_b,
        )

        a = self._prepare(
            embeddings_a
        )

        b = self._prepare(
            embeddings_b
        )

        a, offset = self._distributed_gather(a)
        b, _ = self._distributed_gather(b)

        batch_size = a.shape[0]

        if batch_size < 2:
            raise ValueError(
                "NT-Xent needs at least two pairs"
            )

        embeddings = torch.cat(
            [a, b],
            dim=0,
        )

        logits = (
            embeddings @ embeddings.T
        ) / self.config.temperature

        diagonal = torch.eye(
            2 * batch_size,
            device=logits.device,
            dtype=torch.bool,
        )

        logits = logits.masked_fill(
            diagonal,
            float("-inf"),
        )

        positive_indices = (
            torch.arange(
                2 * batch_size,
                device=logits.device,
            )
            + batch_size
        ) % (2 * batch_size)

        # Only local anchors contribute
        # to the local training loss.
        if self.config.distributed_negatives:
            local_size = embeddings_a.shape[0]

            local_indices = torch.cat(
                [
                    torch.arange(
                        offset,
                        offset + local_size,
                        device=logits.device,
                    ),
                    torch.arange(
                        batch_size + offset,
                        batch_size + offset + local_size,
                        device=logits.device,
                    ),
                ]
            )

            logits = logits[local_indices]
            positive_indices = (
                positive_indices[local_indices]
            )

        return F.cross_entropy(
            logits,
            positive_indices,
        )

    def supervised_contrastive_loss(
        self,
        embeddings: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """
        Supervised contrastive learning.

        Samples with matching labels
        are treated as positive pairs.
        """
        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be [B,D]"
            )

        if (
            labels.ndim != 1
            or labels.shape[0]
            != embeddings.shape[0]
        ):
            raise ValueError(
                "Labels must have shape [B]"
            )

        embeddings = self._prepare(
            embeddings
        )

        if self.config.distributed_negatives:
            raise NotImplementedError(
                "Distributed supervised contrastive "
                "requires gathering labels and "
                "is not enabled in this module"
            )

        similarities = (
            embeddings @ embeddings.T
        ) / self.config.temperature

        batch_size = embeddings.shape[0]

        identity = torch.eye(
            batch_size,
            device=embeddings.device,
            dtype=torch.bool,
        )

        positive_mask = (
            labels.unsqueeze(0)
            == labels.unsqueeze(1)
        ) & ~identity

        similarities = similarities.masked_fill(
            identity,
            float("-inf"),
        )

        log_probabilities = F.log_softmax(
            similarities,
            dim=1,
        )

        positive_count = (
            positive_mask.sum(dim=1)
        )

        valid_anchors = (
            positive_count > 0
        )

        if not valid_anchors.any():
            raise ValueError(
                "No positive pairs found"
            )

        positive_log_probabilities = (
            log_probabilities.masked_fill(
                ~positive_mask,
                0.0,
            ).sum(dim=1)
        )

        losses = (
            -positive_log_probabilities
            / positive_count.clamp_min(1)
        )

        return losses[
            valid_anchors
        ].mean()

    def triplet_loss(
        self,
        anchors: Tensor,
        positives: Tensor,
        negatives: Tensor,
    ) -> Tensor:
        self._validate_embeddings(
            anchors,
            positives,
        )

        self._validate_embeddings(
            anchors,
            negatives,
        )

        anchors = self._prepare(
            anchors
        )

        positives = self._prepare(
            positives
        )

        negatives = self._prepare(
            negatives
        )

        return F.triplet_margin_loss(
            anchors,
            positives,
            negatives,
            margin=self.config.margin,
            p=2,
        )

    def cosine_loss(
        self,
        embeddings_a: Tensor,
        embeddings_b: Tensor,
    ) -> Tensor:
        self._validate_embeddings(
            embeddings_a,
            embeddings_b,
        )

        similarity = F.cosine_similarity(
            embeddings_a.float(),
            embeddings_b.float(),
            dim=-1,
        )

        return (
            1.0 - similarity
        ).mean()

    def byol_loss(
        self,
        predictions: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """
        BYOL-style alignment loss.

        The target branch is detached.
        A full BYOL training system
        additionally needs a target
        network with EMA updates.
        """
        self._validate_embeddings(
            predictions,
            targets,
        )

        predictions = F.normalize(
            predictions.float(),
            dim=-1,
        )

        targets = F.normalize(
            targets.float().detach(),
            dim=-1,
        )

        return (
            2.0
            - 2.0
            * (
                predictions * targets
            ).sum(dim=-1)
        ).mean()

    @staticmethod
    def _off_diagonal(
        matrix: Tensor,
    ) -> Tensor:
        dimension = matrix.shape[0]

        if matrix.shape != (
            dimension,
            dimension,
        ):
            raise ValueError(
                "Matrix must be square"
            )

        mask = ~torch.eye(
            dimension,
            device=matrix.device,
            dtype=torch.bool,
        )

        return matrix[mask]

    def vicreg_loss(
        self,
        embeddings_a: Tensor,
        embeddings_b: Tensor,
    ) -> Tensor:
        """
        VICReg objective:
        - Invariance
        - Variance regularization
        - Covariance decorrelation
        """
        self._validate_embeddings(
            embeddings_a,
            embeddings_b,
        )

        a = embeddings_a.float()
        b = embeddings_b.float()

        if a.shape[0] < 2:
            raise ValueError(
                "VICReg requires batch size >= 2"
            )

        similarity_loss = F.mse_loss(
            a,
            b,
        )

        std_a = torch.sqrt(
            a.var(
                dim=0,
                unbiased=True,
            )
            + self.config.variance_epsilon
        )

        std_b = torch.sqrt(
            b.var(
                dim=0,
                unbiased=True,
            )
            + self.config.variance_epsilon
        )

        variance_loss = (
            F.relu(
                self.config.variance_target
                - std_a
            ).mean()
            + F.relu(
                self.config.variance_target
                - std_b
            ).mean()
        ) / 2.0

        centered_a = (
            a - a.mean(dim=0)
        )

        centered_b = (
            b - b.mean(dim=0)
        )

        covariance_a = (
            centered_a.T @ centered_a
        ) / (a.shape[0] - 1)

        covariance_b = (
            centered_b.T @ centered_b
        ) / (b.shape[0] - 1)

        dimension = a.shape[1]

        covariance_loss = (
            self._off_diagonal(
                covariance_a
            ).square().sum()
            / dimension
            + self._off_diagonal(
                covariance_b
            ).square().sum()
            / dimension
        )

        total = (
            self.config.vicreg_similarity_weight
            * similarity_loss
            + self.config.vicreg_variance_weight
            * variance_loss
            + self.config.vicreg_covariance_weight
            * covariance_loss
        )

        self.last_metrics.update(
            {
                "vicreg_similarity": float(
                    similarity_loss.detach().item()
                ),
                "vicreg_variance": float(
                    variance_loss.detach().item()
                ),
                "vicreg_covariance": float(
                    covariance_loss.detach().item()
                ),
            }
        )

        return total

    def forward(
        self,
        embeddings_a: Tensor,
        embeddings_b: Tensor,
        *,
        labels: Tensor | None = None,
        negatives: Tensor | None = None,
        loss_type: str | None = None,
    ) -> Tensor:
        selected = (
            loss_type
            or self.config.loss_type
        )

        if selected == "infonce":
            loss = self.info_nce_loss(
                embeddings_a,
                embeddings_b,
            )

        elif selected == "nt_xent":
            loss = self.nt_xent_loss(
                embeddings_a,
                embeddings_b,
            )

        elif selected == "supervised":
            if labels is None:
                raise ValueError(
                    "Supervised contrastive loss "
                    "requires labels"
                )

            loss = self.supervised_contrastive_loss(
                embeddings_a,
                labels,
            )

        elif selected == "triplet":
            if negatives is None:
                raise ValueError(
                    "Triplet loss requires negatives"
                )

            loss = self.triplet_loss(
                embeddings_a,
                embeddings_b,
                negatives,
            )

        elif selected == "cosine":
            loss = self.cosine_loss(
                embeddings_a,
                embeddings_b,
            )

        elif selected == "byol":
            loss = self.byol_loss(
                embeddings_a,
                embeddings_b,
            )

        elif selected == "vicreg":
            loss = self.vicreg_loss(
                embeddings_a,
                embeddings_b,
            )

        else:
            raise ValueError(
                f"Unknown loss type: {selected}"
            )

        if (
            self.config.detect_nonfinite
            and not torch.isfinite(loss).item()
        ):
            raise FloatingPointError(
                f"Non-finite {selected} loss"
            )

        self.last_metrics.update(
            {
                "contrastive_loss": float(
                    loss.detach().item()
                ),
                "temperature": (
                    self.config.temperature
                ),
                "batch_size": float(
                    embeddings_a.shape[0]
                ),
            }
        )

        return loss

    def get_last_metrics(
        self,
    ) -> dict[str, float]:
        return dict(
            self.last_metrics
        )

    @torch.no_grad()
    def reset_queue(self) -> None:
        self._queue = torch.empty(
            0,
            0,
            device=self._queue.device,
        )

        self._queue_ptr.zero_()
        self._queue_count.zero_()

    def queue_info(
        self,
    ) -> dict[str, Any]:
        return {
            "enabled": (
                self.config.use_memory_queue
            ),
            "capacity": (
                self.config.queue_size
            ),
            "count": int(
                self._queue_count.item()
            ),
            "pointer": int(
                self._queue_ptr.item()
            ),
        }


def create_contrastive_loss(
    *,
    loss_type: str = "infonce",
    temperature: float = 0.07,
    symmetric: bool = True,
    use_memory_queue: bool = False,
    queue_size: int = 4096,
    distributed_negatives: bool = False,
) -> ContrastiveLoss:
    return ContrastiveLoss(
        ContrastiveLossConfig(
            loss_type=loss_type,
            temperature=temperature,
            symmetric=symmetric,
            use_memory_queue=(
                use_memory_queue
            ),
            queue_size=queue_size,
            distributed_negatives=(
                distributed_negatives
            ),
        )
    )


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 16
    input_dim = 128
    embedding_dim = 64

    encoder = nn.Sequential(
        nn.Linear(
            input_dim,
            256,
        ),
        nn.ReLU(),
        nn.Linear(
            256,
            embedding_dim,
        ),
    )

    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=3e-4,
    )

    criterion = ContrastiveLoss(
        ContrastiveLossConfig(
            loss_type="infonce",
            temperature=0.07,
            symmetric=True,
            use_memory_queue=True,
            queue_size=128,
        )
    )

    for step in range(5):
        inputs = torch.randn(
            batch_size,
            input_dim,
        )

        augmented_a = (
            inputs
            + 0.05 * torch.randn_like(inputs)
        )

        augmented_b = (
            inputs
            + 0.05 * torch.randn_like(inputs)
        )

        embeddings_a = encoder(
            augmented_a
        )

        embeddings_b = encoder(
            augmented_b
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss = criterion(
            embeddings_a,
            embeddings_b,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            encoder.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        print(
            f"Step {step + 1}: "
            f"{criterion.get_last_metrics()}"
        )

    print(
        "Memory queue:",
        criterion.queue_info(),
    )

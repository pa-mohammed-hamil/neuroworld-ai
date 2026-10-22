"""
NeuroWorld AI
File: training/losses/kl_divergence.py

KL divergence losses for:
- Variational autoencoders (VAE)
- World models and RSSM
- Latent dynamics models
- Probabilistic state prediction
- Video representation learning
- Teacher-student distillation
- Discrete latent distributions

Features:
- Diagonal Gaussian KL divergence
- KL divergence against standard normal
- Categorical KL divergence
- Symmetric KL divergence
- Jensen-Shannon divergence
- Balanced KL for world models
- Free nats / free bits
- Beta-VAE weighting
- Linear and cosine KL warmup
- Masked sequence support
- KL capacity scheduling
- Per-component logging
- Numerical stability checks

Requires:
    PyTorch >= 2.1
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class KLDivergenceConfig:
    loss_type: str = "gaussian"

    beta: float = 1.0

    reduction: str = "mean"

    free_nats: float = 0.0
    free_bits: float = 0.0

    warmup_steps: int = 0
    warmup_type: str = "linear"

    min_beta: float = 0.0

    balance_alpha: float = 0.8

    min_logvar: float = -20.0
    max_logvar: float = 10.0

    epsilon: float = 1e-8

    capacity: float | None = None
    capacity_warmup_steps: int = 0

    detach_prior: bool = False
    detach_posterior: bool = False

    detect_nonfinite: bool = True

    def __post_init__(self) -> None:
        supported_losses = {
            "gaussian",
            "standard_normal",
            "categorical",
            "symmetric",
            "js",
            "balanced",
        }

        if self.loss_type not in supported_losses:
            raise ValueError(
                f"Unsupported KL loss: {self.loss_type}"
            )

        if self.reduction not in {
            "mean", "sum", "none"
        }:
            raise ValueError(
                "reduction must be mean, sum or none"
            )

        if self.warmup_type not in {
            "linear", "cosine"
        }:
            raise ValueError(
                "warmup_type must be linear or cosine"
            )

        if not math.isfinite(self.beta) or self.beta < 0:
            raise ValueError(
                "beta must be finite and nonnegative"
            )

        if (
            not math.isfinite(self.min_beta)
            or self.min_beta < 0
            or self.min_beta > self.beta
        ):
            raise ValueError(
                "min_beta must be between 0 and beta"
            )

        if self.free_nats < 0 or self.free_bits < 0:
            raise ValueError(
                "Free nats/bits must be nonnegative"
            )

        if not 0 <= self.balance_alpha <= 1:
            raise ValueError(
                "balance_alpha must be in [0,1]"
            )

        if self.min_logvar >= self.max_logvar:
            raise ValueError(
                "min_logvar must be below max_logvar"
            )

        if self.epsilon <= 0:
            raise ValueError(
                "epsilon must be positive"
            )

        if self.warmup_steps < 0:
            raise ValueError(
                "warmup_steps must be nonnegative"
            )

        if self.capacity_warmup_steps < 0:
            raise ValueError(
                "capacity_warmup_steps must be nonnegative"
            )

        if self.capacity is not None and self.capacity < 0:
            raise ValueError(
                "capacity must be nonnegative"
            )

        if self.detach_prior and self.detach_posterior:
            raise ValueError(
                "Cannot detach both prior and posterior"
            )


class KLDivergenceLoss(nn.Module):
    """
    Configurable KL divergence module.

    Gaussian inputs:
        posterior_mean:   [B, ..., D]
        posterior_logvar: [B, ..., D]
        prior_mean:       [B, ..., D]
        prior_logvar:     [B, ..., D]

    Categorical inputs:
        posterior_logits: [B, ..., C]
        prior_logits:     [B, ..., C]

    Last dimension represents latent
    features or categorical classes.

    Mask:
        [B] or [B, T] for sequences.

    For reduction="none", returns
    per-sample/per-timestep KL values.
    """

    def __init__(
        self,
        config: KLDivergenceConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = (
            config or KLDivergenceConfig()
        )

        self.step_count = 0

        self.last_metrics: dict[str, float] = {}

    @staticmethod
    def _get(
        values: Mapping[str, Any],
        *names: str,
    ) -> Tensor | None:
        for name in names:
            value = values.get(name)

            if isinstance(value, Tensor):
                return value

        return None

    def _validate_gaussian(
        self,
        mean: Tensor,
        logvar: Tensor,
    ) -> None:
        if mean.shape != logvar.shape:
            raise ValueError(
                "Gaussian mean and logvar shapes differ"
            )

        if mean.ndim < 2:
            raise ValueError(
                "Gaussian tensors require [B,D] "
                "or [B,T,D]"
            )

    def _prepare_logvar(
        self,
        logvar: Tensor,
    ) -> Tensor:
        return logvar.float().clamp(
            min=self.config.min_logvar,
            max=self.config.max_logvar,
        )

    @staticmethod
    def _prepare_mask(
        mask: Tensor | None,
        values: Tensor,
    ) -> Tensor | None:
        if mask is None:
            return None

        mask = mask.to(
            device=values.device,
            dtype=torch.bool,
        )

        if mask.shape[0] != values.shape[0]:
            raise ValueError(
                "Mask batch dimension mismatch"
            )

        while mask.ndim < values.ndim:
            mask = mask.unsqueeze(-1)

        try:
            return torch.broadcast_to(
                mask,
                values.shape,
            )

        except RuntimeError as exc:
            raise ValueError(
                "Mask cannot broadcast to KL shape"
            ) from exc

    def _reduce(
        self,
        values: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        mask = self._prepare_mask(
            mask,
            values,
        )

        if mask is not None:
            values = values * mask.to(
                values.dtype
            )

        if self.config.reduction == "none":
            return values

        if self.config.reduction == "sum":
            return values.sum()

        if mask is None:
            return values.mean()

        denominator = (
            mask.sum().clamp_min(1)
        )

        return (
            values.sum() / denominator
        )

    def _apply_free_nats(
        self,
        kl: Tensor,
        latent_dimension: int,
    ) -> Tensor:
        threshold = max(
            self.config.free_nats,
            self.config.free_bits
            * latent_dimension
            * math.log(2.0),
        )

        if threshold > 0:
            kl = kl.clamp_min(
                threshold
            )

        return kl

    def gaussian_kl(
        self,
        posterior_mean: Tensor,
        posterior_logvar: Tensor,
        prior_mean: Tensor | None = None,
        prior_logvar: Tensor | None = None,
        *,
        mask: Tensor | None = None,
        apply_free_nats: bool = True,
    ) -> Tensor:
        """
        KL(q || p) for diagonal Gaussians.

        q = N(posterior_mean, posterior_var)
        p = N(prior_mean, prior_var)

        Defaults to a standard normal prior.
        """
        self._validate_gaussian(
            posterior_mean,
            posterior_logvar,
        )

        if (
            (prior_mean is None)
            != (prior_logvar is None)
        ):
            raise ValueError(
                "Provide both prior mean and logvar"
            )

        q_mean = posterior_mean.float()
        q_logvar = self._prepare_logvar(
            posterior_logvar
        )

        if prior_mean is None:
            p_mean = torch.zeros_like(q_mean)
            p_logvar = torch.zeros_like(q_logvar)

        else:
            self._validate_gaussian(
                prior_mean,
                prior_logvar,
            )

            if prior_mean.shape != q_mean.shape:
                raise ValueError(
                    "Prior/posterior shapes differ"
                )

            p_mean = prior_mean.float()
            p_logvar = self._prepare_logvar(
                prior_logvar
            )

        if self.config.detach_posterior:
            q_mean = q_mean.detach()
            q_logvar = q_logvar.detach()

        if self.config.detach_prior:
            p_mean = p_mean.detach()
            p_logvar = p_logvar.detach()

        variance_ratio = torch.exp(
            q_logvar - p_logvar
        )

        mean_difference = (
            q_mean - p_mean
        ).square() * torch.exp(
            -p_logvar
        )

        kl_per_dimension = 0.5 * (
            p_logvar
            - q_logvar
            + variance_ratio
            + mean_difference
            - 1.0
        )

        kl = kl_per_dimension.sum(
            dim=-1
        )

        if apply_free_nats:
            kl = self._apply_free_nats(
                kl,
                latent_dimension=q_mean.shape[-1],
            )

        return self._reduce(
            kl,
            mask,
        )

    def standard_normal_kl(
        self,
        mean: Tensor,
        logvar: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> Tensor:
        return self.gaussian_kl(
            mean,
            logvar,
            mask=mask,
        )

    def categorical_kl(
        self,
        posterior_logits: Tensor,
        prior_logits: Tensor | None = None,
        *,
        mask: Tensor | None = None,
        apply_free_nats: bool = True,
    ) -> Tensor:
        """
        KL between categorical distributions.

        If prior_logits is None,
        use a uniform categorical prior.
        """
        if posterior_logits.ndim < 2:
            raise ValueError(
                "Categorical logits require [B,C] "
                "or [B,T,C]"
            )

        q_logits = posterior_logits.float()

        if prior_logits is None:
            p_logits = torch.zeros_like(
                q_logits
            )

        else:
            if prior_logits.shape != q_logits.shape:
                raise ValueError(
                    "Categorical distribution "
                    "shapes differ"
                )

            p_logits = prior_logits.float()

        if self.config.detach_posterior:
            q_logits = q_logits.detach()

        if self.config.detach_prior:
            p_logits = p_logits.detach()

        q_log_prob = F.log_softmax(
            q_logits,
            dim=-1,
        )

        p_log_prob = F.log_softmax(
            p_logits,
            dim=-1,
        )

        q_prob = q_log_prob.exp()

        kl = (
            q_prob
            * (q_log_prob - p_log_prob)
        ).sum(dim=-1)

        if apply_free_nats:
            kl = self._apply_free_nats(
                kl,
                latent_dimension=1,
            )

        return self._reduce(
            kl,
            mask,
        )

    def symmetric_kl(
        self,
        mean_a: Tensor,
        logvar_a: Tensor,
        mean_b: Tensor,
        logvar_b: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> Tensor:
        """
        Symmetric Gaussian KL:

        0.5 * [KL(A||B) + KL(B||A)]
        """
        kl_ab = self.gaussian_kl(
            mean_a,
            logvar_a,
            mean_b,
            logvar_b,
            mask=mask,
            apply_free_nats=False,
        )

        kl_ba = self.gaussian_kl(
            mean_b,
            logvar_b,
            mean_a,
            logvar_a,
            mask=mask,
            apply_free_nats=False,
        )

        return 0.5 * (
            kl_ab + kl_ba
        )

    def jensen_shannon(
        self,
        logits_a: Tensor,
        logits_b: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> Tensor:
        """
        Jensen-Shannon divergence
        between categorical distributions.
        """
        if logits_a.shape != logits_b.shape:
            raise ValueError(
                "JS distribution shapes differ"
            )

        log_a = F.log_softmax(
            logits_a.float(),
            dim=-1,
        )

        log_b = F.log_softmax(
            logits_b.float(),
            dim=-1,
        )

        prob_a = log_a.exp()
        prob_b = log_b.exp()

        mixture = 0.5 * (
            prob_a + prob_b
        )

        log_mixture = mixture.clamp_min(
            self.config.epsilon
        ).log()

        kl_a = (
            prob_a
            * (log_a - log_mixture)
        ).sum(dim=-1)

        kl_b = (
            prob_b
            * (log_b - log_mixture)
        ).sum(dim=-1)

        js = 0.5 * (
            kl_a + kl_b
        )

        return self._reduce(
            js,
            mask,
        )

    def balanced_kl(
        self,
        posterior_mean: Tensor,
        posterior_logvar: Tensor,
        prior_mean: Tensor,
        prior_logvar: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> Tensor:
        """
        Balanced KL for world-model training.

        alpha * KL(stopgrad(q) || p)
        +
        (1-alpha) * KL(q || stopgrad(p))

        The first term updates the prior.
        The second term updates the posterior.
        """
        alpha = self.config.balance_alpha

        prior_learning_loss = self.gaussian_kl(
            posterior_mean.detach(),
            posterior_logvar.detach(),
            prior_mean,
            prior_logvar,
            mask=mask,
            apply_free_nats=False,
        )

        posterior_learning_loss = self.gaussian_kl(
            posterior_mean,
            posterior_logvar,
            prior_mean.detach(),
            prior_logvar.detach(),
            mask=mask,
            apply_free_nats=False,
        )

        return (
            alpha * prior_learning_loss
            + (1.0 - alpha)
            * posterior_learning_loss
        )

    def current_beta(
        self,
        step: int | None = None,
    ) -> float:
        if step is None:
            step = self.step_count

        if step < 0:
            raise ValueError(
                "step must be nonnegative"
            )

        if self.config.warmup_steps == 0:
            return self.config.beta

        progress = min(
            1.0,
            step / self.config.warmup_steps,
        )

        if self.config.warmup_type == "cosine":
            progress = 0.5 * (
                1.0 - math.cos(
                    math.pi * progress
                )
            )

        return (
            self.config.min_beta
            + (
                self.config.beta
                - self.config.min_beta
            ) * progress
        )

    def current_capacity(
        self,
        step: int | None = None,
    ) -> float | None:
        if self.config.capacity is None:
            return None

        if step is None:
            step = self.step_count

        if self.config.capacity_warmup_steps == 0:
            return self.config.capacity

        progress = min(
            1.0,
            step / self.config.capacity_warmup_steps,
        )

        return (
            self.config.capacity * progress
        )

    def forward(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any] | None = None,
        *,
        step: int | None = None,
        return_components: bool = True,
    ) -> (
        Tensor
        | tuple[Tensor, dict[str, Tensor]]
    ):
        """
        Calculate KL divergence loss.

        Supported output keys:
            posterior_mean
            posterior_logvar
            latent_mean
            latent_logvar
            prior_mean
            prior_logvar
            posterior_logits
            prior_logits

        Supported target keys:
            mask
            sequence_mask
            frame_mask

        Returns:
            total_loss

        Or:
            total_loss, components
        """
        if targets is None:
            targets = {}

        if step is not None:
            if step < 0:
                raise ValueError(
                    "step must be nonnegative"
                )

            self.step_count = int(step)

        mask = self._get(
            targets,
            "mask",
            "sequence_mask",
            "frame_mask",
        )

        selected = self.config.loss_type

        if selected in {
            "gaussian",
            "standard_normal",
            "symmetric",
            "balanced",
        }:
            posterior_mean = self._get(
                outputs,
                "posterior_mean",
                "latent_mean",
                "mean",
            )

            posterior_logvar = self._get(
                outputs,
                "posterior_logvar",
                "latent_logvar",
                "logvar",
            )

            if (
                posterior_mean is None
                or posterior_logvar is None
            ):
                raise ValueError(
                    "Gaussian KL requires posterior "
                    "mean and logvar"
                )

            prior_mean = self._get(
                outputs,
                "prior_mean",
            )

            prior_logvar = self._get(
                outputs,
                "prior_logvar",
            )

            if selected == "standard_normal":
                raw_loss = self.standard_normal_kl(
                    posterior_mean,
                    posterior_logvar,
                    mask=mask,
                )

            elif selected == "symmetric":
                if (
                    prior_mean is None
                    or prior_logvar is None
                ):
                    raise ValueError(
                        "Symmetric KL requires prior "
                        "mean and logvar"
                    )

                raw_loss = self.symmetric_kl(
                    posterior_mean,
                    posterior_logvar,
                    prior_mean,
                    prior_logvar,
                    mask=mask,
                )

            elif selected == "balanced":
                if (
                    prior_mean is None
                    or prior_logvar is None
                ):
                    raise ValueError(
                        "Balanced KL requires prior "
                        "mean and logvar"
                    )

                raw_loss = self.balanced_kl(
                    posterior_mean,
                    posterior_logvar,
                    prior_mean,
                    prior_logvar,
                    mask=mask,
                )

            else:
                raw_loss = self.gaussian_kl(
                    posterior_mean,
                    posterior_logvar,
                    prior_mean,
                    prior_logvar,
                    mask=mask,
                )

        else:
            posterior_logits = self._get(
                outputs,
                "posterior_logits",
                "latent_logits",
            )

            prior_logits = self._get(
                outputs,
                "prior_logits",
            )

            if posterior_logits is None:
                raise ValueError(
                    "Categorical KL requires "
                    "posterior logits"
                )

            if selected == "js":
                if prior_logits is None:
                    raise ValueError(
                        "JS divergence requires "
                        "prior logits"
                    )

                raw_loss = self.jensen_shannon(
                    posterior_logits,
                    prior_logits,
                    mask=mask,
                )

            else:
                raw_loss = self.categorical_kl(
                    posterior_logits,
                    prior_logits,
                    mask=mask,
                )

        beta = self.current_beta(
            self.step_count
        )

        capacity = self.current_capacity(
            self.step_count
        )

        if capacity is not None:
            # Capacity is applied to a scalar
            # aggregate, not elementwise.
            if raw_loss.ndim != 0:
                raise ValueError(
                    "KL capacity requires "
                    "mean or sum reduction"
                )

            objective = (
                raw_loss - capacity
            ).abs()

        else:
            objective = raw_loss

        total_loss = (
            objective * beta
        )

        if (
            self.config.detect_nonfinite
            and not torch.isfinite(
                total_loss
            ).all().item()
        ):
            raise FloatingPointError(
                "Non-finite KL divergence loss"
            )

        self.last_metrics = {
            "kl_raw": float(
                raw_loss.detach().mean().item()
            ),
            "kl_total": float(
                total_loss.detach().mean().item()
            ),
            "kl_beta": float(beta),
            "kl_capacity": (
                float(capacity)
                if capacity is not None
                else 0.0
            ),
            "kl_step": float(
                self.step_count
            ),
        }

        if step is None:
            self.step_count += 1

        if return_components:
            return total_loss, {
                "total": total_loss,
                "kl": raw_loss,
                "objective": objective,
            }

        return total_loss

    def get_last_metrics(
        self,
    ) -> dict[str, float]:
        return dict(
            self.last_metrics
        )

    def reset(self) -> None:
        self.step_count = 0
        self.last_metrics.clear()

    def get_extra_state(
        self,
    ) -> dict[str, Any]:
        return {
            "step_count": self.step_count,
        }

    def set_extra_state(
        self,
        state: Mapping[str, Any],
    ) -> None:
        self.step_count = int(
            state.get(
                "step_count",
                0,
            )
        )


def create_kl_divergence_loss(
    *,
    loss_type: str = "gaussian",
    beta: float = 1.0,
    free_nats: float = 0.0,
    warmup_steps: int = 0,
    balance_alpha: float = 0.8,
) -> KLDivergenceLoss:
    return KLDivergenceLoss(
        KLDivergenceConfig(
            loss_type=loss_type,
            beta=beta,
            free_nats=free_nats,
            warmup_steps=warmup_steps,
            balance_alpha=balance_alpha,
        )
    )


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 8
    sequence_length = 12
    latent_dim = 64

    class DemoVariationalModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.encoder = nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(),
            )

            self.mean_head = nn.Linear(
                256,
                latent_dim,
            )

            self.logvar_head = nn.Linear(
                256,
                latent_dim,
            )

            self.prior_mean = nn.Parameter(
                torch.zeros(latent_dim)
            )

            self.prior_logvar = nn.Parameter(
                torch.zeros(latent_dim)
            )

        def forward(
            self,
            states: Tensor,
        ) -> dict[str, Tensor]:
            hidden = self.encoder(
                states
            )

            posterior_mean = (
                self.mean_head(hidden)
            )

            posterior_logvar = (
                self.logvar_head(hidden)
            )

            prior_mean = (
                self.prior_mean
                .expand_as(posterior_mean)
            )

            prior_logvar = (
                self.prior_logvar
                .expand_as(posterior_logvar)
            )

            return {
                "posterior_mean": posterior_mean,
                "posterior_logvar": posterior_logvar,
                "prior_mean": prior_mean,
                "prior_logvar": prior_logvar,
            }

    model = DemoVariationalModel()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )

    criterion = KLDivergenceLoss(
        KLDivergenceConfig(
            loss_type="balanced",
            beta=0.1,
            balance_alpha=0.8,
            warmup_steps=100,
            free_nats=0.0,
            reduction="mean",
        )
    )

    for step in range(5):
        states = torch.randn(
            batch_size,
            sequence_length,
            128,
        )

        outputs = model(
            states
        )

        targets = {
            "mask": torch.ones(
                batch_size,
                sequence_length,
                dtype=torch.bool,
            )
        }

        optimizer.zero_grad(
            set_to_none=True
        )

        loss, components = criterion(
            outputs,
            targets,
            step=step,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        print(
            f"Step {step + 1}: "
            f"{criterion.get_last_metrics()}"
        )
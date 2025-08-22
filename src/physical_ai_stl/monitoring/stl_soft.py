# ruff: noqa: I001
from __future__ import annotations

from dataclasses import dataclass

import torch


# ---------------------------------------------------------------------------
# Core smooth aggregations
# ---------------------------------------------------------------------------

def _check_temp(temp: float) -> float:
    if temp <= 0:
        raise ValueError(f"temp must be > 0, got {temp}")
    return float(temp)


def softmin(x: torch.Tensor, *, temp: float = 0.1, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """Smooth approximation of min via log-sum-exp.

    Computes  -τ · logsumexp(-x/τ)  along ``dim``.
    When τ→0⁺ this approaches ``x.min(dim)`` and when τ grows the surface becomes smoother.
    """
    tau = _check_temp(temp)
    return -(tau * torch.logsumexp(-x / tau, dim=dim, keepdim=keepdim))


def softmax(x: torch.Tensor, *, temp: float = 0.1, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """Smooth approximation of max via log-sum-exp.

    Computes  τ · logsumexp(x/τ)  along ``dim``.
    """
    tau = _check_temp(temp)
    return tau * torch.logsumexp(x / tau, dim=dim, keepdim=keepdim)


def soft_and(a: torch.Tensor, b: torch.Tensor, *, temp: float = 0.1) -> torch.Tensor:
    """Smooth conjunction: ≈ min(a, b)."""
    a, b = torch.broadcast_tensors(a, b)
    stacked = torch.stack([a, b], dim=-1)
    return softmin(stacked, temp=temp, dim=-1)


def soft_or(a: torch.Tensor, b: torch.Tensor, *, temp: float = 0.1) -> torch.Tensor:
    """Smooth disjunction: ≈ max(a, b)."""
    a, b = torch.broadcast_tensors(a, b)
    stacked = torch.stack([a, b], dim=-1)
    return softmax(stacked, temp=temp, dim=-1)


def soft_not(r: torch.Tensor) -> torch.Tensor:
    """Quantitative negation for robustness margins."""
    return -r


def soft_implies(a: torch.Tensor, b: torch.Tensor, *, temp: float = 0.1) -> torch.Tensor:
    """Smooth implication: a → b  ≡  ¬a ∨ b."""
    return soft_or(soft_not(a), b, temp=temp)


# ---------------------------------------------------------------------------
# Predicates → per-time robustness margins
# ---------------------------------------------------------------------------

def pred_leq(u: torch.Tensor, c: float | torch.Tensor) -> torch.Tensor:
    """u ≤ c  →  robustness = c - u"""
    c = torch.as_tensor(c, dtype=u.dtype, device=u.device)
    return c - u


def pred_geq(u: torch.Tensor, c: float | torch.Tensor) -> torch.Tensor:
    """u ≥ c  →  robustness = u - c"""
    c = torch.as_tensor(c, dtype=u.dtype, device=u.device)
    return u - c


def pred_abs_leq(u: torch.Tensor, c: float | torch.Tensor) -> torch.Tensor:
    """|u| ≤ c  →  robustness = c - |u|"""
    c = torch.as_tensor(c, dtype=u.dtype, device=u.device)
    return c - u.abs()


def pred_linear_leq(x: torch.Tensor, a: torch.Tensor, b: float | torch.Tensor) -> torch.Tensor:
    r"""a·x ≤ b  →  robustness = b - a·x

    ``x`` and ``a`` are broadcast and the inner product is taken over the last dimension.
    """
    b = torch.as_tensor(b, dtype=x.dtype, device=x.device)
    ax = (x * a).sum(dim=-1)
    return b - ax


# ---------------------------------------------------------------------------
# Temporal operators over a time axis
# ---------------------------------------------------------------------------

def _move_time_last(x: torch.Tensor, time_dim: int) -> tuple[torch.Tensor, int]:
    """Move the chosen time dimension to the last position for windowed ops."""
    time_dim = int(time_dim) % x.ndim
    if time_dim != x.ndim - 1:
        x = x.movedim(time_dim, -1)
    return x, time_dim


def always(margins: torch.Tensor, *, temp: float = 0.1, time_dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """Unbounded 'always' (□): smooth min over the entire time axis."""
    return softmin(margins, temp=temp, dim=time_dim, keepdim=keepdim)


def eventually(margins: torch.Tensor, *, temp: float = 0.1, time_dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """Unbounded 'eventually' (◇): smooth max over the entire time axis."""
    return softmax(margins, temp=temp, dim=time_dim, keepdim=keepdim)


def _unfold_time(x: torch.Tensor, *, window: int, stride: int) -> torch.Tensor:
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    return x.unfold(dimension=-1, size=window, step=stride)


def _windowed_soft_agg(x: torch.Tensor, *, window: int, stride: int, temp: float, kind: str) -> torch.Tensor:
    tau = _check_temp(temp)
    xw = _unfold_time(x, window=window, stride=stride)  # (..., L, W)
    if kind == "max":
        # τ * logsumexp(x / τ) over the window dimension
        y = tau * torch.logsumexp(xw / tau, dim=-1)
    elif kind == "min":
        y = -(tau * torch.logsumexp(-xw / tau, dim=-1))
    else:
        raise ValueError("kind must be 'min' or 'max'")
    return y  # (..., L)


def always_window(
    margins: torch.Tensor,
    *,
    window: int,
    stride: int = 1,
    temp: float = 0.1,
    time_dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:
    """Bounded 'always' (□_[0,window-1])."""
    x, old_dim = _move_time_last(margins, time_dim)
    y = _windowed_soft_agg(x, window=window, stride=stride, temp=temp, kind="min")
    if keepdim:
        y = y.unsqueeze(-1)
    if old_dim != x.ndim - 1:
        y = y.movedim(-1, old_dim)
    return y


def eventually_window(
    margins: torch.Tensor,
    *,
    window: int,
    stride: int = 1,
    temp: float = 0.1,
    time_dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:
    """Bounded 'eventually' (◇_[0,window-1])."""
    x, old_dim = _move_time_last(margins, time_dim)
    y = _windowed_soft_agg(x, window=window, stride=stride, temp=temp, kind="max")
    if keepdim:
        y = y.unsqueeze(-1)
    if old_dim != x.ndim - 1:
        y = y.movedim(-1, old_dim)
    return y


def _valid_hi(T: int, hi: int | None) -> int:
    if hi is None:
        return T - 1
    if hi < 0:
        raise ValueError(f"hi must be ≥ 0, got {hi}")
    return hi


def until_window(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    lo: int = 0,
    hi: int | None = None,
    temp: float = 0.1,
    time_dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Bounded 'until'  (a U_[lo,hi] b) with smooth quantitative semantics.

    Discrete-time semantics (exact) for a sequence starting at t:
        (a U_[lo,hi] b)(t) = max_{k ∈ [lo, hi]}
                                min( b[t+k],  min_{τ ∈ [t, t+k)} a[τ] )

    We approximate the outer max and inner mins with ``softmax`` / ``softmin``
    controlled by ``temp``. The implementation is fully vectorized and uses
    ``logcumsumexp`` to compute all prefix soft-mins efficiently.

    Output length along time is ``T - hi`` (number of valid windows).
    """
    tau = _check_temp(temp)

    # Shape & alignment
    a_m, old_dim = _move_time_last(a, time_dim)
    b_m, _ = _move_time_last(b, time_dim)
    if a_m.shape != b_m.shape:
        raise ValueError(f"a and b must have same shape (after broadcasting along non-time dims); got {a_m.shape=} vs {b_m.shape=}")

    T = a_m.shape[-1]
    if lo < 0:
        raise ValueError(f"lo must be ≥ 0, got {lo}")
    hi = _valid_hi(T, hi)
    hi = min(int(hi), T - 1)
    if lo > hi:
        raise ValueError(f"Require lo ≤ hi, got lo={lo}, hi={hi}")

    W = hi + 1
    # Windows starting at each t of size W: (..., L, W) where L = T - hi
    a_w = _unfold_time(a_m, window=W, stride=1)
    b_w = _unfold_time(b_m, window=W, stride=1)

    # Prefix soft-mins of a within each window:
    # s_k = softmin(a_w[..., :k], dim=-1) with s_0 = +inf (empty prefix).
    # Using logcumsumexp enables computing softmin over the *first k* items for all k.
    z = -a_w / tau                               # (..., L, W)
    lse_prefix = torch.logcumsumexp(z, dim=-1)   # (..., L, W)
    prefix_softmin = -tau * lse_prefix           # softmin over first k+1 elements
    # shift to align: for k = 0 we need +inf; for k >=1 use prefix over first k elements
    plus_inf = torch.full_like(prefix_softmin[..., :1], float("inf"))
    a_prefix_to_k_minus_1 = torch.cat([plus_inf, prefix_softmin[..., :-1]], dim=-1)  # (..., L, W)

    # Candidate robustness for each k: min(b[t+k], min_{τ<t+k} a[τ])
    cand = softmin(torch.stack([b_w, a_prefix_to_k_minus_1], dim=-1), temp=temp, dim=-1)  # (..., L, W)

    # Mask out k < lo
    if lo > 0:
        mask = torch.arange(W, device=cand.device) < lo
        cand = cand.masked_fill(mask.view(*([1] * (cand.ndim - 1)), -1), float("-inf"))

    # Outer max over k in [lo, hi]
    out = tau * torch.logsumexp(cand / tau, dim=-1)  # (..., L)

    if keepdim:
        out = out.unsqueeze(-1)
    if old_dim != a_m.ndim - 1:
        out = out.movedim(-1, old_dim)
    return out


def release_window(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    lo: int = 0,
    hi: int | None = None,
    temp: float = 0.1,
    time_dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Bounded 'release'  (a R_[lo,hi] b).

    Defined via duality:  a R b ≡ ¬( ¬a U ¬b ).
    """
    return soft_not(until_window(soft_not(a), soft_not(b), lo=lo, hi=hi, temp=temp, time_dim=time_dim, keepdim=keepdim))


def shift_left(x: torch.Tensor, *, steps: int, time_dim: int = -1, pad_value: float = float("nan")) -> torch.Tensor:
    """Shift the signal left (toward the future), padding the tail with ``pad_value``."""
    if steps < 0:
        raise ValueError("steps must be non-negative")
    x_m, old_dim = _move_time_last(x, time_dim)
    T = x_m.shape[-1]
    steps = min(int(steps), T)
    if steps == 0:
        out = x_m
    else:
        tail = torch.full_like(x_m[..., :steps], fill_value=pad_value)
        out = torch.cat([x_m[..., steps:], tail], dim=-1)
    if old_dim != x_m.ndim - 1:
        out = out.movedim(-1, old_dim)
    return out


# ---------------------------------------------------------------------------
# Past-time operators (useful for online/streaming monitoring)
# ---------------------------------------------------------------------------

def _flip_time(x: torch.Tensor, time_dim: int = -1) -> torch.Tensor:
    x_m, old = _move_time_last(x, time_dim)
    flipped = torch.flip(x_m, dims=[-1])
    if old != x_m.ndim - 1:
        flipped = flipped.movedim(-1, old)
    return flipped


def once_window(margins: torch.Tensor, *, window: int, temp: float = 0.1, time_dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """Past 'once' (◇⁻): apply eventually on the reversed time axis over a window."""
    rev = _flip_time(margins, time_dim=time_dim)
    y = eventually_window(rev, window=window, temp=temp, time_dim=time_dim, keepdim=keepdim)
    # result is aligned to reversed time; flip back along the produced axis
    return _flip_time(y, time_dim=time_dim if keepdim else -1)


def historically_window(margins: torch.Tensor, *, window: int, temp: float = 0.1, time_dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """Past 'historically' (□⁻): apply always on the reversed time axis over a window."""
    rev = _flip_time(margins, time_dim=time_dim)
    y = always_window(rev, window=window, temp=temp, time_dim=time_dim, keepdim=keepdim)
    return _flip_time(y, time_dim=time_dim if keepdim else -1)


# ---------------------------------------------------------------------------
# Loss: drive robustness positive (violations → penalty)
# ---------------------------------------------------------------------------

@dataclass
class STLPenaltyConfig:
    weight: float = 1.0
    margin: float = 0.0            # desire robustness >= margin
    kind: str = "softplus"         # {'softplus', 'hinge', 'sqhinge', 'logistic'}
    beta: float = 10.0             # sharpness for 'softplus' / 'logistic'
    reduction: str = "mean"        # {'mean', 'sum', 'none'}


class STLPenalty(torch.nn.Module):

    def __init__(
        self,
        weight: float = 1.0,
        margin: float = 0.0,
        *,
        kind: str = "softplus",
        beta: float = 10.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.register_buffer("margin", torch.as_tensor(float(margin)))
        self.weight = float(weight)
        self.kind = str(kind).lower()
        self.beta = float(beta)
        self.reduction = str(reduction).lower()
        if self.kind not in {"softplus", "hinge", "sqhinge", "logistic"}:
            raise ValueError(f"Unsupported kind: {kind}")
        if self.reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")

    def forward(self, robustness: torch.Tensor) -> torch.Tensor:
        delta = self.margin - robustness  # positive when violating desired margin
        if self.kind == "softplus":
            loss = torch.nn.functional.softplus(self.beta * delta) / self.beta
        elif self.kind == "logistic":
            # numerically identical to softplus above; kept for explicitness
            loss = torch.log1p(torch.exp(self.beta * delta)) / self.beta
        elif self.kind == "hinge":
            loss = torch.clamp(delta, min=0.0)
        elif self.kind == "sqhinge":
            loss = torch.clamp(delta, min=0.0).square()
        else:  # pragma: no cover
            raise AssertionError("unreachable")

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        # 'none' returns elementwise

        return loss * self.weight


__all__ = [
    # aggregations
    "softmin",
    "softmax",
    "soft_and",
    "soft_or",
    "soft_not",
    "soft_implies",
    # predicates
    "pred_leq",
    "pred_geq",
    "pred_abs_leq",
    "pred_linear_leq",
    # temporal (future)
    "always",
    "eventually",
    "always_window",
    "eventually_window",
    "until_window",
    "release_window",
    "shift_left",
    # temporal (past)
    "once_window",
    "historically_window",
    # penalty
    "STLPenalty",
    "STLPenaltyConfig",
]

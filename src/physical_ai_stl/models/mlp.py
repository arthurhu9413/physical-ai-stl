from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["MLP", "Sine"]


# --- small utilities ---------------------------------------------------------------------------

class Sine(nn.Module):
    """Sine activation with configurable frequency multiplier ``w0``.

    This is useful for SIREN-style coordinate networks in PINNs/NODEs.
    """
    def __init__(self, w0: float = 1.0) -> None:
        super().__init__()
        self.w0 = float(w0)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        return torch.sin(self.w0 * x)

    def extra_repr(self) -> str:
        return f"w0={self.w0:g}"


ActivationArg = str | nn.Module | Callable[[], nn.Module]


def _normalize_activation(act: ActivationArg) -> Callable[[], nn.Module]:
    """Return a 0-arg constructor from a flexible activation spec."""
    if isinstance(act, str):
        name = act.lower()
        if name in {"tanh"}:
            return nn.Tanh
        if name in {"relu"}:
            return lambda: nn.ReLU(inplace=False)
        if name in {"silu", "swish"}:
            return nn.SiLU
        if name in {"gelu"}:
            return nn.GELU
        if name in {"sigmoid"}:
            return nn.Sigmoid
        if name in {"sine", "sin", "siren"}:
            return Sine  # default w0=1.0 (see init below for SIREN)
        raise ValueError(f"Unknown activation string: {act!r}")
    if isinstance(act, nn.Module):
        return act.__class__
    if callable(act):
        return act
    raise TypeError("activation must be a string, nn.Module, or a 0-arg callable returning an nn.Module.")


# --- model -------------------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Flexible, training-stable multilayer perceptron tailored for physics/ML tasks.

    Highlights
    ----------
    • **STL-friendly hooks**: optional output range mapping and ergonomic parameter counting.
    • **PINN-ready**: supports SIREN init for sinusoidal activations; safe *auto* init.
    • **Configurable**: optional LayerNorm/BatchNorm, dropout, skip concatenations,
      and lightweight residual connections when shapes match.
    • **Ergonomics**: dtype/device wiring; weight norm opt-in; robust shape checks.

    Parameters
    ----------
    in_dim, out_dim : int
        Input and output feature sizes (> 0).
    hidden : Sequence[int]
        Sizes of hidden layers (must be non-empty, values > 0).
    activation : ActivationArg
        Hidden activation. Accepts a string ("tanh", "relu", "silu", "gelu", "sigmoid", "sine"),
        an ``nn.Module`` instance, or a zero-arg constructor.
    out_activation : ActivationArg | None
        Optional activation applied after the output linear layer.
    bias : bool
        If True, use bias in linear layers.
    init : str
        One of {"auto", "xavier", "kaiming", "siren"}. "auto" selects a safe default based on
        the activation ("tanh"→xavier, relu/silu/gelu→kaiming, sine→siren).
    last_layer_scale : float | None
        Optionally scales the output layer *weights* by this factor post-init.
    skip_connections : Sequence[int]
        Indices of hidden layers where the raw input is concatenated to the layer input.
        (Use ``range(len(hidden))`` to concatenate the input at every layer.)
    weight_norm : bool
        If True, wrap all linear layers with weight normalization.
    dtype, device : torch dtype/device
        Passed through to layer constructors.
    norm : str | None
        Optional per-layer normalization: {"layer", "batch"}.  ``None`` to disable.
        LayerNorm is typically the safer choice for small-batch PINN settings.
    dropout : float | Sequence[float]
        Dropout probability(s) applied *after* the activation of each hidden layer. 0 disables.
    residual : bool
        If True, enable lightweight residual connections **when** the (possibly skip-augmented)
        input dim equals the layer output dim. Mismatched shapes automatically disable residual.
    checkpoint : bool
        If True, use ``torch.utils.checkpoint.checkpoint`` for hidden blocks during training
        to reduce activation memory (useful for deeper MLPs). Has no effect in eval mode.
    out_range : tuple[float, float] | None
        If given, linearly squashes the final output to this range via a ``tanh`` mapping:
            y = low + (high - low) * (tanh(y) + 1)/2
        Applied *after* ``out_activation`` (if any).

    Notes
    -----
    The forward signature remains ``(x: Tensor) -> Tensor`` to avoid breaking callers.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int] = (64, 64, 64),
        activation: ActivationArg = "tanh",
        *,
        out_activation: ActivationArg | None = None,
        bias: bool = True,
        init: str = "auto",
        last_layer_scale: float | None = None,
        skip_connections: Sequence[int] = (),
        weight_norm: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        # New, backward-compatible options
        norm: str | None = None,
        dropout: float | Sequence[float] = 0.0,
        residual: bool = False,
        checkpoint: bool = False,
        out_range: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()

        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("in_dim and out_dim must be positive integers.")
        if not isinstance(hidden, Sequence) or len(hidden) == 0:
            raise ValueError("hidden must be a non-empty sequence of positive integers.")
        if any(h <= 0 for h in hidden):
            raise ValueError("All hidden layer sizes must be positive.")

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden = tuple(int(h) for h in hidden)
        # allow an ergonomic alias for "all" layers via slice-like semantics
        if isinstance(skip_connections, Sequence):
            self.skip_connections = tuple(int(i) for i in skip_connections)
        else:  # pragma: no cover (defensive)
            raise TypeError("skip_connections must be a sequence of layer indices.")
        self.checkpoint = bool(checkpoint)
        self.residual = bool(residual)

        # Normalization config
        norm_name = (None if norm is None else str(norm).lower())
        if norm_name not in {None, "layer", "batch"}:
            raise ValueError("norm must be one of {None, 'layer', 'batch'}.")
        self._norm_kind = norm_name

        # Dropout config -> per-layer list
        if isinstance(dropout, Sequence) and not isinstance(dropout, (str, bytes)):
            if len(dropout) not in {1, len(self.hidden)}:
                raise ValueError("dropout sequence must have length 1 or len(hidden).")
            dropouts = list(float(p) for p in dropout)
            if len(dropouts) == 1:
                dropouts = dropouts * len(self.hidden)
        else:
            drop = float(dropout)
            dropouts = [drop] * len(self.hidden)
        if any(p < 0 or p >= 1 for p in dropouts):
            raise ValueError("dropout probabilities must be in [0, 1).")
        self._drop_probs = tuple(dropouts)

        # Output range mapping (registered buffers to be script/compile-friendly)
        if out_range is not None:
            low, high = float(out_range[0]), float(out_range[1])
            if not (high > low):
                raise ValueError("out_range must satisfy high > low.")
            # We keep these as buffers so they move with .to(device/dtype)
            self.register_buffer("_out_low", torch.tensor(low), persistent=False)
            self.register_buffer("_out_scale", torch.tensor(high - low), persistent=False)
        else:
            self.register_buffer("_out_low", torch.tensor(0.0), persistent=False)
            self.register_buffer("_out_scale", torch.tensor(1.0), persistent=False)
        self._use_out_range = out_range is not None

        act_ctor = _normalize_activation(activation)
        out_act_ctor = _normalize_activation(out_activation) if out_activation is not None else None

        # Helpers ----------------------------------------------------------------
        def maybe_weight_norm(linear: nn.Linear) -> nn.Linear:
            return nn.utils.weight_norm(linear) if weight_norm else linear

        def make_norm(dim: int) -> nn.Module | None:
            if self._norm_kind is None:
                return None
            if self._norm_kind == "layer":
                return nn.LayerNorm(dim)
            # batch norm over features; expects (N, C)
            return nn.BatchNorm1d(dim)

        # Build layers with optional skip concatenations and per-layer norms/dropouts
        layers: list[nn.Module] = []
        acts: list[nn.Module] = []
        norms: list[nn.Module | None] = []
        drops: list[nn.Module | None] = []
        residual_flags: list[bool] = []

        last_dim = self.in_dim
        for idx, h in enumerate(self.hidden):
            in_d = last_dim + (self.in_dim if idx in self.skip_connections else 0)
            lin = nn.Linear(in_d, h, bias=bias, dtype=dtype, device=device)
            layers.append(maybe_weight_norm(lin))
            norms.append(make_norm(h))
            acts.append(act_ctor())
            drops.append(nn.Dropout(self._drop_probs[idx]) if self._drop_probs[idx] > 0 else None)
            # enable residual only when shapes match and no concatenative skip at this layer
            residual_flags.append(bool(self.residual and (in_d == h) and (idx not in self.skip_connections)))
            last_dim = h

        # Output layer
        self.out = maybe_weight_norm(nn.Linear(last_dim, self.out_dim, bias=bias, dtype=dtype, device=device))

        # Register module lists
        self.layers = nn.ModuleList(layers)
        self.acts = nn.ModuleList(acts)
        self.norms = nn.ModuleList([n if n is not None else nn.Identity() for n in norms])
        self._has_norm = any(n is not None for n in norms)
        self.dropouts = nn.ModuleList([d if d is not None else nn.Identity() for d in drops])
        self._has_dropout = any(d is not None for d in drops)
        self._residual_flags = tuple(residual_flags)
        self.out_act = out_act_ctor() if out_act_ctor is not None else None

        # Initialization
        self.reset_parameters(init=init, activation=act_ctor, last_layer_scale=last_layer_scale)

    # --- initialization ------------------------------------------------------------------------

    @torch.no_grad()
    def reset_parameters(
        self,
        *,
        init: str = "auto",
        activation: Callable[[], nn.Module] | None = None,
        last_layer_scale: float | None = None,
    ) -> None:
        init = str(init).lower()

        # Inspect activation type if available for auto mode
        act_name = None
        if activation is not None:
            try:
                act_name = activation().__class__.__name__.lower()
            except Exception:
                act_name = None

        def xavier_(m: nn.Linear, gain: float = 1.0) -> None:
            nn.init.xavier_uniform_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        def kaiming_(m: nn.Linear, *, nonlinearity: str = "relu") -> None:
            nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        def siren_(first: nn.Linear | None, rest: Sequence[nn.Linear], w0: float = 30.0) -> None:
            if first is not None:
                in_d = first.weight.shape[1]
                bound = 1.0 / in_d
                nn.init.uniform_(first.weight, -bound, bound)
                if first.bias is not None:
                    nn.init.uniform_(first.bias, -bound, bound)
            for m in rest:
                in_d = m.weight.shape[1]
                # SIREN paper: U(-sqrt(6/in)/w0, sqrt(6/in)/w0)
                bound = (6.0 / in_d) ** 0.5 / w0
                nn.init.uniform_(m.weight, -bound, bound)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Choose scheme
        if init == "auto":
            if act_name and ("relu" in act_name or "silu" in act_name or "gelu" in act_name):
                chosen = "kaiming"
            elif act_name and ("tanh" in act_name or "sigmoid" in act_name):
                chosen = "xavier"
            elif act_name and ("sine" in act_name):
                chosen = "siren"
            else:
                chosen = "xavier"  # safe default
        else:
            chosen = init

        # Apply
        if chosen == "xavier":
            # use tanh gain if likely; otherwise 1.0
            gain = nn.init.calculate_gain("tanh") if (act_name and "tanh" in act_name) else 1.0
            for m in list(self.layers) + [self.out]:
                xavier_(m, gain=gain)
        elif chosen == "kaiming":
            # choose nonlinearity based on activation if known
            nonlin = "relu"
            if act_name and ("gelu" in act_name):
                nonlin = "relu"  # GELU uses ReLU fan-in in practice
            for m in list(self.layers) + [self.out]:
                kaiming_(m, nonlinearity=nonlin)
        elif chosen == "siren":
            first = self.layers[0] if len(self.layers) > 0 else None
            rest = [m for i, m in enumerate(self.layers) if i != 0]
            w0 = self._infer_siren_w0()
            siren_(first, rest, w0=w0)
            # Output layer: small init (as in SIREN examples)
            nn.init.uniform_(self.out.weight, -1e-4, 1e-4)
            if self.out.bias is not None:
                nn.init.zeros_(self.out.bias)
        else:
            raise ValueError(f"Unknown init scheme: {init!r}")

        if last_layer_scale is not None and last_layer_scale > 0:
            self.out.weight.mul_(float(last_layer_scale))

    def _infer_siren_w0(self) -> float:
        # Look for Sine activations and pick w0 from the first one if present.
        for a in self.acts:
            if isinstance(a, Sine):
                # default to 30.0 as in the SIREN paper if left at constructor default
                return float(getattr(a, "w0", 30.0)) or 30.0
        return 30.0

    # --- forward -------------------------------------------------------------------------------

    def _block(self, h: Tensor, x0: Tensor, idx: int) -> Tensor:
        """One hidden block: optional concat-skip, Linear -> Norm? -> (+res) -> Act -> Dropout."""
        hx = h
        if idx in self.skip_connections:
            hx = torch.cat((hx, x0), dim=-1)
        z = self.layers[idx](hx)
        if self._has_norm:
            z = self.norms[idx](z)
        if self._residual_flags[idx]:
            # shape guaranteed to match by construction when residual flag is True
            z = z + h
        y = self.acts[idx](z)
        if self._has_dropout:
            y = self.dropouts[idx](y)
        return y

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"Expected input with last dim {self.in_dim}, got {tuple(x.shape)}.")

        h = x
        if self.checkpoint and self.training:  # memory-efficient mode
            from torch.utils.checkpoint import checkpoint as _ckpt  # local import to avoid overhead
            for idx in range(len(self.layers)):
                h = _ckpt(lambda _h, _x0, _i=idx: self._block(_h, _x0, _i), h, x)
        else:
            for idx in range(len(self.layers)):
                h = self._block(h, x, idx)

        y = self.out(h)
        if self.out_act is not None:
            y = self.out_act(y)
        if self._use_out_range:
            # y <- low + scale * (tanh(y)+1)/2
            y = self._out_low + self._out_scale * (torch.tanh(y).add(1.0).mul_(0.5))
        return y

    # --- ergonomics ----------------------------------------------------------------------------

    def count_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        act_names = [a.__class__.__name__ for a in self.acts[:2]]
        act = act_names[0] + ("..." if len(self.acts) > 1 and len(set(act_names)) == 1 else "")
        skips = f", skip={self.skip_connections}" if self.skip_connections else ""
        opts = []
        if self._norm_kind is not None:
            opts.append(f"norm={self._norm_kind}")
        if self._has_dropout and any(self._drop_probs):
            opts.append(f"dropout={self._drop_probs}")
        if self.residual:
            opts.append("residual=True")
        if self._use_out_range:
            opts.append("out_range=True")
        opt_str = (", " + ", ".join(opts)) if opts else ""
        return f"in={self.in_dim}, out={self.out_dim}, hidden={self.hidden}, act={act}{skips}{opt_str}"

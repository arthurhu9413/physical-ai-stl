# tests/test_stl_soft.py
# High-signal unit tests for differentiable STL "soft" semantics.
# These tests emphasize numerical soundness, gradients, and API ergonomics.
from __future__ import annotations

import math
import pathlib
import pytest

# Skip these tests if PyTorch is not installed in the environment
torch = pytest.importorskip("torch")


# Support running tests from a source checkout without installation
try:  # pragma: no cover - import convenience
    from physical_ai_stl.monitoring import stl_soft as stl  # type: ignore
except Exception:  # pragma: no cover - import convenience
    ROOT = pathlib.Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if SRC.exists():
        import sys
        sys.path.insert(0, str(SRC))
    from physical_ai_stl.monitoring import stl_soft as stl  # type: ignore  # noqa: E402


def _devices():
    yield torch.device("cpu")
    if torch.cuda.is_available():
        yield torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # PyTorch MPS (Apple Silicon) sometimes has different numeric kernels.
        # Only use it for dtype/device propagation (not strict value checks).
        yield torch.device("mps")


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
def test_pred_leq_basic_and_broadcast():
    u = torch.tensor([0.1, 0.4, 0.5], dtype=torch.float32)
    c = 0.5
    margins = stl.pred_leq(u, c)
    expected = torch.tensor([0.4, 0.1, 0.0], dtype=torch.float32)
    assert torch.allclose(margins, expected, atol=1e-8)

    # Broadcasting: (2 x 3) - scalar
    U = torch.tensor([[0.0, 1.0, 2.0],
                      [0.5, 0.5, 0.5]], dtype=torch.float32)
    C = 1.0
    M = stl.pred_leq(U, C)
    EXP = torch.tensor([[1.0, 0.0, -1.0],
                        [0.5, 0.5, 0.5]], dtype=torch.float32)
    assert torch.allclose(M, EXP, atol=1e-8)
    # Dtype/device preservation
    assert M.dtype == U.dtype and M.device == U.device


def test_pred_leq_broadcast_vector_threshold():
    # Broadcasting against a per-feature threshold vector
    U = torch.tensor([[0.0, 1.0, 2.0],
                      [2.5, 2.0, 1.5]], dtype=torch.float64)
    C = torch.tensor([1.0, 2.0, 1.5], dtype=torch.float64)  # shape (3,)
    M = stl.pred_leq(U, C)  # expected (2,3) via broadcasting
    EXP = torch.tensor([[1.0, 1.0, -0.5],
                        [-1.5, 0.0, 0.0]], dtype=torch.float64)
    assert torch.allclose(M, EXP, atol=1e-12)
    assert M.shape == U.shape and M.dtype == U.dtype and M.device == U.device


@pytest.mark.parametrize("fn", [stl.softmin, stl.softmax])
def test_softmin_softmax_dtype_and_device(fn):
    for dev in _devices():
        x = torch.tensor([0.2, 0.8, 0.5], dtype=torch.float32, device=dev, requires_grad=True)
        y = fn(x, temp=0.1, dim=-1)
        # Reduction preserves dtype/device and produces a scalar along reduced dim
        assert y.dtype == x.dtype and y.device == x.device
        assert y.shape == () or y.shape == torch.Size([])
        # Must be connected to the graph
        (g,) = torch.autograd.grad(y, x, retain_graph=False, allow_unused=False)
        assert g is not None and torch.isfinite(g).all()


def test_softmin_softmax_bounds_and_singleton():
    x = torch.tensor([0.2, 0.8, 0.5], dtype=torch.float32)
    # With a small temperature they should be tight lower/upper bounds
    sm = stl.softmin(x, temp=0.05)
    sM = stl.softmax(x, temp=0.05)
    assert sm.item() <= x.min().item() + 1e-6
    assert sM.item() >= x.max().item() - 1e-6

    # Singletons act as identity for any temperature
    y = torch.tensor([0.42], dtype=torch.float32)
    for t in (1e-3, 0.1, 1.0):
        assert torch.allclose(stl.softmin(y, temp=t), y, atol=1e-8)
        assert torch.allclose(stl.softmax(y, temp=t), y, atol=1e-8)


def test_softmin_softmax_gradients_are_soft_weights():
    # For f(x) = T * logsumexp(x/T), ∂f/∂x = softmax(x/T)
    # For g(x) = -T * logsumexp(-x/T), ∂g/∂x = softmin_weights(x/T)
    x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64, requires_grad=True)
    T = 0.25
    f = stl.softmax(x, temp=T)
    (grad_f,) = torch.autograd.grad(f, x, create_graph=False)
    # Probabilities: in [0,1] and sum to 1
    assert torch.all(grad_f >= 0) and torch.all(grad_f <= 1)
    assert torch.allclose(grad_f.sum(), torch.tensor(1.0, dtype=grad_f.dtype), atol=1e-12)
    # Mass should concentrate on the max entry for small T
    assert grad_f.argmax().item() == x.argmax().item()

    x2 = torch.tensor([2.0, 1.0, 0.0], dtype=torch.float64, requires_grad=True)
    g = stl.softmin(x2, temp=T)
    (grad_g,) = torch.autograd.grad(g, x2, create_graph=False)
    assert torch.all(grad_g >= 0) and torch.all(grad_g <= 1)
    assert torch.allclose(grad_g.sum(), torch.tensor(1.0, dtype=grad_g.dtype), atol=1e-12)
    assert grad_g.argmax().item() == x2.argmin().item()  # concentrates at the min


def test_softmin_softmax_temperature_sharpness():
    # Smaller temperature -> closer to hard min/max
    x = torch.tensor([-1.0, 0.0, 3.0, 0.5], dtype=torch.float64)
    Tmin, Tmid = 0.01, 0.25
    hard_min, hard_max = x.min(), x.max()
    sm_min = stl.softmin(x, temp=Tmin)
    sM_min = stl.softmax(x, temp=Tmin)
    sm_mid = stl.softmin(x, temp=Tmid)
    sM_mid = stl.softmax(x, temp=Tmid)
    assert abs(sm_min.item() - hard_min.item()) <= abs(sm_mid.item() - hard_min.item()) + 1e-12
    assert abs(sM_min.item() - hard_max.item()) <= abs(sM_mid.item() - hard_max.item()) + 1e-12


def test_numerical_stability_small_temperature():
    # Large dynamic range should not cause inf/nan thanks to log-sum-exp
    x = torch.linspace(-50.0, 50.0, steps=101, dtype=torch.float32)
    for fn in (stl.softmin, stl.softmax):
        y = fn(x, temp=1e-3)
        assert torch.isfinite(y).all(), "Output contains inf/nan for very small temperature"


# ---------------------------------------------------------------------------
# Temporal operators
# ---------------------------------------------------------------------------
def test_temporal_always_eventually_reduce_over_time_dim():
    # margins shape: (batch, time)
    margins = torch.tensor([[ 0.0,  1.0, -1.0,  2.0],
                            [-2.0, -0.5,  0.5,  3.0]], dtype=torch.float32)
    # Default time_dim = -1
    G = stl.always(margins, temp=0.05)      # approx min over time
    F = stl.eventually(margins, temp=0.05)  # approx max over time
    assert G.shape == (2,) and F.shape == (2,)
    # They should bound the true min/max
    true_min = margins.min(dim=-1).values
    true_max = margins.max(dim=-1).values
    assert torch.all(G <= true_min + 1e-6)
    assert torch.all(F >= true_max - 1e-6)

    # Explicitly specify the time dimension
    G0 = stl.always(margins.t(), temp=0.05, time_dim=0)
    F0 = stl.eventually(margins.t(), temp=0.05, time_dim=0)
    assert torch.allclose(G0, G, atol=1e-6) and torch.allclose(F0, F, atol=1e-6)


def test_temporal_ops_singleton_time_axis():
    # If there is only one time step, both operators are identity
    margins = torch.tensor([[0.3], [-1.2]], dtype=torch.float32)  # shape (B, 1)
    G = stl.always(margins, time_dim=-1)
    F = stl.eventually(margins, time_dim=-1)
    assert torch.allclose(G.squeeze(-1), margins.squeeze(-1), atol=1e-8)
    assert torch.allclose(F.squeeze(-1), margins.squeeze(-1), atol=1e-8)


def test_always_eventually_match_softmin_softmax_on_dim():
    # Check exact equivalence to softmin/softmax when called over the same time_dim.
    B, T, D = 2, 5, 3
    x = torch.arange(B*T*D, dtype=torch.float64).reshape(B, T, D) / 7.0 - 4.0
    # time_dim=1 -> outputs (B, D)
    G = stl.always(x, temp=0.1, time_dim=1)
    F = stl.eventually(x, temp=0.1, time_dim=1)
    G_ref = stl.softmin(x, temp=0.1, dim=1)
    F_ref = stl.softmax(x, temp=0.1, dim=1)
    assert torch.allclose(G, G_ref, atol=1e-12)
    assert torch.allclose(F, F_ref, atol=1e-12)
    assert G.shape == (B, D) and F.shape == (B, D)


def test_temporal_gradients_sum_to_one_over_time():
    # Gradients of the log-sum-exp smoothed temporal ops should be soft weights
    x = torch.tensor([[ -1.0,  0.0,  2.0, -0.5,  1.0 ]], dtype=torch.float64, requires_grad=True)  # (1,T)
    Ttemp = 0.2

    # Eventually ~ softmax over time (dim=-1)
    F = stl.eventually(x, temp=Ttemp)  # shape (1,)
    (gF,) = torch.autograd.grad(F, x, retain_graph=True)
    # gF is (1,T) of non-negative weights that sum to 1 along time
    assert torch.all(gF >= 0) and torch.all(gF <= 1)
    assert torch.allclose(gF.sum(dim=-1), torch.ones_like(F), atol=1e-12)

    # Always ~ softmin over time (dim=-1)
    G = stl.always(x, temp=Ttemp)
    (gG,) = torch.autograd.grad(G, x, retain_graph=False)
    assert torch.all(gG >= 0) and torch.all(gG <= 1)
    assert torch.allclose(gG.sum(dim=-1), torch.ones_like(G), atol=1e-12)


def test_temporal_ops_handle_noncontiguous_tensors():
    # Use transposes/slices to produce non-contiguous inputs
    X = torch.randn(3, 8, 5, dtype=torch.float32)
    X_nc = X.transpose(0, 1)[1:, :, :]  # shape (7,3,5), non-contiguous
    # Reduce over time_dim=0 (first axis of X_nc), should preserve other dims
    G = stl.always(X_nc, temp=0.123, time_dim=0)
    F = stl.eventually(X_nc, temp=0.123, time_dim=0)
    assert G.shape == (3, 5) and F.shape == (3, 5)
    # Reference using softmin/softmax along same dim
    G_ref = stl.softmin(X_nc, temp=0.123, dim=0)
    F_ref = stl.softmax(X_nc, temp=0.123, dim=0)
    assert torch.allclose(G, G_ref, atol=1e-7)
    assert torch.allclose(F, F_ref, atol=1e-7)


# ---------------------------------------------------------------------------
# Penalty module
# ---------------------------------------------------------------------------
def test_stlpenalty_behavior_and_margin_shift():
    # For large positive robustness, penalty ~ 0; for large negative, penalty grows.
    penalty = stl.STLPenalty(weight=1.0, margin=0.0)
    high = torch.tensor([10.0], dtype=torch.float64, requires_grad=True)
    low  = torch.tensor([-10.0], dtype=torch.float64, requires_grad=True)
    v_high = penalty(high)
    v_low  = penalty(low)
    assert v_high.item() < 1e-6
    assert v_low.item()  > 1.0

    # Monotonic decreasing in robustness
    r = torch.tensor([-2.0, -1.0, 0.0, 1.0], dtype=torch.float64)
    vals = penalty(r)
    assert torch.all(vals[:-1] >= vals[1:])

    # Weight = 0 short-circuits to 0 and blocks gradients
    penalty_zero = stl.STLPenalty(weight=0.0, margin=0.0)
    z = torch.tensor([-1.0, 1.0], dtype=torch.float64, requires_grad=True)
    out = penalty_zero(z)
    assert float(out) == 0.0
    (g,) = torch.autograd.grad(out, z, retain_graph=False, allow_unused=True)
    # grad can be None if path is fully blocked; treat that as zero
    if g is not None:
        assert torch.allclose(g, torch.zeros_like(z))

    # Margin acts as a shift: at robustness == margin, softplus(0) = log(2)
    # Use beta=1.0 so softplus(0)/beta = log(2)
    p = stl.STLPenalty(weight=1.0, margin=1.0, beta=1.0)
    val_at_margin = p(torch.tensor([1.0], dtype=torch.float64))
    assert abs(val_at_margin.item() - math.log(2.0)) < 1e-6


def test_stlpenalty_mean_reduction_and_grad_sign():
    # The module uses mean() reduction and multiplies by weight.
    w, m, b = 3.5, -0.25, 7.0
    penalty = stl.STLPenalty(weight=w, margin=m, beta=b)
    r = torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float64, requires_grad=True)
    out = penalty(r)
    # Expected: w * mean( softplus(b * (m - r_i)) / b )
    expected = w * (torch.nn.functional.softplus(b * (m - r)) / b).mean()
    assert torch.allclose(out, expected, atol=1e-12)

    # Gradients should be negative (increasing robustness lowers penalty):
    # d/dr softplus(b*(m - r))/b = -sigmoid(b*(m - r))
    (g,) = torch.autograd.grad(out, r, retain_graph=False)
    assert torch.all(g <= 0.0 + 1e-12)
    # And magnitude bounded by w / N (since |sigmoid|<=1 and mean over N)
    N = r.numel()
    assert torch.all(torch.abs(g) <= w / N + 1e-6)

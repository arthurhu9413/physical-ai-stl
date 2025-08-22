from __future__ import annotations

"""Minimal, dependency-friendly spatial monitoring demo.

This module builds a toy 2D scene (a moving circular *agent* and a static
circular *goal*) and evaluates a **spatial–temporal** property using the
[SpaTiaL] library when available:

    F[0, T-1] ( distance(agent, goal) <= eps )

with *quantitative* semantics (i.e., real-valued robustness).  When SpaTiaL is
not installed, or if the formula cannot be parsed in the local SpaTiaL version,
we fall back to a **closed-form robustness** computation that mirrors SpaTiaL's
semantics for the `distance(·,·) <= eps` predicate.

Design goals
------------
1) **Meets professor's requirement**: demonstrate STL/STREL-style monitoring
   over a physicsy toy scene and return a quantitative robustness value.
2) **Robust across SpaTiaL versions**: accept grammar variants and avoid
   zero-epsilon pitfalls (SpaTiaL currently requires a *positive* `eps` in
   `distance_compare`).  We therefore use a tiny epsilon when the requested
   tolerance is zero.
3) **Fast and dependency-light**: import SpaTiaL lazily; provide a math-only
   fallback with O(1) evaluation time.

References: SpaTiaL API (modules `spatial.logic` and `spatial.geometry`).
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Optional

# Optional dependency: SpaTiaL (published on PyPI under the module name "spatial")
try:
    from spatial.logic import Spatial  # type: ignore
except Exception:  # pragma: no cover - optional dependency path
    Spatial = None  # type: ignore


@dataclass(slots=True)
class ToyScene:
    """Configuration for the toy scene.

    Attributes
    ----------
    T : int
        Number of discrete time steps in the horizon (evaluated over [0, T-1]).
    agent_speed : float
        Constant speed (m/step) of the agent moving along the +x axis.
    agent_radius : float
        Radius of the agent's circular footprint.
    goal_pos : tuple[float, float]
        Center position of the goal.
    goal_radius : float
        Radius of the goal's circular footprint.
    reach_eps : float
        Distance tolerance for the `distance(agent, goal) <= eps` predicate.
        If 0.0, we interpret it as *touching* in the geometric sense.  Since
        SpaTiaL's `distance_compare` currently asserts `eps > 0`, we replace
        0.0 by a tiny positive epsilon when constructing the SpaTiaL formula.
    """
    T: int = 50
    agent_speed: float = 0.35
    agent_radius: float = 0.30
    goal_pos: tuple[float, float] = (12.0, 0.0)
    goal_radius: float = 0.40
    reach_eps: float = 0.0  # 0.0 means "touching"


def build_scene(T: int = 50) -> ToyScene:
    return ToyScene(T=T)


# --------- Internal helpers (kept small & allocation-light) ----------------- #

def _tiny_eps() -> float:
    """Return a numerically safe, strictly positive epsilon (~1e-12).

    SpaTiaL's `distance_compare` asserts that `eps` is positive, not merely
    non-negative. We use a very small value instead of 0.0 when needed.
    """
    # Avoid importing numpy solely for this; float epsilon is enough here.
    return 1e-12


def _build_spatial_objects(cfg: ToyScene) -> Dict[str, Any]:
    """Construct SpaTiaL geometry objects for the scene.

    We import SpaTiaL *inside* this function so the module remains importable
    when SpaTiaL is not present (e.g., in minimal CI).
    """
    try:
        import numpy as np
        from spatial.geometry import Circle, DynamicObject, PolygonCollection, StaticObject  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "SpaTiaL not available – install with: `pip install spatial shapely lark-parser`\n"
            "If you do not wish to install SpaTiaL, call `evaluate_formula` and it "
            "will use an analytical fallback instead."
        ) from e

    agent = DynamicObject()
    # Build a single-circle collection per time step; this implements SpatialInterface.
    # For T=O(10^2), constructing fresh circles is fast and avoids shared-state bugs.
    for t in range(cfg.T):
        x = float(t) * float(cfg.agent_speed)
        center = np.array([x, 0.0], dtype=float)
        footprint = PolygonCollection({Circle(center, float(cfg.agent_radius))})
        agent.addObject(footprint, time=t)

    goal_center = np.array(cfg.goal_pos, dtype=float)
    goal_shape = PolygonCollection({Circle(goal_center, float(cfg.goal_radius))})
    goal = StaticObject(goal_shape)

    return {"agent": agent, "goal": goal}


def _first_parsed(sp: Any, candidates: List[str]) -> Tuple[str, Any]:
    """Return the first SpaTiaL formula (and parsed tree) that parses successfully.

    The SpaTiaL grammar has seen small changes across releases; we therefore
    try a handful of semantically equivalent spellings.
    """
    for s in candidates:
        try:
            tree = sp.parse(s)
            if tree is not None:
                return s, tree
        except Exception:
            # Try the next spelling/variant
            pass
    # If we get here, re-raise a helpful message showing the attempted spellings.
    raise ValueError(
        "Could not parse any SpaTiaL formula; tried variants:\n" + "\n".join(candidates)
    )


# -------------------- Analytical (no-dependency) fallback ------------------- #

def _analytical_robustness(cfg: ToyScene, eps: Optional[float] = None) -> float:
    """Closed-form robustness for `F[0,T-1] ( distance(agent, goal) <= eps )`.

    SpaTiaL (quantitative semantics) uses the robustness:
        sup_{t in [0,T-1]} ( eps - distance(agent(t), goal) )

    where `distance` is the *non-negative* separation between shapes (0 if
    touching/overlapping). For two discs with radii r_a, r_g, moving along x:

        sep(t) = max(0, |x_goal - v * t| - (r_a + r_g))

    The supremum over discrete times is achieved at the integer time closest
    to `x_goal / v` (clamped to [0, T-1]).
    """
    v = float(cfg.agent_speed)
    T = int(cfg.T)
    x_goal = float(cfg.goal_pos[0])
    r_sum = float(cfg.agent_radius + cfg.goal_radius)
    eps_val = float(cfg.reach_eps if eps is None else eps)
    if eps_val <= 0.0:
        eps_val = _tiny_eps()  # approximate "touching"

    # Handle degenerate speed
    if v <= 0.0 or T <= 0:
        # Agent does not move: separation is |x_goal| - r_sum (non-negative)
        center_d = abs(x_goal)
        sep = max(0.0, center_d - r_sum)
        return eps_val - sep

    # Candidate times near the closest approach, clamped to the horizon
    t_star = x_goal / v
    t_candidates = {int(t) for t in (round(t_star), int(t_star), int(t_star) - 1, int(t_star) + 1)}
    t_candidates = {max(0, min(T - 1, t)) for t in t_candidates}

    def sep_at(t: int) -> float:
        center_d = abs(x_goal - v * float(t))
        return max(0.0, center_d - r_sum)

    # Compute minimal separation over discrete t, then robustness = eps - min_sep
    min_sep = min(sep_at(t) for t in t_candidates)
    return eps_val - min_sep


# --------------------------- Public evaluation API -------------------------- #

def evaluate_formula(cfg: ToyScene) -> float:
    """Evaluate the spatial–temporal property over the scene and return robustness.

    If SpaTiaL is available, we parse and interpret the formula using its
    quantitative semantics. If not, or if interpretation fails (e.g., because
    the local SpaTiaL build uses a slightly different grammar), we gracefully
    fall back to the analytical computation above.

    Returns
    -------
    float
        Robustness value (>= 0 means the property is satisfied).
    """
    # If SpaTiaL is absent, use the analytical robustness.
    if Spatial is None:
        return _analytical_robustness(cfg)

    # Build variables for SpaTiaL
    vars_map = _build_spatial_objects(cfg)

    # Create a quantitative interpreter
    sp = Spatial(quantitative=True)  # type: ignore[arg-type]

    # Register variables with both internal interpreters for maximum compatibility
    for name, obj in vars_map.items():
        sp.assign_variable(name, obj)
    # Also update the temporal interpreter's view of variables (SpaTiaL ≥ 0.2)
    try:
        sp.update_variables(vars_map)  # type: ignore[attr-defined]
    except Exception:
        pass

    upper = cfg.T - 1
    # SpaTiaL requires eps > 0 in distance_compare; approximate touching if needed.
    eps = float(cfg.reach_eps)
    if not (eps > 0.0):  # handles NaN and <= 0
        eps = _tiny_eps()

    # A small set of grammar variants (bounded eventually + distance compare).
    # We order by *most precise quantitative semantics* first.
    candidates = [
        f"F[0, {upper}] ( distance(agent, goal) <= {eps} )",
        f"eventually [0, {upper}] ( distance(agent, goal) <= {eps} )",
        f"F[0,{upper}](distance(agent, goal) <= {eps})",
        f"eventually[0,{upper}](distance(agent, goal) <= {eps})",
    ]

    try:
        formula_str, tree = _first_parsed(sp, candidates)
        val = sp.interpret(tree, lower=0, upper=upper)  # type: ignore[arg-type]
        # SpaTiaL returns a float in quantitative mode
        return float(val)
    except Exception:
        # Fall back to a dependency-free analytical computation
        return _analytical_robustness(cfg, eps=eps)


def run_demo(T: int = 50) -> float:
    return evaluate_formula(ToyScene(T=T))


__all__ = ["ToyScene", "build_scene", "evaluate_formula", "run_demo"]

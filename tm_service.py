"""Shared forward prediction and FEM verification for the CLI and local demo."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla
import torch

from tm_core import FastBandSolver, Mn_pt_int, get_S1, get_T1, read_yaml

ROOT = Path(__file__).absolute().parent
DESIGN_DIM = 36
EPS_AIR = 1.0
EPS_CONTRAST = 7.9


def validate_design(values) -> np.ndarray:
    """Require one finite, normalized value for each p4m wedge cell."""
    try:
        design = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Design must contain 36 numeric values in [0, 1].") from error
    if (
        design.shape != (DESIGN_DIM,)
        or not np.isfinite(design).all()
        or np.any((design < 0) | (design > 1))
    ):
        raise ValueError("Design must be exactly 36 finite values in [0, 1].")
    return design


def example_vector() -> np.ndarray:
    """A deterministic illustrative pattern, independent of saved experiments."""
    return np.asarray(
        [float(row * row + col * col <= 25) for row in range(8) for col in range(row, 8)],
        dtype=np.float64,
    )


def load_solver(device: str = "cpu") -> FastBandSolver:
    """Load the published checkpoint and FEM assets once."""
    cfg = read_yaml(ROOT / "config.yaml")
    for key in ("basis_path", "matrices_path", "utilities_path", "weight_path"):
        path = ROOT / "assets" / Path(cfg[key]).name
        if not path.is_file():
            raise FileNotFoundError(f"Required demo asset is missing: {path}")
        cfg[key] = str(path)
    return FastBandSolver(config=cfg, batch_size=1, device=device)


def permittivity(design) -> np.ndarray:
    return EPS_AIR + EPS_CONTRAST * validate_design(design)


def k_path_ticks(solver: FastBandSolver) -> list[int]:
    """Locate Γ–X–M turns; the final Γ is one unsampled endpoint."""
    k_points = solver.K_raw.detach().cpu().numpy()
    steps = np.diff(k_points, axis=0)
    directions = steps / np.linalg.norm(steps, axis=1, keepdims=True)
    turns = np.flatnonzero(
        np.linalg.norm(np.diff(directions, axis=0), axis=1) > 1e-4
    ) + 1
    if len(turns) != 2:
        raise RuntimeError("Expected a three-segment Γ–X–M–Γ k path.")
    return [0, *(int(index) for index in turns), len(k_points)]


def sample_k_indices(solver: FastBandSolver, k_points: int) -> list[int]:
    """Sample each path segment evenly without changing its endpoints."""
    if isinstance(k_points, bool) or k_points not in range(9, 31, 3):
        raise ValueError("k_points must be a multiple of 3 between 9 and 30.")
    ticks = k_path_ticks(solver)
    per_segment = k_points // 3
    if any(end - start < per_segment for start, end in zip(ticks, ticks[1:])):
        raise ValueError("The stored k path has too few points for this selection.")
    return [
        start + round(offset * (end - start) / per_segment)
        for start, end in zip(ticks, ticks[1:])
        for offset in range(per_segment)
    ]


def predict_bands(
    design, solver: FastBandSolver, k_indices: list[int] | None = None
) -> tuple[np.ndarray, float]:
    """Return [Nk, Neig] band frequencies and elapsed inference seconds."""
    eps = permittivity(design)
    started = time.perf_counter()
    with torch.inference_mode():
        values, _ = solver.solve(
            torch.as_tensor(eps, dtype=torch.float32, device=solver.device),
            k_indices=k_indices,
        )
    return values.detach().cpu().numpy(), time.perf_counter() - started


def fem_bands(
    eps_wedge: np.ndarray, solver: FastBandSolver, k_indices: list[int] | None = None
) -> np.ndarray:
    """Solve the original constrained generalized eigenproblem at each k point."""
    uc_idx = solver.interp.uc_idx.detach().cpu().numpy()
    uc = np.asarray(eps_wedge, dtype=np.float64)[uc_idx]
    mass = Mn_pt_int(solver.pre, solver.pmid, uc, solver.xyq)
    z = sla.orth(solver.P.detach().cpu().numpy())
    bz = z.conj().T @ (mass @ z)
    bz = (bz + bz.conj().T) / 2
    all_k = solver.K_raw.detach().cpu().numpy()
    selected_k = all_k if k_indices is None else all_k[k_indices]
    result = np.zeros((len(selected_k), solver.Neig), dtype=np.float64)
    for i, k in enumerate(selected_k):
        stiffness = get_S1(solver.S_sp, solver.M_sp, get_T1(solver.pre, k), k)
        az = z.conj().T @ (stiffness @ z)
        az = (az + az.conj().T) / 2
        eig = sla.eigh(az, bz, eigvals_only=True, subset_by_index=[0, solver.Neig - 1])
        result[i] = np.sqrt(np.clip(np.sort(eig), 0.0, None)) / (2 * np.pi)
    return result


def comparison_metrics(predicted: np.ndarray, fem: np.ndarray) -> dict:
    error = predicted - fem
    metrics = {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "per_band_mae": np.abs(error).mean(axis=0).tolist(),
    }
    # A closed gap in both spectra has no interval to compare and is excluded.
    per_gap_iou = []
    for band in range(fem.shape[1] - 1):
        true_gap = (float(fem[:, band].max()), float(fem[:, band + 1].min()))
        pred_gap = (float(predicted[:, band].max()), float(predicted[:, band + 1].min()))
        true_open = true_gap[1] > true_gap[0]
        pred_open = pred_gap[1] > pred_gap[0]
        if not (true_open or pred_open):
            per_gap_iou.append(None)
            continue
        if not (true_open and pred_open):
            per_gap_iou.append(0.0)
            continue
        intersection = max(0.0, min(true_gap[1], pred_gap[1]) - max(true_gap[0], pred_gap[0]))
        union = max(true_gap[1], pred_gap[1]) - min(true_gap[0], pred_gap[0])
        per_gap_iou.append(intersection / union)
    active_ious = [value for value in per_gap_iou if value is not None]
    metrics["per_gap_iou"] = per_gap_iou
    metrics["mean_open_gap_iou"] = float(np.mean(active_ious)) if active_ious else None
    return metrics


def verify_bands(
    design,
    solver: FastBandSolver,
    k_indices: list[int] | None = None,
    fem_solver: FastBandSolver | None = None,
) -> tuple[np.ndarray, np.ndarray, dict, float, float]:
    """Predict and verify one immutable design snapshot."""
    predicted, prediction_seconds = predict_bands(design, solver, k_indices)
    started = time.perf_counter()
    fem = fem_bands(permittivity(design), fem_solver or solver, k_indices)
    fem_seconds = time.perf_counter() - started
    return predicted, fem, comparison_metrics(predicted, fem), prediction_seconds, fem_seconds

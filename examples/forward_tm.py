"""Predict TM bands for one normalized 36-value design; optionally check by FEM."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
from plots import plot_band_with_material
from tm_service import (
    comparison_metrics,
    fem_bands,
    load_solver,
    permittivity,
    predict_bands,
    example_vector,
    validate_design,
)


def read_vector(path: Path | None) -> np.ndarray:
    if path is None:
        return example_vector()
    raw = path.read_text(encoding="utf-8")
    values = (
        json.loads(raw)
        if path.suffix.lower() == ".json"
        else np.fromstring(raw.replace(",", " "), sep=" ")
    )
    return validate_design(np.asarray(values, dtype=np.float64).reshape(-1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vector-file",
        type=Path,
        help="JSON or whitespace/comma-separated 36 values; default is an illustrative pattern",
    )
    parser.add_argument(
        "--fem",
        action="store_true",
        help="compute reference bands and accuracy metrics",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/forward")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.set_num_threads(1)
    vector = read_vector(args.vector_file)
    eps = permittivity(vector)
    solver = load_solver(args.device)
    den, prediction_seconds = predict_bands(vector, solver)
    metrics = {"design_dim": 36, "prediction_seconds": prediction_seconds}
    fem = None
    if args.fem:
        started = time.perf_counter()
        fem = fem_bands(eps, solver)
        metrics["fem_seconds"] = time.perf_counter() - started
        metrics.update(comparison_metrics(den, fem))
    material = vector[solver.interp.uc_idx.cpu().numpy()].reshape(
        solver.n_grid, solver.n_grid
    )
    shown = den if fem is None else fem
    intervals = [(float(shown[:, band].max()), float(shown[:, band + 1].min()))
                 for band in range(shown.shape[1] - 1)]
    gap = max(intervals, key=lambda pair: pair[1] - pair[0])
    if fem is None:
        fig, _ = plot_band_with_material(den, uc16=material, gap=gap)
    else:
        fig, _ = plot_band_with_material(fem, den, material, gap=gap)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "bands.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

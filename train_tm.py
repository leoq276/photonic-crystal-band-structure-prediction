"""Train the TM eigenspace model on a uniformly sampled dataset."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset import Matrices, SubspaceDataset
from model import DEN
from tm_core import (
    K4_transform,
    ev_ef_metrics,
    get_trn_tst_idx,
    read_yaml,
    subspace_loss_Q,
    train_model,
)

ROOT = Path(__file__).resolve().parent
EPOCHS = 101
BATCH_SIZE = 32
TEST_BATCH_SIZE = 200
SEED = 42


def write_json(path: Path, payload: dict) -> None:
    """Atomically replace a JSON report so partial progress survives interruptions."""

    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(clean(payload), indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def _table_value(value, scale: float = 1.0) -> str:
    value = float(value) * scale
    return f"{value:.4f}" if math.isfinite(value) else ""


def table2_rows(metrics: dict) -> list[list[str]]:
    """Build the Table 2-style per-band summary, including its Average row."""
    ev = np.asarray(metrics["ev"])
    ef = metrics.get("ef")
    ef = None if ef is None else np.asarray(ef)
    residual = np.asarray(metrics["res"]).reshape(-1)
    d_ch = float(np.asarray(metrics["subspace"]).reshape(-1)[2])
    rows = []

    for band in range(len(ev)):
        rel_l1 = ef[band, 2] if ef is not None and band < len(ef) else np.nan
        res = residual[band] if band < len(residual) else np.nan
        # Table 2 shows the single test-set d_ch value centered in the column.
        d_ch_cell = _table_value(d_ch, 1e2) if band == len(ev) // 2 else ""
        rows.append(
            [
                str(band + 1),
                _table_value(ev[band, 0], 1e4),
                _table_value(rel_l1, 1e2),
                _table_value(res, 1e2),
                d_ch_cell,
            ]
        )

    def average(values, scale: float) -> str:
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]
        return _table_value(finite.mean(), scale) if finite.size else ""

    rel_l1_values = ef[:, 2] if ef is not None else np.array([])
    rows.append(
        [
            "Average",
            average(ev[:, 0], 1e4),
            average(rel_l1_values, 1e2),
            average(residual, 1e2),
            "-",
        ]
    )
    return rows


def write_metrics_csv(path: Path, metrics: dict) -> None:
    """Save every evaluation metric in a machine-readable long-form CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(("section", "metric", "band", "value", "unit"))

        def write_vector(section: str, name: str, values, unit: str = "") -> None:
            for band, value in enumerate(np.asarray(values).reshape(-1), start=1):
                numeric = float(value)
                writer.writerow(
                    (
                        section,
                        name,
                        band,
                        numeric if math.isfinite(numeric) else "",
                        unit,
                    )
                )

        def write_scalar(section: str, name: str, value, unit: str = "") -> None:
            numeric = float(value)
            writer.writerow(
                (section, name, "", numeric if math.isfinite(numeric) else "", unit)
            )

        ev = np.asarray(metrics["ev"])
        for column, name in enumerate(("mae", "r2", "mape")):
            write_vector("eigenvalue", name, ev[:, column])

        ef = metrics.get("ef")
        if ef is not None:
            ef = np.asarray(ef)
            for column, name in enumerate(
                ("max_absolute_error", "cosine_similarity", "relative_l1")
            ):
                write_vector("eigenfunction", name, ef[:, column])

        write_vector("eigenpair", "residual_norm", metrics["res"])
        write_vector("eigenpair", "residual_denominator", metrics["denum"])

        subspace_names = ("subspace_loss", "d_pr", "d_ch")
        for name, value in zip(
            subspace_names, np.asarray(metrics["subspace"]).reshape(-1)
        ):
            write_scalar("subspace", name, value)

        for name, value in zip(
            ("dataloader", "matrix_setup", "eigenpairs", "total"),
            np.asarray(metrics["time"]).reshape(-1),
        ):
            write_scalar("timing", name, value, "seconds")


def print_metrics(
    metrics: dict, csv_path: Path, test_materials: int, test_triplets: int
) -> None:
    """Print the Table 2-style summary with the matching physical scales."""
    rows = table2_rows(metrics)
    headers = (
        "Index",
        "MAE (x10^-4)",
        "RelL1 (x10^-2)",
        "||r_j||2 (x10^-2)",
        "d_ch (x10^-2)",
    )
    widths = tuple(
        max(len(header), *(len(row[i]) for row in rows))
        for i, header in enumerate(headers)
    )

    print(
        f"\nTest errors for the first {len(rows) - 1} eigenvalues/eigenfunctions "
        f"on the test set ({test_materials} materials, {test_triplets:,} triplets)."
    )
    print(
        "Residual and RelL1 columns are test-set means; d_ch is computed once per "
        "sample for the entire predicted subspace and then averaged."
    )
    print(" ".join(header.rjust(widths[i]) for i, header in enumerate(headers)))
    print(" ".join("-" * width for width in widths))
    for row in rows[:-1]:
        print(" ".join(value.rjust(widths[i]) for i, value in enumerate(row)))
    print(" ".join("-" * width for width in widths))
    print(" ".join(value.rjust(widths[i]) for i, value in enumerate(rows[-1])))
    print(f"Metrics CSV: {csv_path}")


def build_model(cfg: dict, basis: torch.Tensor, device: torch.device) -> DEN:
    return DEN(
        modes=cfg["modes"],
        width=cfg["width"],
        basis=basis.to(device),
        d_in=cfg["d_in"],
        d_out=cfg["d_out"],
        n_layers=cfg["n_layers"],
        k_net_ls=cfg["k_net_ls"],
        k_feats=cfg["k_feats"],
        J=cfg["J"],
        k_hidden=cfg["k_hidden"],
        cplx=cfg["cplx"],
        rank=cfg["rank_ls"],
        bw=cfg["bw"],
    ).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT.parent / "data/ev10-rand_uniform.mat",
        help="MAT dataset (default: ../data/ev10-rand_uniform.mat)",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/training")
    parser.add_argument(
        "--test-designs",
        type=int,
        default=None,
        help="number of initial materials reserved for testing (default: min(200, 20%% of materials))",
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS, help="training epochs (default: 101)"
    )
    parser.add_argument(
        "--skip-fem-eval",
        action="store_true",
        help="skip the slow post-training Rayleigh-Ritz/FEM-matrix evaluation",
    )
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Training data not found: {args.data}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    cfg = read_yaml(ROOT / "config.yaml")
    asset = ROOT / "assets"
    device = torch.device(args.device)

    dataset = SubspaceDataset([str(args.data)])
    print(
        "Dataset dimensions: "
        f"{dataset.N} materials x {dataset.Nk} k-points/material; "
        f"{dataset.Nm} mesh nodes; {dataset.Neig} modes; "
        f"{dataset.N * dataset.Nk} material-k samples"
    )
    if dataset.N < 2:
        parser.error(
            "At least two materials are required to create train and test splits"
        )
    test_count = args.test_designs
    if test_count is None:
        test_count = min(200, max(1, int(round(dataset.N * 0.2))))
    if not 1 <= test_count < dataset.N:
        parser.error(f"--test-designs must be between 1 and {dataset.N - 1}")
    test_ids = range(test_count)
    train_ids = range(test_count, dataset.N)
    print(
        f"Split: {len(train_ids)} training materials x {dataset.Nk} k-points; "
        f"{len(test_ids)} test materials x {dataset.Nk} k-points"
    )
    test_idx, train_idx = get_trn_tst_idx(
        test_ids, train_ids, Nk=dataset.Nk, shuffle=True
    )
    dataset.transformation_K(K4_transform)
    dataset.load_basis(
        str(asset / Path(cfg["basis_path"]).name), cfg["modes"], key=cfg["basis_key"]
    )
    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=min(TEST_BATCH_SIZE, len(test_idx)),
        shuffle=False,
    )

    # Match the notebook: the seed is set after the one-time train-index shuffle.
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = build_model(cfg, dataset.basis, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-6)
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "subden_rand_uniform_ev20_k4_BLR.pth"
    history_path = args.output_dir / "history.json"
    metrics_path = args.output_dir / "test_metrics.json"
    metadata = {
        "data_file": str(args.data.resolve()),
        "materials": dataset.N,
        "k_points_per_material": dataset.Nk,
        "mesh_nodes": dataset.Nm,
        "modes": dataset.Neig,
        "training_material_ids": list(train_ids),
        "test_material_ids": list(test_ids),
        "training_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "epochs_requested": args.epochs,
        "seed": SEED,
        "device": str(device),
        "checkpoint": str(checkpoint),
    }
    history_report = {"status": "running", "dataset": metadata, "history": {}}
    metrics_report = {
        "status": "pending",
        "dataset": metadata,
        "note": "Written after post-training FEM/Rayleigh-Ritz evaluation.",
    }
    write_json(history_path, history_report)
    write_json(metrics_path, metrics_report)

    def persist_history(epoch: int, log: dict) -> None:
        history_report["completed_epochs"] = epoch
        history_report["history"] = {
            key: values for key, values in log.items() if key != "w_ls"
        }
        write_json(history_path, history_report)

    try:
        log = train_model(
            None,
            model,
            subspace_loss_Q,
            train_loader,
            test_loader,
            len(train_idx),
            len(test_idx),
            args.epochs,
            optimizer,
            schedule,
            device=device,
            save_path=checkpoint,
            epoch_callback=persist_history,
        )
        history_report["status"] = "completed"
        history_report["completed_epochs"] = len(log["train_loss"])
        history_report["checkpoint"] = str(checkpoint)
        write_json(history_path, history_report)
    except Exception as error:
        history_report["status"] = "failed"
        history_report["error"] = f"{type(error).__name__}: {error}"
        write_json(history_path, history_report)
        metrics_report["status"] = "not_run_training_failed"
        metrics_report["error"] = history_report["error"]
        write_json(metrics_path, metrics_report)
        raise

    if args.skip_fem_eval:
        metrics_report["status"] = "skipped"
        metrics_report["note"] = "Evaluation was skipped by --skip-fem-eval."
        write_json(metrics_path, metrics_report)
    else:
        try:
            mats = Matrices(
                str(asset / Path(cfg["matrices_path"]).name), dataset, test_idx
            )
            mats.assemble_matrices()
            result = ev_ef_metrics(
                {"subspace": model},
                test_loader,
                dataset,
                mats,
                test_idx,
                model_pth={"subspace": str(checkpoint)},
                reorder=False,
                device=device,
            )
            metrics_report["status"] = "completed"
            metrics_report["metrics"] = {
                key: np.asarray(value).tolist()
                for key, value in result.items()
                if value is not None
            }
            write_json(metrics_path, metrics_report)
            metrics_csv_path = args.output_dir / "test_metrics.csv"
            write_metrics_csv(metrics_csv_path, result)
            print_metrics(
                result,
                metrics_csv_path,
                test_materials=len(test_ids),
                test_triplets=len(test_idx),
            )
        except Exception as error:
            metrics_report["status"] = "failed"
            metrics_report["error"] = f"{type(error).__name__}: {error}"
            write_json(metrics_path, metrics_report)
            raise
    print(f"Checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()

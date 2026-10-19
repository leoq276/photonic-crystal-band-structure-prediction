"""Exercise the real TM dataset/model/loss/optimizer path on synthetic data."""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset import Matrices, SubspaceDataset
from train_tm import build_model
from tm_core import K4_transform, ev_ef_metrics, read_yaml, subspace_loss_Q


def main() -> None:
    data = ROOT / "assets/smoke_training.mat"
    if not data.exists():
        raise FileNotFoundError(f"Run tests/make_smoke_fixture.py first: {data}")
    torch.manual_seed(42)
    cfg = read_yaml(ROOT / "config.yaml")
    dataset = SubspaceDataset([str(data)])
    assert (dataset.N, dataset.Nk, dataset.Nm, dataset.Neig) == (4, 1, 1089, 10)
    dataset.transformation_K(K4_transform)
    dataset.load_basis(str(ROOT / "assets/ev10-rand_uniform_basis.mat"), cfg["modes"])
    model = build_model(cfg, dataset.basis, torch.device("cpu"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-6)
    _, x, k, y = next(iter(DataLoader(dataset, batch_size=2)))
    loss, _ = subspace_loss_Q(None, y, model(x, k))
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    indices = torch.arange(4)
    matrices = Matrices(
        str(ROOT / "assets/ev10-uniform_matrices.mat"), dataset, indices
    )
    matrices.assemble_matrices()
    result = ev_ef_metrics(
        {"subspace": model},
        DataLoader(dataset, batch_size=2),
        dataset,
        matrices,
        indices,
        reorder=False,
        device="cpu",
    )
    assert result["ev"].shape[0] == 10
    print(
        f"Training + evaluation smoke passed; loss={float(loss):.6f}; MAE={result['ev'][:, 0].mean():.6f}"
    )


if __name__ == "__main__":
    main()

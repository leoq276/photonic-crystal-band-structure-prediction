"""Fast checks for forward-only design and FEM comparison semantics."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tm_service import comparison_metrics, example_vector, validate_design


def main() -> None:
    design = example_vector()
    assert design.shape == (36,)
    np.testing.assert_array_equal(validate_design(design), design)
    assert set(design) == {0.0, 1.0}

    fem = np.array([[1.0, 2.0, 3.0], [1.2, 2.1, 3.1]])
    predicted = np.array([[1.0, 2.1, 2.5], [1.2, 2.2, 3.1]])
    metrics = comparison_metrics(predicted, fem)
    np.testing.assert_allclose(metrics["per_gap_iou"], [0.8 / 0.9, 0.3 / 0.9])
    assert np.isclose(metrics["mean_open_gap_iou"], np.mean(metrics["per_gap_iou"]))

    closed = np.array([[1.0, 0.9], [1.1, 1.0]])
    closed_metrics = comparison_metrics(closed, closed)
    assert closed_metrics["per_gap_iou"] == [None]
    assert closed_metrics["mean_open_gap_iou"] is None
    print("Forward metrics passed.")


if __name__ == "__main__":
    main()

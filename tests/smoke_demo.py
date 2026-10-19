"""CPU integration check for the browser demo (requires httpx for TestClient)."""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from fastapi.testclient import TestClient

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
from demo.app import app
from tm_core import p4m_wedge_to_unit_cell_index
from tm_service import predict_bands, sample_k_indices


def main() -> None:
    torch.set_num_threads(1)
    with TestClient(app) as client:
        meta_response = client.get("/api/meta")
        assert meta_response.status_code == 200
        meta = meta_response.json()
        assert (meta["design_dim"], meta["grid_size"], meta["k_points"], meta["bands"]) == (
            36,
            16,
            30,
            10,
        )
        assert meta["path_ticks"] == [0, 10, 20, 30]
        assert meta["k_point_options"] == list(range(9, 31, 3))
        assert meta["device"] == "cpu"
        assert meta["default_device"] == "cpu"
        assert meta["cpu_threads"] >= 1
        assert meta["available_devices"] == (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
        mapping = np.asarray(meta["full_index_grid"])
        np.testing.assert_array_equal(
            mapping, p4m_wedge_to_unit_cell_index(16).numpy().reshape(16, 16)
        )
        for row in range(8):
            for col in range(8):
                assert meta["wedge_index_grid"][row][col] == (
                    int(mapping[row, col]) if row <= col else None
                )
        editable = meta["editable_index_grid"]
        assert sorted(index for row in editable for index in row if index is not None) == list(
            range(36)
        )
        for row in range(16):
            for col in range(16):
                assert editable[row][col] == (
                    int(mapping[row, col])
                    if row < 8 and col >= 8 and row + col - 8 >= 7
                    else None
                )

        design = meta["example_design"]
        payload = {"revision": 7, "design": design}
        invalid = (
            design[:-1],
            design + [0.0],
            [-0.1] + design[1:],
            [2.0] + design[1:],
            [None] + design[1:],
            [True] + design[1:],
            ["0.5"] + design[1:],
            [float("nan")] + design[1:],
            [float("inf")] + design[1:],
        )
        for values in invalid:
            for route in ("/api/predict", "/api/fem"):
                response = client.post(route, json={"revision": 1, "design": values})
                assert response.status_code == 422, (route, response.status_code)
        for count in (8, 10, 31, 9.0, True, "9"):
            for route in ("/api/predict", "/api/fem"):
                response = client.post(route, json={**payload, "k_points": count})
                assert response.status_code == 422, (route, count, response.status_code)
        for route in ("/api/predict", "/api/fem"):
            response = client.post(route, json={**payload, "device": "other"})
            assert response.status_code == 422, (route, response.status_code)
        if torch.cuda.is_available():
            app.state.cuda_available = False
            try:
                response = client.post("/api/predict", json={**payload, "device": "cuda"})
                assert response.status_code == 422
            finally:
                app.state.cuda_available = True

        prediction = client.post("/api/predict", json=payload)
        assert prediction.status_code == 200, prediction.text
        predicted = np.asarray(prediction.json()["bands"])
        assert predicted.shape == (30, 10)
        assert prediction.json()["revision"] == 7
        assert prediction.json()["device"] == "cpu"
        direct, _ = predict_bands(design, app.state.solver)
        np.testing.assert_allclose(predicted, direct, rtol=0, atol=1e-6)
        sampled_payload = {**payload, "revision": 8, "k_points": 9}
        sampled_prediction = client.post("/api/predict", json=sampled_payload)
        assert sampled_prediction.status_code == 200, sampled_prediction.text
        sampled = sampled_prediction.json()
        assert sampled["path_ticks"] == [0, 3, 6, 9]
        assert np.asarray(sampled["bands"]).shape == (9, 10)
        indices = sample_k_indices(app.state.solver, 9)
        np.testing.assert_allclose(sampled["bands"], predicted[indices], rtol=0, atol=1e-4)
        if torch.cuda.is_available():
            cuda_prediction = client.post("/api/predict", json={**payload, "device": "cuda"})
            assert cuda_prediction.status_code == 200, cuda_prediction.text
            assert cuda_prediction.json()["device"] == "cuda"
            np.testing.assert_allclose(cuda_prediction.json()["bands"], predicted, rtol=0, atol=1e-4)
        else:
            response = client.post("/api/predict", json={**payload, "device": "cuda"})
            assert response.status_code == 422

        # Dispatch FEM in one worker, then prove another prediction can finish
        # before the verification does.
        with ThreadPoolExecutor(max_workers=2) as pool:
            future = pool.submit(client.post, "/api/fem", json=payload)
            time.sleep(1)
            started = time.perf_counter()
            concurrent = client.post("/api/predict", json=payload)
            concurrent_seconds = time.perf_counter() - started
            assert concurrent.status_code == 200, concurrent.text
            assert not future.done(), "FEM unexpectedly blocked or already finished"
            fem_response = future.result(timeout=90)
        assert fem_response.status_code == 200, fem_response.text
        result = fem_response.json()
        assert result["revision"] == 7
        assert np.asarray(result["fem_bands"]).shape == (30, 10)
        assert np.asarray(result["prediction_bands"]).shape == (30, 10)
        metrics = result["metrics"]
        assert len(metrics["per_band_mae"]) == 10
        assert np.isfinite(metrics["mae"]) and metrics["mae"] >= 0
        assert np.isfinite(metrics["rmse"]) and metrics["rmse"] >= metrics["mae"]
        assert len(metrics["per_gap_iou"]) == 9
        active_ious = [value for value in metrics["per_gap_iou"] if value is not None]
        assert all(0 <= value <= 1 for value in active_ious)
        if active_ious:
            assert np.isclose(metrics["mean_open_gap_iou"], np.mean(active_ious))
        else:
            assert metrics["mean_open_gap_iou"] is None
        assert concurrent_seconds < result["fem_seconds"]
        sampled_fem_response = client.post("/api/fem", json=sampled_payload)
        assert sampled_fem_response.status_code == 200, sampled_fem_response.text
        sampled_fem = sampled_fem_response.json()
        assert np.asarray(sampled_fem["fem_bands"]).shape == (9, 10)
        np.testing.assert_allclose(
            sampled_fem["fem_bands"], np.asarray(result["fem_bands"])[indices],
            rtol=0, atol=1e-7,
        )
        if torch.cuda.is_available():
            cuda_fem = client.post("/api/fem", json={**sampled_payload, "device": "cuda"})
            assert cuda_fem.status_code == 200, cuda_fem.text
            np.testing.assert_allclose(
                cuda_fem.json()["fem_bands"], sampled_fem["fem_bands"], rtol=0, atol=1e-8
            )
        print(
            json.dumps(
                {
                    "status": "pass",
                    "prediction_seconds": prediction.json()["prediction_seconds"],
                    "concurrent_prediction_seconds": concurrent_seconds,
                    "fem_seconds": result["fem_seconds"],
                    "sampled_prediction_seconds": sampled["prediction_seconds"],
                    "sampled_fem_seconds": sampled_fem["fem_seconds"],
                    "mae": metrics["mae"],
                    "mean_open_gap_iou": metrics["mean_open_gap_iou"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()

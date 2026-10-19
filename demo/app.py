"""FastAPI routes for one locally running TM band solver."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Literal, Union

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StrictFloat, StrictInt

from tm_service import (
    k_path_ticks,
    load_solver,
    predict_bands,
    example_vector,
    sample_k_indices,
    validate_design,
    verify_bands,
)

STATIC_DIR = Path(__file__).absolute().parent / "static"


class DesignRequest(BaseModel):
    revision: int = Field(ge=0)
    design: list[Union[StrictFloat, StrictInt]]
    k_points: StrictInt = Field(default=30, ge=9, le=30, multiple_of=3)
    device: Literal["cpu", "cuda"] | None = None


def checked_design(payload: DesignRequest) -> np.ndarray:
    try:
        return validate_design(payload.design)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@asynccontextmanager
async def lifespan(app: FastAPI):
    default_device = os.environ.get("TM_DEMO_DEVICE", "cpu")
    if default_device not in ("cpu", "cuda"):
        raise RuntimeError(f"Unsupported device: {default_device}")
    cuda_available = torch.cuda.is_available()
    if default_device == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but is unavailable.")
    cpu_threads = int(os.environ.get("TM_DEMO_CPU_THREADS", min(4, os.cpu_count() or 1)))
    if cpu_threads < 1:
        raise RuntimeError("TM_DEMO_CPU_THREADS must be positive.")
    torch.set_num_threads(cpu_threads)
    app.state.solver = load_solver("cpu")
    app.state.cuda_solver = load_solver("cuda") if default_device == "cuda" else None
    app.state.cuda_available = cuda_available
    app.state.default_device = default_device
    app.state.solver_load_lock = Lock()
    app.state.example_design = example_vector()
    app.state.predict_lock = Lock()
    app.state.fem_lock = Lock()
    yield
    del app.state.cuda_solver
    del app.state.solver


def selected_solver(request: Request, payload: DesignRequest):
    device = payload.device or request.app.state.default_device
    if device == "cpu":
        return request.app.state.solver, device
    if not request.app.state.cuda_available:
        raise HTTPException(status_code=422, detail="CUDA is unavailable on this server.")
    with request.app.state.solver_load_lock:
        if request.app.state.cuda_solver is None:
            request.app.state.cuda_solver = load_solver("cuda")
    return request.app.state.cuda_solver, device


app = FastAPI(title="TM Band Explorer", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/meta")
def meta(request: Request):
    solver = request.app.state.solver
    full = solver.interp.uc_idx.detach().cpu().numpy().reshape(
        solver.n_grid, solver.n_grid
    )
    half = solver.n_grid // 2
    wedge = [
        [int(full[row, col]) if row <= col else None for col in range(half)]
        for row in range(half)
    ]
    editable = [
        [
            int(full[row, col])
            if row < half and col >= half and row + col - half >= half - 1
            else None
            for col in range(solver.n_grid)
        ]
        for row in range(solver.n_grid)
    ]
    editable_indices = [index for row in editable for index in row if index is not None]
    if sorted(editable_indices) != list(range(half * (half + 1) // 2)):
        raise RuntimeError("The quadrant-I editable triangle does not cover the wedge.")
    return {
        "design_dim": int(full.max()) + 1,
        "grid_size": solver.n_grid,
        "k_points": solver.Nk,
        "bands": solver.Neig,
        "path_ticks": k_path_ticks(solver),
        "k_point_options": list(range(9, 31, 3)),
        "device": request.app.state.default_device,
        "available_devices": ["cpu", "cuda"] if request.app.state.cuda_available else ["cpu"],
        "default_device": request.app.state.default_device,
        "cuda_ready": request.app.state.cuda_solver is not None,
        "cpu_threads": torch.get_num_threads(),
        "example_design": request.app.state.example_design.tolist(),
        "wedge_index_grid": wedge,
        "editable_index_grid": editable,
        "full_index_grid": full.tolist(),
    }


@app.post("/api/predict")
def predict(payload: DesignRequest, request: Request):
    design = checked_design(payload)
    solver, device = selected_solver(request, payload)
    indices = sample_k_indices(solver, payload.k_points)
    lock = request.app.state.predict_lock
    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Prediction is already running.")
    try:
        bands, seconds = predict_bands(design, solver, indices)
    finally:
        lock.release()
    return {
        "revision": payload.revision,
        "device": device,
        "k_points": payload.k_points,
        "path_ticks": [0, payload.k_points // 3, 2 * payload.k_points // 3, payload.k_points],
        "bands": bands.tolist(),
        "prediction_seconds": seconds,
    }


@app.post("/api/fem")
def fem(payload: DesignRequest, request: Request):
    design = checked_design(payload)
    solver, device = selected_solver(request, payload)
    indices = sample_k_indices(solver, payload.k_points)
    lock = request.app.state.fem_lock
    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="FEM verification is already running.")
    try:
        predicted, verified, metrics, prediction_seconds, fem_seconds = verify_bands(
            design, solver, indices, fem_solver=request.app.state.solver
        )
    finally:
        lock.release()
    return {
        "revision": payload.revision,
        "device": device,
        "k_points": payload.k_points,
        "path_ticks": [0, payload.k_points // 3, 2 * payload.k_points // 3, payload.k_points],
        "prediction_bands": predicted.tolist(),
        "fem_bands": verified.tolist(),
        "metrics": metrics,
        "prediction_seconds": prediction_seconds,
        "fem_seconds": fem_seconds,
    }

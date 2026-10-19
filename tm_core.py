"""TM finite-element assembly, Rayleigh-Ritz solving, training, and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable

import numpy as np
import scipy.io as scio
import scipy.sparse as scsp
import torch
import yaml
from scipy.interpolate import NearestNDInterpolator, griddata
from timeit import default_timer


def read_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def K4_transform(K: torch.Tensor, dataset=None) -> torch.Tensor:
    kx = K[:, 0:1]
    ky = K[:, 1:2]
    k2 = kx**2 + ky**2
    return torch.cat([kx, ky, torch.ones_like(kx), k2], dim=1)


def qr(A: torch.Tensor) -> torch.Tensor:
    Q, _ = torch.linalg.qr(A)
    return Q


def subspace_loss_Q(omega, QB: torch.Tensor, A: torch.Tensor, M=None):
    p = QB.shape[2]
    _, _, K2 = A.shape
    k = K2 // 2
    A = torch.complex(A[..., :k], A[..., k:])
    QA, _ = torch.linalg.qr(A)
    inner = QB.conj().transpose(1, 2) @ QA
    loss = p - torch.sum(torch.abs(inner) ** 2, dim=(1, 2))
    return torch.sum(loss), QA


def subspace_loss_Q_reshaped(omega, QB: torch.Tensor, A: torch.Tensor):
    bs, out_dim, K = A.shape
    p = QB.shape[-1]
    k = K // 2
    A = torch.complex(A[..., :k], A[..., k:])
    QA, _ = torch.linalg.qr(A.reshape(bs, out_dim, k))
    inner = QB.conj().transpose(1, 2) @ QA
    loss = p - torch.sum(torch.abs(inner) ** 2, dim=(1, 2))
    return torch.sum(loss), QA


def subspace_metrics(QB: torch.Tensor, A: torch.Tensor, averaged: bool = True, M=None):
    N, _, k = QB.shape
    loss, QA = subspace_loss_Q_reshaped(None, QB, A)
    gram = QA.conj().transpose(1, 2) @ QB
    _, S, _ = torch.linalg.svd(gram)
    S_clipped = torch.clamp(S, 1e-8, 1.0)
    pa = torch.acos(S_clipped)
    d_pr = torch.sin(pa[:, -1])
    d_ch = torch.sqrt(torch.clamp(k - torch.sum(S**2, dim=1), min=0.0))
    if averaged:
        loss = loss.mean()
        d_pr = d_pr.mean()
        d_ch = d_ch.mean()
        pa = pa.mean(dim=0)
    return loss / N, d_pr, d_ch, pa, QA


def get_trn_tst_idx(tst_sid, trn_sid, Nk: int = 50, shuffle: bool = True):
    tst_idx = torch.cat([torch.arange(s * Nk, (s + 1) * Nk) for s in tst_sid], dim=0)
    trn_idx = torch.cat([torch.arange(s * Nk, (s + 1) * Nk) for s in trn_sid], dim=0)
    if shuffle:
        trn_idx = trn_idx[torch.randperm(len(trn_idx))]
    return tst_idx, trn_idx


def as_scipy_coo(mat) -> scsp.coo_matrix:
    if scsp.issparse(mat):
        return mat.tocoo()
    return scsp.coo_matrix(mat)


def scipy_to_torch_sparse(
    mat,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    mat = as_scipy_coo(mat)
    indices = np.vstack([mat.row, mat.col])
    indices = torch.as_tensor(indices, dtype=torch.long, device=device)
    values = torch.as_tensor(mat.data, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(indices, values, mat.shape, device=device).coalesce()


def scipy_to_torch_dense(
    mat,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    if scsp.issparse(mat):
        arr = mat.toarray()
    else:
        arr = np.asarray(mat)
    return torch.as_tensor(arr, dtype=dtype, device=device)


def sparse_or_dense_mm(A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    if A.layout == torch.sparse_coo:
        return torch.sparse.mm(A, X)
    return A @ X


def find_mid_on_edge(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    tt = t[:, :3]
    e = p[tt, :]
    eplace = np.array([[0.5, 0.0, 0.5], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5]])
    return (eplace @ e).reshape(-1, 2)


class PreAssembleObj:
    def __init__(
        self,
        pre,
        *,
        use_torch: bool = False,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.weight = pre[0][0][0]
        self.yyT = pre[0][0][1]
        self.detB = pre[0][0][2]
        self.gphi = pre[0][0][3]
        self.yloc = pre[0][0][4]
        self.ii = pre[0][0][5].squeeze()
        self.jj = pre[0][0][6].squeeze()
        self.nt = pre[0][0][7].item()
        self.np = pre[0][0][8].item()
        self.device = torch.device(device)
        self.dtype = dtype

        if use_torch:
            self.to_torch(device=device, dtype=dtype)

    def to_torch(
        self,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "PreAssembleObj":
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype

        self.weight_th = torch.as_tensor(
            self.weight, dtype=self.dtype, device=self.device
        )
        self.yyT_th = torch.as_tensor(self.yyT, dtype=self.dtype, device=self.device)
        self.detB_th = torch.as_tensor(self.detB, dtype=self.dtype, device=self.device)
        indices = np.stack([self.ii - 1, self.jj - 1], axis=0).astype(
            np.int64, copy=False
        )
        self.iijj = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        return self


def get_T1(pre: PreAssembleObj, k: np.ndarray) -> scsp.coo_matrix:
    Yw = pre.yloc * pre.weight
    Yk = np.concatenate([k[0] * Yw, k[1] * Yw], axis=1)
    G = np.transpose(pre.gphi, [3, 2, 0, 1]).reshape(pre.nt, 6, 3, order="F")
    vT = (Yk @ G).reshape(pre.nt, 9, order="F")
    vT = vT * pre.detB
    return scsp.coo_matrix(
        (vT.reshape(-1), (pre.ii - 1, pre.jj - 1)),
        shape=(pre.np, pre.np),
    )


def get_Mn(pre: PreAssembleObj, eps_x_edge: np.ndarray) -> scsp.coo_matrix:
    vMn = ((pre.yyT * pre.weight) @ eps_x_edge) * pre.detB.T
    return scsp.coo_matrix(
        (vMn.reshape(-1, order="F"), (pre.ii - 1, pre.jj - 1)),
        shape=(pre.np, pre.np),
    )


def get_epsilon_int(
    points: np.ndarray, pixels: np.ndarray, xyq: np.ndarray
) -> np.ndarray:
    return griddata(xyq, pixels, points, method="nearest")


def Mn_pt_int(
    pre: PreAssembleObj, pmid: np.ndarray, pixels: np.ndarray, xyq: np.ndarray
):
    eps_x_edge = get_epsilon_int(pmid, pixels, xyq)
    return get_Mn(pre, eps_x_edge.reshape(3, -1, order="F"))


def get_S1(S, M, T1, k: np.ndarray):
    return S + 1j * T1 - 1j * T1.T + float((k**2).sum()) * M


def assemble_stiffness_torch(
    S,
    M,
    pre: PreAssembleObj,
    K_points: np.ndarray | torch.Tensor,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.complex64,
    dense: bool = False,
) -> list[torch.Tensor]:
    if isinstance(K_points, torch.Tensor):
        K_np = K_points.detach().cpu().numpy()
    else:
        K_np = np.asarray(K_points)

    S_coo = as_scipy_coo(S)
    M_coo = as_scipy_coo(M)
    out = []
    for k in K_np:
        T1 = get_T1(pre, k)
        S1 = get_S1(S_coo, M_coo, T1, k)
        if dense:
            out.append(scipy_to_torch_dense(S1, device=device, dtype=dtype))
        else:
            out.append(scipy_to_torch_sparse(S1, device=device, dtype=dtype))
    return out


def assemble_mass_torch(
    pre: PreAssembleObj,
    eps_edge: torch.Tensor,
    *,
    dense: bool = True,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    if not hasattr(pre, "yyT_th"):
        pre.to_torch(device=eps_edge.device)

    if eps_edge.ndim == 1:
        eps_edge = eps_edge.reshape(-1, 3).T
    elif eps_edge.shape[0] != 3:
        eps_edge = eps_edge.reshape(-1, 3).T

    eps_edge = eps_edge.to(device=pre.device, dtype=pre.dtype)
    vMn = ((pre.yyT_th * pre.weight_th) @ eps_edge) * pre.detB_th.T
    values = vMn.T.reshape(-1)
    Mn = torch.sparse_coo_tensor(
        pre.iijj,
        values,
        (pre.np, pre.np),
        device=pre.device,
    ).coalesce()
    Mn = Mn.to(dtype)
    return Mn.to_dense() if dense else Mn


def assemble_mass_from_pixels_torch(
    pre: PreAssembleObj,
    pixels: torch.Tensor,
    eps_mid_idx: torch.Tensor,
    *,
    dense: bool = True,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    eps_mid_idx = eps_mid_idx.to(device=pixels.device, dtype=torch.long)
    eps_edge = pixels.reshape(-1)[eps_mid_idx]
    return assemble_mass_torch(pre, eps_edge, dense=dense, dtype=dtype)


def row_space_complement_projection(
    mat,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
):
    if scsp.issparse(mat):
        mat = mat.toarray()
    mat = np.asarray(mat)
    q, _ = np.linalg.qr(np.conj(mat.swapaxes(-1, -2)), mode="reduced")
    eye = np.eye(mat.shape[-1], dtype=q.dtype)
    proj = eye - q @ np.conj(q.swapaxes(-1, -2))
    if device is None:
        return proj
    return torch.as_tensor(proj, dtype=dtype or torch.complex64, device=device)


def res_projection(mat):
    return row_space_complement_projection(mat)


def complex_subspace_from_real(f_ls: torch.Tensor) -> torch.Tensor:
    m2 = f_ls.shape[-1]
    if m2 % 2 != 0:
        raise ValueError(f"Expected an even real channel dimension, got {m2}.")
    m = m2 // 2
    return torch.complex(f_ls[..., :m], f_ls[..., m:])


def orthogonalize_and_project(
    f_ls: torch.Tensor,
    P: torch.Tensor | np.ndarray | None = None,
) -> torch.Tensor:
    V = complex_subspace_from_real(f_ls)
    Q, _ = torch.linalg.qr(V)
    if P is None:
        return Q
    if not isinstance(P, torch.Tensor):
        P = torch.as_tensor(P, dtype=Q.dtype, device=Q.device)
    else:
        P = P.to(dtype=Q.dtype, device=Q.device)
    return torch.matmul(P, Q)


def _select_batch_item(obj, idx: int):
    if isinstance(obj, (list, tuple)):
        return obj[idx]
    if isinstance(obj, torch.Tensor) and obj.ndim == 3:
        return obj[idx]
    return obj


def project_generalized_matrices(
    Q: torch.Tensor,
    A_mats,
    B_mats,
    *,
    idx_A: Iterable[int] | None = None,
    idx_B: Iterable[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    N, _, m = Q.shape
    device = Q.device
    dtype = Q.dtype
    idx_A = list(range(N)) if idx_A is None else list(idx_A)
    idx_B = list(range(N)) if idx_B is None else list(idx_B)

    Ak = torch.empty((N, m, m), dtype=dtype, device=device)
    Bk = torch.empty((N, m, m), dtype=dtype, device=device)
    for n in range(N):
        Qn = Q[n]
        An = _select_batch_item(A_mats, idx_A[n]).to(device=device, dtype=dtype)
        Bn = _select_batch_item(B_mats, idx_B[n]).to(device=device, dtype=dtype)
        AQ = sparse_or_dense_mm(An, Qn)
        BQ = sparse_or_dense_mm(Bn, Qn)
        Ak[n] = Qn.conj().T @ AQ
        Bk[n] = Qn.conj().T @ BQ
    return Ak, Bk


def solve_projected_hermitian(
    Ak: torch.Tensor,
    Bk: torch.Tensor,
    *,
    Neig: int | None = None,
    return_vecs: bool = False,
    jitter: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if jitter:
        eye = torch.eye(Bk.shape[-1], dtype=Bk.dtype, device=Bk.device)
        Bk = Bk + jitter * eye

    L = torch.linalg.cholesky(Bk)
    inv_L_A = torch.linalg.solve_triangular(L, Ak, upper=False)
    C = (
        torch.linalg.solve_triangular(
            L,
            inv_L_A.conj().transpose(-2, -1),
            upper=False,
        )
        .conj()
        .transpose(-2, -1)
    )
    C = 0.5 * (C + C.conj().transpose(-2, -1))

    if return_vecs:
        evals, y = torch.linalg.eigh(C)
    else:
        evals = torch.linalg.eigvalsh(C)
        y = None

    if Neig is not None:
        evals = evals[:, :Neig]
        if y is not None:
            y = y[:, :, :Neig]

    if y is None:
        return evals, None

    z = torch.linalg.solve_triangular(
        L.conj().transpose(-2, -1),
        y,
        upper=True,
    )
    return evals, z


def get_eigenpairs_torch(
    A_mats,
    f_ls: torch.Tensor,
    B_mats,
    P: torch.Tensor | np.ndarray | None = None,
    *,
    Neig: int | None = None,
    idx_A: Iterable[int] | None = None,
    idx_B: Iterable[int] | None = None,
    return_vecs: bool = False,
    return_lambda: bool = False,
    jitter: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    Q = orthogonalize_and_project(f_ls, P)
    Ak, Bk = project_generalized_matrices(Q, A_mats, B_mats, idx_A=idx_A, idx_B=idx_B)
    lam, small_vecs = solve_projected_hermitian(
        Ak,
        Bk,
        Neig=Neig,
        return_vecs=return_vecs,
        jitter=jitter,
    )

    values = (
        lam
        if return_lambda
        else torch.sqrt(torch.clamp(lam.real, min=0.0)) / (2 * np.pi)
    )
    if not return_vecs:
        return values, None
    eigvecs = torch.matmul(Q, small_vecs)
    return values, eigvecs


def ensure_torch_matrix_list(
    mats,
    *,
    device: str | torch.device,
    dtype: torch.dtype = torch.complex64,
    dense: bool = False,
):
    if isinstance(mats, (list, tuple)):
        out = []
        for mat in mats:
            if isinstance(mat, torch.Tensor):
                out.append(mat.to(device=device, dtype=dtype))
            elif dense:
                out.append(scipy_to_torch_dense(mat, device=device, dtype=dtype))
            else:
                out.append(scipy_to_torch_sparse(mat, device=device, dtype=dtype))
        return out
    if isinstance(mats, torch.Tensor):
        return mats.to(device=device, dtype=dtype)
    return (
        scipy_to_torch_dense(mats, device=device, dtype=dtype)
        if dense
        else scipy_to_torch_sparse(mats, device=device, dtype=dtype)
    )


def get_eigenpairs_fast(
    A_ls,
    f_ls: torch.Tensor,
    B_ls,
    P: torch.Tensor | np.ndarray | None = None,
    *,
    Neig: int | None = None,
    idx_A: Iterable[int] | None = None,
    idx_B: Iterable[int] | None = None,
    device: str | torch.device | None = None,
    return_vecs: bool = False,
    return_lambda: bool = False,
    return_torch: bool = False,
    dense_A: bool = False,
    dense_B: bool = False,
    jitter: float = 0.0,
):
    if device is None:
        device = f_ls.device if isinstance(f_ls, torch.Tensor) else "cuda"
    f_ls = torch.as_tensor(f_ls, dtype=torch.float32, device=device)
    A_torch = ensure_torch_matrix_list(A_ls, device=device, dense=dense_A)
    B_torch = ensure_torch_matrix_list(B_ls, device=device, dense=dense_B)
    values, vecs = get_eigenpairs_torch(
        A_torch,
        f_ls,
        B_torch,
        P,
        Neig=Neig,
        idx_A=idx_A,
        idx_B=idx_B,
        return_vecs=return_vecs,
        return_lambda=return_lambda,
        jitter=jitter,
    )
    if return_torch:
        return values, vecs
    values_np = values.detach().cpu().numpy()
    vecs_np = None if vecs is None else vecs.detach().cpu().numpy()
    return values_np, vecs_np


def train_model(
    omega,
    model,
    myloss,
    train_loader,
    test_loader,
    ntrain,
    ntest,
    epochs,
    optimizer,
    schedule,
    device: str | torch.device = "cuda",
    save_path: str | Path | None = None,
    transfer: bool = False,
    retrain_weights=None,
    pretrained_pth: str | Path | None = None,
    grad_hook=None,
    epoch_callback=None,
):
    device = torch.device(device)
    if transfer:
        if pretrained_pth is None:
            raise ValueError("Path of pretrained model is missing.")
        model.load_state_dict(torch.load(pretrained_pth, map_location=device))
    model.to(device)

    log = {
        "train_loss": [],
        "test_loss": [],
        "epoch_times": [],
        "lr": [],
        "w_ls": [],
    }
    for name, p in model.named_parameters():
        if "weight1" in name:
            log["w_ls"].append(p.detach().cpu().numpy())

    for ep in range(epochs):
        model.train()
        if transfer:
            if "weights_mask" in globals():
                model = globals()["weights_mask"](model, retrain_weights)
            if grad_hook is not None:
                grad_hook(model)

        t1 = default_timer()
        train_l2 = 0.0
        for sid, x, k, y in train_loader:
            x = x.to(device)
            k = k.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x, k)
            loss, _ = myloss(omega, y, out)
            loss.backward()
            optimizer.step()
            train_l2 += loss.item()

        log["lr"].append(optimizer.param_groups[0]["lr"])
        schedule.step()

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for sid, x, k, y in test_loader:
                x = x.to(device)
                k = k.to(device)
                y = y.to(device)
                out = model(x, k)
                loss, _ = myloss(omega, y, out)
                test_l2 += loss.item()

        train_l2 /= ntrain
        test_l2 /= ntest
        log["train_loss"].append(train_l2)
        log["test_loss"].append(test_l2)

        t2 = default_timer()
        log["epoch_times"].append(t2 - t1)
        print(
            f"Epoch [{ep + 1}/{epochs}], Train Loss: {train_l2:.6f}, "
            f"Test Loss: {test_l2:.6f}, Time: {t2 - t1:.2f}s"
        )

        if save_path is not None and (ep % 10 == 0 or ep == epochs - 1):
            print("Saving model.")
            torch.save(model.state_dict(), save_path)
            for name, p in model.named_parameters():
                if "weight1" in name:
                    log["w_ls"].append(p.detach().cpu().numpy())
        if epoch_callback is not None:
            epoch_callback(ep + 1, log)
    return log


def run_through_dataloader(
    model_dict, loader, M: int = 1425, device: str | torch.device = "cuda"
):
    device = torch.device(device)
    out = {"y_ls": []}
    for key in model_dict:
        out[key] = []

    with torch.no_grad():
        for sid, x, k, y in loader:
            x = x.to(device)
            k = k.to(device)
            for key, model in model_dict.items():
                model.eval()
                if key == "subspace":
                    pred = model(x, k)
                else:
                    pred = model(x)
                out[key].append(pred.detach().cpu())
            out["y_ls"].append(y.reshape(-1, M, y.shape[-1]).detach().cpu())

    for key in out:
        out[key] = torch.cat(out[key], dim=0)
    return out


def get_ev_metrics(ev, ev0):
    try:
        ev = ev.detach().cpu().numpy()
    except AttributeError:
        pass
    try:
        ev0 = ev0.detach().cpu().numpy()
    except AttributeError:
        pass
    diff = ev - ev0
    mae = np.mean(np.abs(diff), axis=0)
    mape = np.mean(np.abs(diff) / (np.abs(ev0) + 1e-5), axis=0)
    squared_error = np.sum(np.abs(diff) ** 2, axis=0)
    total_variance = np.sum(np.abs(ev0 - ev0.mean(axis=0)) ** 2, axis=0)
    r2 = 1 - np.divide(
        squared_error,
        total_variance,
        out=np.full_like(total_variance, np.nan, dtype=float),
        where=total_variance != 0,
    )
    return np.stack((mae, r2, mape), axis=1)


def get_ef_metrics(ef, ef0, averaged=True):
    # ef, ef0: [N, M, K]
    N, M, K = ef.shape
    try:
        ef0 = ef0.numpy()
    except AttributeError:
        pass
    width = 32
    C0 = np.sqrt(width**2 * 2)
    norm0 = np.sqrt(np.sum(np.abs(ef0) ** 2, axis=1, keepdims=True))
    norm = np.sqrt(np.sum(np.abs(ef) ** 2, axis=1, keepdims=True))
    V_norm = ef / norm * C0
    V0_norm = ef0 / norm0 * C0
    norm0_1 = np.sum(np.abs(V0_norm), axis=1)

    cs, rell1, maxae = np.zeros((N, K)), np.zeros((N, K)), np.zeros((N, K))
    for n in range(N):
        for k in range(K):
            v0 = V0_norm[n, :, k]
            v = V_norm[n, :, k]
            d = np.dot(v0, v.conj())
            Copt = d.conj() / np.abs(d)
            AE = np.abs(Copt * v0 - v)
            rell1[n, k] = np.sum(AE)
            cs[n, k] = np.abs(d)
            maxae[n, k] = np.max(AE)

    RelL1 = rell1 / norm0_1
    CS = cs / C0**2
    MaxAE = maxae
    if averaged:
        RelL1 = np.mean(RelL1, 0)
        CS = np.mean(CS, 0)
        MaxAE = np.mean(MaxAE, 0)
    return np.stack((MaxAE, CS, RelL1), 1)


def ev_ef_metrics(
    model_dict,
    test_loader,
    dataset,
    mats,
    tst_idx,
    model_pth=None,
    reorder: bool = True,
    time_only: bool = False,
    device: str | torch.device = "cuda",
    return_vecs=True,
):
    device = torch.device(device)
    if model_pth is not None:
        for key, f in model_pth.items():
            print(f"Loading {key} model: {f}")
            model_dict[key].load_state_dict(
                torch.load(f, map_location=device, weights_only=True)
            )

    t1 = time.time()
    print("Running through dataloader...")
    M = dataset.V.shape[-2] if hasattr(dataset, "V") else dataset.eps_x.shape[-2]
    out = run_through_dataloader(model_dict, test_loader, M=M, device=device)
    f_ls = out["subspace"]
    y_ls = out["y_ls"]
    t2 = time.time()

    print("Getting A and B...")
    A_ls = mats.S1
    B_ls = mats.Mn
    P = getattr(mats, "P_S21", None)
    if P is None:
        P = getattr(mats, "P", None)
    t3 = time.time()

    print("Getting eigenpairs...")
    neig = getattr(dataset, "Neig", y_ls.shape[-1])
    if not isinstance(neig, int):
        neig = y_ls.shape[-1]
    E_ls, V_ls = get_eigenpairs_fast(
        A_ls,
        f_ls,
        B_ls,
        P,
        Neig=neig,
        device=device,
        return_vecs=return_vecs,
        return_torch=False,
    )
    t4 = time.time()
    print(f"Cost {t4 - t1} seconds.")

    if time_only:
        return {"time": np.array([t2 - t1, t3 - t2, t4 - t3, t4 - t1])}

    print("Getting metrics...")
    res, denum = get_residual(A_ls, B_ls, E_ls, V_ls, P)
    loss, d_pr, d_ch, _, Q_ls = subspace_metrics(y_ls, f_ls)

    sample_indices = (
        tst_idx.detach().cpu().numpy()
        if isinstance(tst_idx, torch.Tensor)
        else np.asarray(tst_idx)
    )
    ev0 = dataset.ev[sample_indices, ...]
    ef0 = dataset.ef[sample_indices, ...] if hasattr(dataset, "ef") else None
    if reorder:
        if "RGSR" in globals() and V_ls is not None:
            print("Reordering...")
            rgsr_obj = globals()["RGSR"](
                dataset, tst_idx, loc=None, method="lambda_min"
            )
            E_ls, V_ls, _, _, _ = rgsr_obj.eigen_sortn(E_ls, V_ls)
            ev0, ef0, _, _, _ = rgsr_obj.eigen_sortn(ev0, ef0)
        else:
            print(
                "Reorder requested, but RGSR/eigenvectors are unavailable; skipping reorder."
            )

    ev_metrics = get_ev_metrics(E_ls, ev0)
    ef_metrics = get_ef_metrics(V_ls, ef0) if ef0 is not None else None
    return {
        "ev": ev_metrics,
        "ef": ef_metrics,
        "res": res.mean(0),
        "denum": denum.mean(0),
        "subspace": np.array(
            [
                loss.detach().cpu().item() if isinstance(loss, torch.Tensor) else loss,
                d_pr.detach().cpu().item() if isinstance(d_pr, torch.Tensor) else d_pr,
                d_ch.detach().cpu().item() if isinstance(d_ch, torch.Tensor) else d_ch,
            ]
        ),
        "time": np.array([t2 - t1, t3 - t2, t4 - t3, t4 - t1]),
    }


def get_residual(A_ls, B_ls, E_ls, V_ls, P):
    N, M, Neig = V_ls.shape
    norm = np.sqrt(np.sum(np.abs(V_ls) ** 2, axis=1, keepdims=True))
    V_ls = V_ls / norm
    E_ls = (E_ls * 2 * np.pi) ** 2
    res = np.zeros((N, Neig))
    denum = np.zeros((N, Neig))
    for n in range(N):
        A = A_ls[n]
        B = B_ls[n]
        lam = E_ls[[n], :]
        v = V_ls[n, :, :]
        Av = P @ (A @ v)
        lBv = P @ (lam * (B @ v))
        denum[n, :] = np.linalg.norm(Av, ord=2, axis=0) + np.linalg.norm(
            lBv, ord=2, axis=0
        )
        res[n, :] = np.linalg.norm(Av - lBv, ord=2, axis=0)
    return res, denum


@dataclass
class MeshInterpolationIndex:
    uc_idx: torch.Tensor
    eps_p_idx: torch.Tensor
    eps_mid_idx: torch.Tensor


def p4m_wedge_to_unit_cell_index(n_grid: int) -> torch.Tensor:
    h = n_grid // 2
    uc_idx = torch.arange(h * (h + 1) // 2, dtype=torch.long)
    full = torch.zeros((n_grid, n_grid), dtype=torch.long)
    tri_x, tri_y = torch.triu_indices(h, h)
    quadrant = torch.zeros((h, h), dtype=torch.long)
    quadrant[tri_x, tri_y] = uc_idx
    quadrant = quadrant + quadrant.T - torch.diag(torch.diag(quadrant))
    full[:h, :h] = quadrant
    full[:h, h:] = torch.fliplr(quadrant)
    full[h:, :] = torch.flipud(full[:h, :])
    return full.flatten()


def build_mesh_interpolation_index(
    *,
    n_grid: int,
    xyq: np.ndarray,
    p: np.ndarray,
    pmid: np.ndarray,
    device: str | torch.device = "cuda",
) -> MeshInterpolationIndex:
    uc_xy = np.arange(n_grid**2).astype(int)
    interp = NearestNDInterpolator(xyq, uc_xy)
    uc_idx = p4m_wedge_to_unit_cell_index(n_grid).to(device)
    eps_p_idx = torch.as_tensor(interp(p), dtype=torch.long, device=device)
    eps_mid_idx = torch.as_tensor(interp(pmid), dtype=torch.long, device=device)
    return MeshInterpolationIndex(
        uc_idx=uc_idx, eps_p_idx=eps_p_idx, eps_mid_idx=eps_mid_idx
    )


class FastBandSolver:
    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        config: dict | None = None,
        batch_size: int = 1,
        device: str | torch.device = "cuda",
        load_model: bool = True,
    ):
        if config is None:
            if config_path is None:
                raise ValueError("Either config_path or config must be provided.")
            config = read_yaml(config_path)
        self.config = config
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.cdtype = torch.complex64
        self.rdtype = torch.float32
        self.Neig = int(config["Neig"])
        self.m = int(config["d_out"] // 2)
        self._load_utilities()
        self._load_basis()
        self._load_matrices()
        self._assemble_stiffness()
        self.model = None
        if load_model:
            self._load_model()

    def _load_utilities(self):
        data = scio.loadmat(self.config["utilities_path"])
        self.K_raw = torch.as_tensor(data["K"].T, dtype=self.rdtype)
        self.Nk = self.K_raw.shape[0]
        self.N = self.Nk * self.batch_size
        self.K_feat = K4_transform(self.K_raw)
        self.K_batch = (
            self.K_feat.unsqueeze(0)
            .expand(self.batch_size, -1, -1)
            .reshape(self.N, -1)
            .to(self.device)
        )

        self.xq = data["xq"].reshape(-1, 1)
        self.yq = data["yq"].reshape(-1, 1)
        self.xyq = np.concatenate([self.xq, self.yq], axis=1)
        self.p = torch.as_tensor(data["p"].T, dtype=self.rdtype)
        self.p_np = self.p.numpy()
        self.p_batch = self.p.unsqueeze(0).expand(self.N, -1, -1).to(self.device)
        self.t = data["t"].T - 1
        self.pmid = find_mid_on_edge(self.p_np, self.t)
        self.n_grid = int(self.config["n_grid"])
        self.interp = build_mesh_interpolation_index(
            n_grid=self.n_grid,
            xyq=self.xyq,
            p=self.p_np,
            pmid=self.pmid,
            device=self.device,
        )
        self.idx_A = (
            torch.arange(self.Nk)
            .unsqueeze(0)
            .expand(self.batch_size, -1)
            .reshape(-1)
            .tolist()
        )

    def _load_basis(self):
        data = scio.loadmat(self.config["basis_path"])
        key = self.config.get("basis_key", "U")
        modes = int(self.config["modes"])
        self.basis = torch.as_tensor(
            data[key][:, :modes],
            dtype=self.cdtype,
            device=self.device,
        )

    def _load_matrices(self):
        data = scio.loadmat(self.config["matrices_path"])
        self.M_sp = as_scipy_coo(data["M"])
        self.S_sp = as_scipy_coo(data["S"])
        self.S21 = data["S21"]
        self.pre = PreAssembleObj(
            data["pre"],
            use_torch=True,
            device=self.device,
            dtype=self.rdtype,
        )
        self.P = row_space_complement_projection(
            self.S21,
            device=self.device,
            dtype=self.cdtype,
        )

    def _assemble_stiffness(self):
        self.S1 = assemble_stiffness_torch(
            self.S_sp,
            self.M_sp,
            self.pre,
            self.K_raw,
            device=self.device,
            dtype=self.cdtype,
            dense=False,
        )

    def _load_model(self):
        from model import DEN

        self.model = DEN(
            modes=self.config["modes"],
            width=self.config["width"],
            basis=self.basis,
            d_in=self.config["d_in"],
            d_out=self.config["d_out"],
            n_layers=self.config["n_layers"],
            k_net_ls=self.config["k_net_ls"],
            k_feats=self.config["k_feats"],
            J=self.config["J"],
            k_hidden=self.config["k_hidden"],
            cplx=self.config["cplx"],
            rank=self.config["rank_ls"],
            bw=self.config["bw"],
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(
                self.config["weight_path"], map_location=self.device, weights_only=True
            )
        )
        self.model.eval()

    def material_from_wedge(self, design_vec: torch.Tensor) -> torch.Tensor:
        design_vec = design_vec.to(self.device)
        return design_vec.reshape(-1)[self.interp.uc_idx]

    def mass_from_unit_cell(self, uc_flat: torch.Tensor) -> torch.Tensor:
        return assemble_mass_from_pixels_torch(
            self.pre,
            uc_flat.to(self.device),
            self.interp.eps_mid_idx,
            dense=True,
            dtype=self.cdtype,
        )

    def model_input_from_unit_cell(
        self, uc_flat: torch.Tensor, k_indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        eps_x = uc_flat.to(self.device).reshape(-1)[self.interp.eps_p_idx]
        positions = (
            self.p_batch[: self.Nk]
            if k_indices is None
            else self.p_batch.index_select(0, k_indices)
        )
        eps_x = eps_x.reshape(1, -1, 1).repeat(positions.shape[0], 1, 1)
        return torch.cat([eps_x, positions], dim=2)

    def solve_from_subspace(
        self,
        f_ls: torch.Tensor,
        mass: torch.Tensor,
        *,
        k_indices: torch.Tensor | None = None,
        return_vecs: bool = False,
        return_lambda: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        idx_A = (
            list(range(f_ls.shape[0]))
            if k_indices is None
            else k_indices.detach().cpu().tolist()
        )
        return get_eigenpairs_torch(
            self.S1,
            f_ls,
            mass,
            self.P,
            Neig=self.Neig,
            idx_A=idx_A,
            idx_B=[0] * f_ls.shape[0],
            return_vecs=return_vecs,
            return_lambda=return_lambda,
        )

    def solve(
        self,
        design: torch.Tensor,
        *,
        from_wedge: bool = True,
        k_indices: list[int] | None = None,
        return_vecs: bool = False,
        return_lambda: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.model is None:
            raise RuntimeError("FastBandSolver was created with load_model=False.")
        if k_indices is not None and self.batch_size != 1:
            raise ValueError("k_indices is supported only with batch_size=1.")
        uc_flat = (
            self.material_from_wedge(design)
            if from_wedge
            else design.reshape(-1).to(self.device)
        )
        mass = self.mass_from_unit_cell(uc_flat)
        selected = (
            None
            if k_indices is None
            else torch.as_tensor(k_indices, dtype=torch.long, device=self.device)
        )
        x = self.model_input_from_unit_cell(uc_flat, selected)
        k_features = (
            self.K_batch[: self.Nk]
            if selected is None
            else self.K_batch.index_select(0, selected)
        )
        f_ls = self.model(x, k_features)
        values, vecs = self.solve_from_subspace(
            f_ls,
            mass,
            k_indices=selected,
            return_vecs=return_vecs,
            return_lambda=return_lambda,
        )
        return values.reshape(f_ls.shape[0], self.Neig), vecs

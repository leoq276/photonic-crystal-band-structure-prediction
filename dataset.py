import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as scio
from tm_core import (
    qr,
    PreAssembleObj,
    res_projection,
    find_mid_on_edge,
    Mn_pt_int,
    get_T1,
    get_S1,
)


class SubspaceDataset(Dataset):
    def __init__(self, filenames: list):
        self.filenames = filenames
        self._load_data()

    def __len__(self):
        return self.V.size(0)

    def __getitem__(self, idx):
        y = self.V[idx]  # [*, Nm, Neig]
        sid = self.sample_id[idx]  # index of sample id
        x = self.eps_x[sid]
        k = self.K[idx]  # [*, 2]
        x = torch.cat([x, self.p], dim=1)  # [*, Nm, 1+2]
        return sid, x, k, y

    def _load_data(self):
        self.eps_x = []
        self.V = []
        self.ev = []
        self.ef = []
        self.K = []
        self.uc16, self.xq, self.yq = [], None, None
        for f in self.filenames:
            print(f"Loading {f}")
            data = scio.loadmat(f)
            # input
            eps_x = torch.tensor(data["eps_x"], dtype=torch.float32)  # [N, Nm]
            self.eps_x.append(eps_x)
            Ntmp = data["eps_x"].shape[0]
            tmp = data["K_ls"].reshape(Ntmp, 2, -1)
            self.K.append(
                torch.tensor(np.transpose(tmp, [0, 2, 1]), dtype=torch.float32)
            )  # [N, 2, Nk] -> [M, Nk, 2]
            # output
            V = torch.tensor(data["V_ls"], dtype=torch.complex64)  # [N, Nk, Nm, Neig]
            self.V.append(qr(V))
            # other data
            self.ev.append(data["E_ls"])  # [N, Nk, Neig]
            self.ef.append(data["V_ls"])  # [N, Nk, Nm, Neig]
            self.uc16.append(data["uc16"])  # [N, 16, 16]
            self.xq = data["xq"]  # [16, 16]
            self.yq = data["yq"]  # [16, 16]
        self.p = torch.tensor(data["p"].T, dtype=torch.float32)
        self.t = data["t"].T - 1
        self._preprocess_data()

    def _preprocess_data(self):
        self.eps_x = torch.cat(self.eps_x, dim=0).unsqueeze(-1)  # [N, Nm, 1]
        self.K = torch.cat(self.K, dim=0).reshape(-1, 2)  # [N, Nk, 2] -> [N*Nk, 2]
        self.K_ = self.K.clone()
        self.V = torch.cat(self.V, dim=0)
        self.N, self.Nk, self.Nm, self.Neig = self.V.shape
        self.V = self.V.reshape(-1, self.Nm, self.Neig)
        self.sample_id = torch.arange(self.N).repeat(self.Nk, 1).T.reshape(-1)
        self.uc16 = np.concatenate(self.uc16, axis=0).reshape(self.N, 256)
        self.uc16 = 7.9 * (self.uc16) + 1
        self.xq = self.xq.reshape(256, 1)
        self.yq = self.yq.reshape(256, 1)
        self.xyq = np.concatenate([self.xq, self.yq], axis=1)
        self.ev = np.concatenate(self.ev, axis=0).reshape(-1, self.Neig)
        self.ef = np.concatenate(self.ef, axis=0).reshape(-1, self.Nm, self.Neig)

    def load_basis(self, filename: str, basis_dim: int, key: str = "U"):
        print(f"Loading basis from {filename}")
        data = scio.loadmat(filename)
        self.basis = torch.tensor(data[key][:, :basis_dim], dtype=torch.complex64)

    def transformation_K(self, transform):
        self.K = transform(self.K_, self)


class Matrices:
    def __init__(self, filename: str, dataset: SubspaceDataset, idx):
        self.filename = filename
        self.idx = idx.numpy().astype(np.int64)
        self.sid = dataset.sample_id[idx].numpy().astype(np.int64)
        self.K = dataset.K_[idx, :2].numpy()
        self.uc16 = dataset.uc16[self.sid]
        self.xyq = dataset.xyq
        self.p = dataset.p.numpy()
        self.t = dataset.t
        self.load_matrices()

    def load_matrices(self):
        print(f"Loading matrices from {self.filename}")
        data = scio.loadmat(self.filename)
        self.M = data["M"]
        self.S = data["S"]
        self.S21 = data["S21"]
        pre_arr = data["pre"]
        self.pre = PreAssembleObj(pre_arr)
        self.P_S21 = res_projection(self.S21)

    def assemble_matrices(self):
        # Mn - sid
        pmid = find_mid_on_edge(self.p, self.t)
        self.Mn = []
        for i in range(len(self.idx)):
            self.Mn.append(Mn_pt_int(self.pre, pmid, self.uc16[i], self.xyq))

        # S1 - idx
        self.S1 = []
        for i in range(len(self.idx)):
            T1 = get_T1(self.pre, self.K[i, :])
            self.S1.append(get_S1(self.S, self.M, T1, self.K[i, :]))

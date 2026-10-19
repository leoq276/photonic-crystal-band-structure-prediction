import torch
import torch.nn as nn

import torch.nn.functional as F


class K2AlphaMLP(nn.Module):
    def __init__(self, k_feats, J, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k_feats, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, J),
        )

    def forward(self, k):
        return self.net(k).to(torch.complex64)


class EigenConv(nn.Module):
    """
    EigenConv maps x_real (bs, 2*in_ch, M) -> out_real (bs, 2*out_ch, M)
    basis: complex tensor (M, modes) provided externally at forward.
    Parameters:
        in_ch (int)  : number of complex input channels (d_hidden)
        out_ch (int) : number of complex output channels (h_hidden)
        modes (int)  : number of spectral modes (columns of basis)
    Note: real input is formatted as [real_part | imag_part] along channel dim (2*in_ch).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        modes: int,
        scale: float = None,
        k_net=False,
        k_feats=4,
        J=8,
        k_hidden=16,
        cplx=True,
        rank=64,
        bw=24,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes = modes
        self.cplx = cplx
        self.rank = rank
        self.k_feats = k_feats
        self.J = J
        self.k_hidden = k_hidden
        self.k_net = k_net
        if scale is None:
            scale = 1.0 / (in_ch * out_ch)
        scale_u = 1.0 / (modes * rank) if rank != -1 else None
        scale_v = 1.0 / (rank * modes) if rank != -1 else None
        scale_w = 1.0 / (modes**2)
        w = torch.randn(in_ch, out_ch, modes, dtype=torch.complex64) * scale
        if self.k_net:
            if self.rank == -1:
                w1 = torch.randn(J, modes, modes, dtype=torch.complex64) * scale_w
                self.M = BandedLinear(w1, modes, bw, k_net=True)
            else:
                u = torch.randn(J, modes, rank, dtype=torch.complex64) * scale_u
                v = torch.randn(J, rank, modes, dtype=torch.complex64) * scale_v
                self.M = BandedLinear(u, modes, bw, v=v, k_net=True)
            self.k_net_fun = K2AlphaMLP(k_feats, J, hidden=k_hidden)
        else:
            if self.rank == -1:
                w1 = torch.randn(modes, modes, dtype=torch.complex64) * scale_w
                self.M = BandedLinear(w1, modes, bw, k_net=False)
            else:
                u = torch.randn(modes, rank, dtype=torch.complex64) * scale_u
                v = torch.randn(rank, modes, dtype=torch.complex64) * scale_v
                self.M = BandedLinear(u, modes, bw, v=v, k_net=False)
            self.k_net_fun = lambda x: None
        self.weight = nn.Parameter(w)  # complex64 parameter

    def forward(
        self,
        x_real: torch.Tensor,
        k: torch.Tensor,
        basis: torch.Tensor,
        basis_conj_T: torch.Tensor,
    ):
        """
        x_real: float32 tensor shape (bs, 2*in_ch, M)
        basis: complex64 tensor shape (M, modes)
        returns: float32 tensor shape (bs, 2*out_ch, M)
        """
        real = x_real[:, : self.in_ch, :]  # (bs, in_ch, M)
        imag = x_real[:, self.in_ch :, :]  # (bs, in_ch, M)
        xc = torch.complex(real, imag)
        coeff = torch.einsum("pm,bcm->bcp", basis_conj_T, xc)  # (bs, in_ch, modes)
        h = torch.einsum("bcm,cdm->bdm", coeff, self.weight)  # (bs, out_ch, modes)
        alpha_k = self.k_net_fun(k)
        h = self.M(h, alpha_k)

        basis_T = basis.transpose(0, 1)  # (modes, M)
        yc = torch.einsum("bdm,mp->bdp", h, basis_T)  # (bs, out_ch, M), complex64

        if self.cplx:
            yc_real = yc.real.to(torch.float32)
            yc_imag = yc.imag.to(torch.float32)
            out = torch.cat([yc_real, yc_imag], dim=1)  # (bs, 2*out_ch, M)
        else:
            out = yc
        return out


class BandedLinear(nn.Module):
    def __init__(self, u, num_modes, bandwidth=3, v=None, k_net=False):
        super().__init__()
        self.num_modes = num_modes
        self.bandwidth = bandwidth
        self.k_net = k_net
        self.weight1 = nn.Parameter(u)
        if v is None:
            self.register_buffer("weight2", torch.eye(u.size(1), dtype=torch.complex64))
        else:
            self.weight2 = nn.Parameter(v)
        mask = self._create_banded_mask(num_modes, bandwidth)
        self.register_buffer("mask", mask)

    def _create_banded_mask(self, size, k):
        indices = torch.arange(size).unsqueeze(0)  # [1, M]
        diff = torch.abs(indices - indices.T)
        mask = (diff <= k).float()
        if self.k_net:
            mask = mask.unsqueeze(0)
        return mask

    def forward(self, x, alpha_k=None):
        # x: [batch, channels, modes]
        W = self.weight1 @ self.weight2
        w_effective = W * self.mask
        if self.k_net:
            w_effective = torch.einsum("bj,jmn->bmn", alpha_k, w_effective)
            return torch.einsum("bcm,bmn->bcn", x, w_effective)
        else:
            return torch.einsum("bcm,mn->bcn", x, w_effective)


class PointwiseMLP1D(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        # use Conv1d with kernel_size=1 acting on M dimension
        self.conv1 = nn.Conv1d(in_ch, mid_ch, kernel_size=1)
        self.conv2 = nn.Conv1d(mid_ch, out_ch, kernel_size=1)

    def forward(self, x):
        # x: (bs, in_ch, M)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x


class DEN(nn.Module):
    """
    FNO-like network on 1D node ordering of mesh (M nodes).
    - Input: x_in (bs, M, d_in)
    - Output: (bs, M, d_out)
    - basis must be provided at forward: complex tensor (M, modes)
    Notes:
      - width: number of complex channels (d_hidden)
      - internal representation for spectral convs uses 2*width real channels (real+imag)
    """

    def __init__(
        self,
        modes: int,
        width: int,
        basis,
        d_in: int = 1,
        d_out: int = 1,
        n_layers: int = 4,
        padding: int = 0,
        k_net_ls=None,
        k_feats=4,
        J=8,
        k_hidden=16,
        cplx=True,
        rank=64,
        bw=24,
    ):
        super().__init__()
        self.modes = modes
        self.width = width
        self.d_in = d_in
        self.d_out = d_out
        self.n_layers = n_layers
        self.padding = padding  # not used for now, kept for API similarity
        self.cplx = cplx
        self.k_net_ls = [True] * n_layers if k_net_ls is None else k_net_ls
        self.rank = rank if isinstance(rank, list) else [rank] * n_layers
        self.basis = basis if isinstance(basis, list) else [basis] * n_layers
        self.basis_T = [b.conj().T for b in self.basis]
        self.p = nn.Linear(d_in, 2 * width)
        if self.cplx:
            self.spec_convs = nn.ModuleList(
                [
                    EigenConv(
                        width,
                        width,
                        modes,
                        k_net=self.k_net_ls[i],
                        k_feats=k_feats,
                        J=J,
                        k_hidden=k_hidden,
                        cplx=self.cplx,
                        rank=self.rank[i],
                        bw=bw,
                    )
                    for i in range(n_layers)
                ]
            )
        else:
            self.spec_convs = nn.ModuleList(
                [
                    EigenConv(2 * width, 2 * width, modes, cplx=self.cplx)
                    for _ in range(n_layers)
                ]
            )
        self.ws = nn.ModuleList(
            [nn.Conv1d(2 * width, 2 * width, kernel_size=1) for _ in range(n_layers)]
        )
        # These unused blocks preserve checkpoint keys and initialization order.
        self.mlps = nn.ModuleList(
            [PointwiseMLP1D(2 * width, 2 * width, 2 * width) for _ in range(n_layers)]
        )

        # final projection: map real 2*width channels -> d_out channels (real)
        self.q = PointwiseMLP1D(2 * width, 4 * width, d_out)

    def forward(self, x_in: torch.Tensor, k: torch.Tensor):
        """
        x_in: (bs, M, d_in)  (float)
        basis: (M, modes) complex64
        returns: (bs, M, d_out) float
        """
        bs, M, din = x_in.shape
        device = x_in.device

        x1 = self.p(x_in)  # (bs, M, 2*width)
        # permute to channels-first (bs, 2*width, M) for conv1d usage
        x = x1.permute(0, 2, 1).contiguous()  # (bs, 2*width, M)

        # main layers
        for i in range(self.n_layers):
            spec_out = self.spec_convs[i](
                x, k, self.basis[i], self.basis_T[i]
            )  # already real concatenated
            # W operator maps real x (bs, 2*width,M) -> (bs, 2*width,M)
            w_out = self.ws[i](x)  # (bs, 2*width, M)
            x = spec_out + w_out  # (bs, 2*width, M)
            x = F.gelu(x)

        out = self.q(x)  # (bs, d_out, M)
        out = out.permute(0, 2, 1).contiguous()  # (bs, M, d_out)
        return out

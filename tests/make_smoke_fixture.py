"""Create a tiny synthetic MAT file for I/O and one-step training checks only."""
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.interpolate import NearestNDInterpolator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/smoke_training.mat"


def main() -> None:
    rng = np.random.default_rng(7)
    utilities = sio.loadmat(ROOT / "assets/utilities_uniform.mat")
    basis = sio.loadmat(ROOT / "assets/ev10-rand_uniform_basis.mat")["U"]
    p = utilities["p"]
    xy = np.column_stack([utilities["xq"].reshape(-1), utilities["yq"].reshape(-1)])
    n = 4
    pixels = rng.integers(0, 2, size=(n, 16, 16), dtype=np.uint8)
    eps_mesh = np.stack(
        [1.0 + 7.9 * NearestNDInterpolator(xy, img.reshape(-1))(p.T) for img in pixels]
    ).astype(np.float32)
    k = rng.uniform(0, np.pi, size=(n, 2, 1)).astype(np.float32)
    vectors = np.broadcast_to(basis[:, :10], (n, 1, basis.shape[0], 10)).copy()
    frequencies = np.linspace(0.1, 1.0, 10)[None, None, :] + 0.01 * np.arange(n)[:, None, None]
    sio.savemat(
        OUT,
        {
            "eps_x": eps_mesh,
            "K_ls": k,
            "V_ls": vectors,
            "E_ls": frequencies,
            "uc16": pixels,
            "xq": utilities["xq"],
            "yq": utilities["yq"],
            "p": p,
            "t": utilities["t"],
        },
        do_compression=True,
    )
    # scipy's MAT header normally embeds the current time. Stabilize fixture hashes.
    header = b"MATLAB 5.0 MAT-file, Platform: Python, DEN TM synthetic smoke fixture"
    with OUT.open("r+b") as stream:
        stream.write(header.ljust(116, b" "))
    print(OUT)


if __name__ == "__main__":
    main()

"""
plot_band.py
============
Band-structure figure with material inset (ported from plot_band_with_material.m).

One figure shows:
  * true bands       - solid lines, palette colors
  * predicted bands  - dashed lines, color = clip(c - 0.2, 0, 1)
  * target gap       - two black dashed horizontal lines (gap edges)
  * omega (optional) - one thick horizontal line marking the target frequency
  * material (uc16)   - square grayscale inset at top center (1 -> dark, 0 -> light)

k-path is assumed Gamma -> X -> M -> Gamma sampled in equal thirds; tick positions
are derived from the number of k-points so it works for any Nk.
"""

import numpy as np
import matplotlib.pyplot as plt


def get_31_colors():
    return np.array(
        [
            [0.57, 0.69, 0.30],
            [0.89, 0.88, 0.57],
            [0.76, 0.49, 0.58],
            [0.47, 0.76, 0.81],
            [0.21, 0.21, 0.35],
            [0.28, 0.57, 0.54],
            [0.07, 0.35, 0.40],
            [0.41, 0.20, 0.42],
            [0.60, 0.24, 0.18],
            [0.76, 0.84, 0.65],
            [0.77, 0.18, 0.78],
            [0.21, 0.33, 0.64],
            [0.88, 0.17, 0.56],
            [0.20, 0.69, 0.28],
            [0.26, 0.15, 0.47],
            [0.83, 0.27, 0.44],
            [0.87, 0.85, 0.42],
            [0.85, 0.51, 0.87],
            [0.99, 0.62, 0.76],
            [0.52, 0.43, 0.87],
            [0.00, 0.68, 0.92],
            [0.26, 0.45, 0.77],
            [0.98, 0.75, 0.00],
            [0.72, 0.81, 0.76],
            [0.77, 0.18, 0.78],
            [0.28, 0.39, 0.44],
            [0.22, 0.26, 0.24],
            [0.64, 0.52, 0.64],
            [0.87, 0.73, 0.78],
            [0.94, 0.89, 0.85],
            [0.85, 0.84, 0.86],
        ]
    )


def get_10_colors():
    selected = [0, 1, 2, 3, 29, 5, 23, 8, 10, 30]  # MATLAB [1 2 3 4 30 6 24 9 11 31]
    return get_31_colors()[selected]


def plot_band_with_material(
    E_n,  # bands shown solid [Nk, Neig] (true bands, or DEN if no truth)
    Ep_n=None,  # bands shown dashed [Nk, Neig] or None
    uc16=None,  # material [g, g] in [0, 1] (1 = material -> dark)
    fname="",  # save path; "" -> don't save
    eig_range=None,  # 0-indexed bands to show; None -> all
    colors=None,  # [>=Neig, 3]; default get_10_colors()
    gap=None,  # (lo, hi) gap edges -> two black dashed lines
    band_idx=None,  # if gap is None, compute gap from E_n at this index
    omega=None,  # if set, draw a thick horizontal line at this frequency
    omega_color=(0.35, 0.35, 0.35),
    omega_lw=2.5,
    flip=True,  # flipud the material (match MATLAB orientation)
    ax=None,  # plot into an existing Axes; None -> new figure
    subtitle=None,  # per-subplot title
):
    E_n = np.asarray(E_n).squeeze()
    if eig_range is None:
        eig_range = range(E_n.shape[1])
    eig_range = list(eig_range)
    E_n = E_n[:, eig_range]

    plot_pred = Ep_n is not None and np.size(Ep_n) > 0
    Ep_n = np.asarray(Ep_n).squeeze()[:, eig_range] if plot_pred else E_n

    colors = get_10_colors() if colors is None else np.asarray(colors)

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(4.0, 3.0))
    else:
        fig = ax.figure

    nK = E_n.shape[0]
    x = np.arange(1, nK + 1)

    # ---- bands --------------------------------------------------------------
    for ie in range(len(eig_range)):
        c = colors[ie % len(colors)]
        ax.plot(x, E_n[:, ie], lw=1.5, color=c)
        if plot_pred:
            ax.plot(x, Ep_n[:, ie], lw=1.5, ls="--", color=np.clip(c - 0.2, 0, 1))

    # ---- target gap (two black dashed lines) --------------------------------
    if gap is None and band_idx is not None:
        lo = E_n[:, band_idx - 1].max()
        hi = E_n[:, band_idx].min()
        gap = (lo, hi)
    if gap is not None and (gap[1] - gap[0] > 0):
        ax.axhline(float(gap[0]), color="k", ls="--", lw=1.0)
        ax.axhline(float(gap[1]), color="k", ls="--", lw=1.0)

    # ---- axis ranges (leave headroom for the inset; include omega) ----------
    y_min = float(np.min([E_n.min(), Ep_n.min()]))
    y_max = float(np.max([E_n.max(), Ep_n.max()]))
    if omega is not None:
        y_min, y_max = min(y_min, omega), max(y_max, omega)
    y_range = (y_max - y_min) or 1.0
    ax.set_xlim(1, nK)
    ax.set_ylim(y_min - 0.03 * y_range, y_max + 0.42 * y_range)

    # ---- omega marker (thick horizontal line) -------------------------------
    if omega is not None:
        ax.axhline(float(omega), color=omega_color, lw=omega_lw, ls="-", zorder=1)

    # ---- material inset (square) --------------------------------------------
    if uc16 is not None:
        uc = np.asarray(uc16, dtype=float).squeeze()
        if flip:
            uc = np.flipud(uc)
        img = np.repeat((0.85 * (1.0 - uc))[:, :, None], 3, axis=2)
        axins = ax.inset_axes([0.40, 0.70, 0.20, 0.28])
        axins.imshow(img, interpolation="nearest")
        axins.set_box_aspect(1)
        axins.set_anchor("N")
        axins.axis("off")

    # ---- k-path ticks: Gamma X M Gamma --------------------------------------
    ax.set_xticks([1, round(nK / 3), round(2 * nK / 3), nK])
    ax.set_xticklabels([r"$\Gamma$", "X", "M", r"$\Gamma$"])
    ax.set_ylabel(r"$\omega a/2\pi c$")
    if subtitle is not None:
        ax.set_title(subtitle, fontsize=10)
    ax.tick_params(labelsize=11)
    for s in ax.spines.values():
        s.set_visible(True)

    if created:
        fig.set_size_inches(4.0, 3.0)
        fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"Figure saved in: {fname}")
    return fig, ax

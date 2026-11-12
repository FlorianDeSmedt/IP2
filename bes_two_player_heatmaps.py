"""
Heatmap version of the BES figures.

Runs the SAME algorithm as bes_two_player.py (imported, not copied) and draws,
for every round, the surfaces the surrogate game actually optimises:

  top-left  : player A's UCB = mean + KAPPA*std of GP_A over (t_A, t_B).
              A's best reply = for every t_B (row) the t_A with the highest UCB.
  top-right : the same for player B (best reply = best t_B in every t_A column).
  bottom-left  : both best-reply curves together; they cross at the surrogate
                 equilibrium (the candidate). Shading = GP uncertainty.
  bottom-right : candidate per round vs the exact Nash.

Output (folder vis_heatmaps/): one PNG per round, bes_heatmaps_evolution.gif,
bes_heatmaps_summary.png.  Put this file next to bes_two_player.py.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

import bes_two_player as bes_mod
from bes_two_player import (bes, validate_nash, exact_nash, model_quiet, draw_plane,
                            draw_history, KAPPA, TAU_MAX)

OUT = "vis_heatmaps"
os.makedirs(OUT, exist_ok=True)

NG = 201
AX = np.linspace(0.0, TAU_MAX, NG)
GX, GY = np.meshgrid(AX, AX)                       # GX = t_A (columns), GY = t_B (rows)
PTS = np.column_stack([GX.ravel(), GY.ravel()])


def ucb_surface(gp):
    m, sd = gp.predict(PTS)
    return (m + KAPPA * sd).reshape(GX.shape)       # shape (t_B rows, t_A cols)


def draw_ucb(ax, s, player, vmax):
    name = "AB"[player]
    gp = (s["gpA"], s["gpB"])[player]
    U = ucb_surface(gp)
    im = ax.imshow(U, origin="lower", extent=[0, TAU_MAX, 0, TAU_MAX], aspect="auto",
                   cmap="viridis", vmin=0, vmax=vmax)
    # best reply of this player on the UCB (what the surrogate game uses)
    if player == 0:      # A: for every t_B row, best t_A
        ax.plot(AX[U.argmax(axis=1)], AX, color="w", lw=2.2)
        ax.plot(AX[U.argmax(axis=1)], AX, color="C0", lw=1.2, label="A's best reply (UCB)")
    else:                # B: for every t_A column, best t_B
        ax.plot(AX, AX[U.argmax(axis=0)], color="w", lw=2.2)
        ax.plot(AX, AX[U.argmax(axis=0)], color="C1", lw=1.2, label="B's best reply (UCB)")
    p = s["path"]
    ax.plot(p[:, 0], p[:, 1], "-", color="0.85", lw=1, marker=".", ms=3)
    ax.scatter(s["X"][:, 0], s["X"][:, 1], c="k", s=28, edgecolors="w", linewidths=0.6, zorder=6)
    ax.scatter(*s["cand"], marker="*", c="C2", s=230, edgecolors="k", zorder=7)
    ax.scatter(*exact_nash(), marker="X", c="gold", s=90, edgecolors="k", zorder=8)
    ax.set_xlim(0, TAU_MAX); ax.set_ylim(0, TAU_MAX)
    ax.set_xlabel("toll A  $t_A$"); ax.set_ylabel("toll B  $t_B$")
    ax.set_title(f"Player {name}: UCB of $R_{name}$ (mean + {KAPPA:g}σ)", fontsize=10)
    return im


def make_figures(snapshots):
    n = len(snapshots)
    star_rev = model_quiet(*exact_nash())
    vmaxes = [2.0 * v for v in star_rev]            # fixed colour scale -> comparable frames
    frames = []
    for k, s in enumerate(snapshots, start=1):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9.4), dpi=110)
        imA = draw_ucb(axes[0, 0], s, 0, vmaxes[0])
        imB = draw_ucb(axes[0, 1], s, 1, vmaxes[1])
        fig.colorbar(imA, ax=axes[0, 0], label="UCB revenue A", fraction=0.046, pad=0.02)
        fig.colorbar(imB, ax=axes[0, 1], label="UCB revenue B", fraction=0.046, pad=0.02)
        draw_plane(axes[1, 0], s, legend=False)
        axes[1, 0].set_title("Both best replies: crossing = surrogate equilibrium", fontsize=10)
        draw_history(axes[1, 1], s, n)
        h, l = axes[1, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=False)
        nxt = ("candidate already evaluated -> STOP, then validate" if s["seen"] else
               f"run real model at ({s['cand'][0]:.2f}, {s['cand'][1]:.2f})")
        fig.suptitle(f"BES round {s['rnd']}: {len(s['X'])} model runs so far; next: {nxt}", fontsize=12)
        fig.tight_layout(rect=[0, 0.07, 1, 0.97])
        fname = f"{OUT}/bes_heatmaps_round_{k:02d}.png"
        fig.savefig(fname); plt.close(fig)
        frames.append(imageio.imread(fname))
    frames += [frames[-1]] * 2
    imageio.mimsave(f"{OUT}/bes_heatmaps_evolution.gif", frames, fps=1.5, loop=0)

    picks = sorted(set([0, n // 2, n - 1]))
    fig, axes = plt.subplots(2, len(picks), figsize=(5.2 * len(picks), 9), squeeze=False)
    for c, i in enumerate(picks):
        draw_ucb(axes[0, c], snapshots[i], 0, vmaxes[0])
        draw_ucb(axes[1, c], snapshots[i], 1, vmaxes[1])
        axes[0, c].set_title(f"Round {snapshots[i]['rnd']}: A's UCB", fontsize=10)
        axes[1, c].set_title(f"Round {snapshots[i]['rnd']}: B's UCB", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/bes_heatmaps_summary.png", dpi=130); plt.close(fig)
    print(f"Saved {n} round figures, {OUT}/bes_heatmaps_evolution.gif and {OUT}/bes_heatmaps_summary.png")


if __name__ == "__main__":
    t, R_t, X, R, hist, snaps = bes()
    runs_loop = bes_mod.n_model_calls
    ok, gains = validate_nash(t, R_t)
    make_figures(snaps)
    tA_star, tB_star = exact_nash()
    print(f"\nBES result : tA = {t[0]:.3f}, tB = {t[1]:.3f}   ({runs_loop} model runs in the loop)")
    print(f"Exact Nash : tA = {tA_star:.3f}, tB = {tB_star:.3f}")
    print(f"Validation : valid eps-local Nash = {ok}")

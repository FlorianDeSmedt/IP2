"""
Bayesian Equilibration Scheme (BES) for a 2-player toll game.

Algorithm (as on the slide)
---------------------------
1. Evaluate the real model at a few initial toll pairs.
2. Repeat:
   a. Fit one GP per player; each takes BOTH tolls as input.
   b. Solve the surrogate game (NO model runs): start from the previous
      candidate and alternate best replies on UCB = mean + kappa*std
         - A: best t_A given the current t_B (1-D search on GP_A)
         - B: best t_B given the current t_A (1-D search on GP_B)
      until the tolls stop moving.
   c. If the candidate was already evaluated (within a tolerance) -> stop.
      Otherwise run the REAL model there (one expensive call) and add the
      result for both players to the data.
3. Validate on the real model: can any player gain more than eps by changing
   only their own toll by +-phi?  (the check used in the MBE paper)

Everything specific to the toy problem lives in `model()` and the "toy only"
helpers (exact Nash / exact best replies, used only to draw reference lines in
the figures); swap `model()` for the real simulator.

Output (folder vis/): one PNG per round, bes_evolution.gif, bes_summary.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

os.makedirs("vis", exist_ok=True)   # output folder for figures and GIF

# ------------------------------------------------------------------ settings
TAU_MAX   = 10.0      # tolls live in [0, TAU_MAX]
LENGTH    = np.array([4.0, 4.0])   # GP length-scale per toll (fixed, as in the 1-player code)
KAPPA     = 2.0       # UCB = mean + KAPPA * std  (the "top of the light blue area")
ALPHA     = 0.5       # damping of the best-response updates (1 = undamped)
BR_TOL    = 1e-3      # surrogate game stops when tolls move less than this
BR_MAXIT  = 200       # ... or after this many best-response rounds
DELTA     = 0.05      # candidate counts as "already evaluated" within this distance
MAX_ROUNDS = 40       # budget on real-model evaluations in the loop
PHI, EPS  = 0.05, 1.0 # validation: perturbation size and allowed gain
N_GRID    = 401       # points of the 1-D search per player (cheap: GP only)
SEED      = 0

n_model_calls = 0     # counter of expensive evaluations


# -------------------------------------------------------- the "expensive model"
def model(tA, tB):
    """Returns (revenue_A, revenue_B). Roads are substitutes; linear demand."""
    global n_model_calls
    n_model_calls += 1
    N = 1000
    DA = max(0.0, N * (1 - 0.12 * tA + 0.05 * tB))
    DB = max(0.0, N * (1 - 0.15 * tB + 0.06 * tA))
    return tA * DA, tB * DB


# ------------------------------------------------------------------ the GP
def rbf(A, B):
    A, B = A / LENGTH, B / LENGTH
    sq = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return np.exp(-0.5 * sq)                      # sigma_f = 1 on standardised y

class GP:
    """Same maths as gp_posterior in the 1-player code, but inputs have shape
    (n, d) and the Cholesky is computed once (fit) and reused (predict)."""
    def __init__(self, X, y, noise=1e-6):
        self.X = X
        self.y_mean = y.mean()
        self.y_std = y.std() if y.std() > 1e-8 else 1.0
        z = (y - self.y_mean) / self.y_std
        K = rbf(X, X) + noise * np.eye(len(X))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, z))

    def predict(self, Xs):
        Ks = rbf(self.X, Xs)
        mean = Ks.T @ self.alpha
        v = np.linalg.solve(self.L, Ks)
        var = np.clip(1.0 - np.sum(v**2, axis=0), 1e-12, None)
        return mean * self.y_std + self.y_mean, np.sqrt(var) * self.y_std

    def rel_std(self, Xs):
        """Posterior std on the standardised scale (0 = certain, 1 = prior).
        Depends only on WHERE we evaluated, so it is the same for both players."""
        v = np.linalg.solve(self.L, rbf(self.X, Xs))
        return np.sqrt(np.clip(1.0 - np.sum(v**2, axis=0), 0.0, None))


# ------------------------------------------------- the surrogate game (no model calls)
LINE = np.linspace(0.0, TAU_MAX, N_GRID)

def best_reply(gp, t, player, kappa=KAPPA):
    """Best toll for `player` (0=A, 1=B) on the GP's UCB, other toll fixed."""
    pts = np.tile(t, (N_GRID, 1))
    pts[:, player] = LINE                         # vary only the player's own toll
    mean, std = gp.predict(pts)
    return LINE[np.argmax(mean + kappa * std)]

def solve_surrogate_game(gpA, gpB, start):
    """Alternate damped best replies on the UCBs. Returns (tolls, path, converged);
    `path` = the tolls after every round (only used for plotting)."""
    t = np.array(start, float)
    path, converged = [t.copy()], False
    for _ in range(BR_MAXIT):
        t_old = t.copy()
        t[0] = (1 - ALPHA) * t[0] + ALPHA * best_reply(gpA, t, 0)   # A replies to B
        t[1] = (1 - ALPHA) * t[1] + ALPHA * best_reply(gpB, t, 1)   # B replies to A
        path.append(t.copy())
        if np.abs(t - t_old).max() < BR_TOL:
            converged = True
            break
    return t, np.array(path), converged


# ----------------------------------------------------------- validation on real model
def validate_nash(t, base, phi=PHI, eps=EPS):
    """Largest gain any player can get by moving ONLY their own toll by +-phi
    (4 extra real-model runs; `base` = revenues at t, already known from the
    loop). Valid eps-local Nash if all gains <= eps."""
    gains = {}
    for p, name in enumerate("AB"):
        for sign in (+1, -1):
            t2 = t.copy()
            t2[p] = np.clip(t2[p] + sign * phi, 0, TAU_MAX)
            gains[f"{name}{'+' if sign > 0 else '-'}"] = float(model(*t2)[p] - base[p])
    return max(gains.values()) <= eps, gains


# ------------------------------------------------------------------ BES loop
def bes(n_init=4, verbose=True):
    rng = np.random.default_rng(SEED)
    X = rng.uniform(0, TAU_MAX, size=(n_init, 2))            # initial design
    R = np.array([model(*x) for x in X])                     # (n, 2): R_A, R_B
    cand = np.array([TAU_MAX / 2, TAU_MAX / 2])
    history, snapshots = [], []

    for rnd in range(1, MAX_ROUNDS + 1):
        gpA, gpB = GP(X, R[:, 0]), GP(X, R[:, 1])            # (a) one GP per player
        start = cand.copy()
        cand, path, conv = solve_surrogate_game(gpA, gpB, cand)   # (b) cheap game solve
        history.append(cand.copy())
        if verbose:
            print(f"round {rnd:2d}: candidate = ({cand[0]:.3f}, {cand[1]:.3f})   "
                  f"model runs so far: {n_model_calls}"
                  + ("" if conv else "   [surrogate game hit BR_MAXIT]"))

        seen = np.min(np.abs(X - cand).max(axis=1)) < DELTA  # (c) already evaluated?
        snapshots.append(dict(rnd=rnd, X=X.copy(), R=R.copy(), gpA=gpA, gpB=gpB,
                              start=start, path=path, cand=cand.copy(),
                              conv=conv, seen=seen, hist=np.array(history)))
        if seen:
            if verbose: print("-> candidate already evaluated: stop")
            break
        X = np.vstack([X, cand])                             #     real model, once
        R = np.vstack([R, model(*cand)])

    # candidate = the evaluated point closest to the last surrogate equilibrium
    i_final = np.argmin(np.abs(X - cand).max(axis=1))
    return X[i_final], R[i_final], X, R, np.array(history), snapshots


# ------------------------------------------------ toy only: exact solution (for plots)
def exact_nash():
    return np.linalg.solve([[0.24, -0.05], [-0.06, 0.30]], [1, 1])

def true_br_A(tB): return (1 + 0.05 * tB) / 0.24     # from dR_A/dt_A = 0
def true_br_B(tA): return (1 + 0.06 * tA) / 0.30     # from dR_B/dt_B = 0


# ------------------------------------------------------------------ figures
N_PLANE = 81
PLANE = np.linspace(0.0, TAU_MAX, N_PLANE)
GX, GY = np.meshgrid(np.linspace(0, TAU_MAX, 101), np.linspace(0, TAU_MAX, 101))
GRID_PTS = np.column_stack([GX.ravel(), GY.ravel()])

def draw_plane(ax, s, legend=True):
    """Toll plane: shading = GP uncertainty, curves = best replies on the UCB.
    Where the solid curves cross is the surrogate game's equilibrium."""
    star = exact_nash()
    unc = s["gpA"].rel_std(GRID_PTS).reshape(GX.shape)
    ax.imshow(unc, origin="lower", extent=[0, TAU_MAX, 0, TAU_MAX], cmap="Blues",
              vmin=0, vmax=1, alpha=0.55, aspect="auto")
    # best-reply curves of the surrogate (UCB)
    brA = [best_reply(s["gpA"], np.array([0.0, tb]), 0) for tb in PLANE]
    brB = [best_reply(s["gpB"], np.array([ta, 0.0]), 1) for ta in PLANE]
    ax.plot(brA, PLANE, "C0", lw=2, label="A's best reply (UCB)")
    ax.plot(PLANE, brB, "C1", lw=2, label="B's best reply (UCB)")
    # true best replies (toy only)
    ax.plot(true_br_A(PLANE), PLANE, "C0--", lw=1, label="A's true best reply")
    ax.plot(PLANE, true_br_B(PLANE), "C1--", lw=1, label="B's true best reply")
    # surrogate-game iterations
    p = s["path"]
    ax.plot(p[:, 0], p[:, 1], "-", color="0.35", lw=1, marker=".", ms=4, label="best-reply iterations")
    ax.scatter(*s["start"], facecolors="none", edgecolors="0.35", s=70, zorder=5, label="start (prev. candidate)")
    # data
    X = s["X"]
    ax.scatter(X[:-1, 0], X[:-1, 1], c="k", s=30, zorder=6, label="model runs")
    ax.scatter(X[-1:, 0], X[-1:, 1], c="k", s=30, zorder=6)
    ax.scatter(*s["cand"], marker="*", c="C2", s=260, edgecolors="k", zorder=7,
               label="candidate equilibrium")
    ax.scatter(*star, marker="X", c="gold", s=110, edgecolors="k", zorder=8, label="exact Nash")
    ax.set_xlim(0, TAU_MAX); ax.set_ylim(0, TAU_MAX)
    ax.set_xlabel("toll A  $t_A$"); ax.set_ylabel("toll B  $t_B$")
    if legend:
        ax.legend(fontsize=6.5, loc="upper right", framealpha=0.9)

def draw_slice(ax, s, player, ylim):
    """GP of one player along its OWN toll, the other toll fixed at the candidate."""
    gp = (s["gpA"], s["gpB"])[player]
    name = "AB"[player]
    t = s["cand"].copy()
    pts = np.tile(t, (N_GRID, 1)); pts[:, player] = LINE
    m, sd = gp.predict(pts)
    true = np.array([model_quiet(*(p)) [player] for p in pts])
    ax.plot(LINE, true, "k--", lw=2, label=f"true $R_{name}$")
    ax.plot(LINE, m, "C0", lw=2, label="GP mean")
    ax.fill_between(LINE, m - 2 * sd, m + 2 * sd, color="C0", alpha=0.2, label="±2σ (top = UCB)")
    ax.scatter(s["X"][:, player], s["R"][:, player], c="0.6", s=25, zorder=4,
               label="model runs (other toll differs)")
    ax.axvline(t[player], color="C2", ls=":", lw=2, label=f"best reply $t_{name}$ = {t[player]:.2f}")
    other = "B" if player == 0 else "A"
    ax.set_xlim(0, TAU_MAX); ax.set_ylim(*ylim)
    ax.set_xlabel(f"toll {name}  $t_{name}$")
    ax.set_ylabel(f"revenue {name}")
    ax.set_title(f"Player {name}: GP slice at $t_{other}$ = {t[1 - player]:.2f}", fontsize=10)
    ax.legend(fontsize=7, loc="lower center")

def model_quiet(tA, tB):
    """Same formula as model() but does not count as an expensive run (plots only)."""
    N = 1000
    DA = max(0.0, N * (1 - 0.12 * tA + 0.05 * tB))
    DB = max(0.0, N * (1 - 0.15 * tB + 0.06 * tA))
    return tA * DA, tB * DB

def draw_history(ax, s, n_total):
    star = exact_nash()
    h = s["hist"]; r = np.arange(1, len(h) + 1)
    ax.plot(r, h[:, 0], "C0-o", ms=4, label="$t_A$ candidate")
    ax.plot(r, h[:, 1], "C1-o", ms=4, label="$t_B$ candidate")
    ax.axhline(star[0], color="C0", ls="--", lw=1)
    ax.axhline(star[1], color="C1", ls="--", lw=1)
    ax.set_xlim(0.5, n_total + 0.5); ax.set_ylim(0, TAU_MAX)
    ax.set_xlabel("round"); ax.set_ylabel("toll")
    ax.set_title("Candidate equilibrium per round (dashed = exact Nash)", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")

def make_figures(snapshots):
    n = len(snapshots)
    star = exact_nash()
    ylim = [(-0.3 * v, 1.6 * v) for v in model_quiet(*star)]
    frames = []
    for k, s in enumerate(snapshots, start=1):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=110)
        draw_slice(axes[0, 0], s, 0, ylim[0])
        draw_slice(axes[0, 1], s, 1, ylim[1])
        draw_plane(axes[1, 0], s, legend=False)
        axes[1, 0].set_title("Toll plane: best replies on UCB; shading = GP uncertainty", fontsize=10)
        h, l = axes[1, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=False)
        draw_history(axes[1, 1], s, n)
        nxt = ("candidate already evaluated -> STOP, then validate" if s["seen"] else
               f"run real model at ({s['cand'][0]:.2f}, {s['cand'][1]:.2f})")
        fig.suptitle(f"BES round {s['rnd']}: {len(s['X'])} model runs so far; next: {nxt}",
                     fontsize=12)
        fig.tight_layout(rect=[0, 0.07, 1, 0.97])
        fname = f"vis/bes_round_{k:02d}.png"
        fig.savefig(fname); plt.close(fig)
        frames.append(imageio.imread(fname))
    frames += [frames[-1]] * 2                                # linger on the last frame
    imageio.mimsave("vis/bes_evolution.gif", frames, fps=1.5, loop=0)

    picks = sorted(set([0, n // 2, n - 1]))
    fig, axes = plt.subplots(1, len(picks), figsize=(5.5 * len(picks), 5.2))
    for ax, i in zip(np.atleast_1d(axes), picks):
        draw_plane(ax, snapshots[i], legend=(i == picks[0]))
        ax.set_title(f"Round {snapshots[i]['rnd']} ({len(snapshots[i]['X'])} model runs)", fontsize=10)
    fig.tight_layout(); fig.savefig("vis/bes_summary.png", dpi=130); plt.close(fig)
    print(f"Saved {n} round figures, vis/bes_evolution.gif and vis/bes_summary.png")


if __name__ == "__main__":
    t, R_t, X, R, hist, snaps = bes()
    runs_loop = n_model_calls
    ok, gains = validate_nash(t, R_t)
    make_figures(snaps)

    # toy only: exact Nash from the first-order conditions (for comparison)
    tA_star, tB_star = exact_nash()
    print(f"\nBES result : tA = {t[0]:.3f}, tB = {t[1]:.3f}"
          f"   ({runs_loop} model runs in the loop)")
    print(f"Exact Nash : tA = {tA_star:.3f}, tB = {tB_star:.3f}")
    print(f"Validation : valid eps-local Nash = {ok}  "
          f"(gains from deviating: { {k: round(v, 2) for k, v in gains.items()} })")
    print(f"Total model runs incl. validation: {n_model_calls}")

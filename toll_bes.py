"""
Two roads, two toll authorities: find the Nash equilibrium of the tolls
with Bayesian equilibration (BES).

Travellers go from A to B over road 1 (toll tau1, set by authority 1) or
road 2 (toll tau2, set by authority 2). Each authority maximises its own
toll revenue. Raising one toll pushes traffic onto the other road, so each
authority's best toll depends on the other's toll: it is a game.

Part 1: the "expensive model": route choice + elastic demand
Part 2: the exact Nash equilibrium, as a reference to check BES against
Part 3: one Gaussian Process per player, over BOTH tolls
Part 4: the BES loop: solve the game on the GPs, check with the real model
Part 5: interactive figure and web page
"""
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------------------------------------------------
# 0. Parameters
# ----------------------------------------------------------------------
# Run locally, these defaults apply. The web page toll_bes.html runs this
# same file in the browser (Pyodide, no server) and passes its form values
# in WEB_PARAMS, which override the defaults.
# Times, costs and tolls are all in minutes (a toll is converted to time
# with a value of time), so revenue is in vehicle-minutes.
IN_BROWSER = sys.platform == "emscripten"
P = dict(N=1000, C_MAX=60.0, T0=[10.0, 15.0], A=[0.03, 0.015], TAU_MAX=20.0,
         N_INIT=4, SEED=4, MAX_ITER=10, LENGTH=6.0, DELTA=0.5, EPS=2.0, N_SEEDS=1)
P.update(globals().get("WEB_PARAMS", {}))
for key in ("N_INIT", "SEED", "MAX_ITER", "N_SEEDS"):
    P[key] = int(P[key])
if len(P["T0"]) != 2 or len(P["A"]) != 2:
    raise ValueError("T0 and A need exactly 2 values, one per road.")
if min(P["A"]) <= 0 or P["N_INIT"] < 2 or P["LENGTH"] <= 0 or P["DELTA"] <= 0:
    raise ValueError("A, LENGTH and DELTA must be positive, and N_INIT at least 2.")
if not 1 <= P["N_SEEDS"] <= 200:
    raise ValueError("N_SEEDS must be between 1 and 200.")

# Labels for the web page form: name -> (group, short description)
HELP = dict(N=("Expensive model", "trips if travel were free"),
            C_MAX=("Expensive model", "trip cost at which nobody travels"),
            T0=("Expensive model", "free-flow time of road 1, road 2"),
            A=("Expensive model", "extra minutes per vehicle, road 1, road 2"),
            TAU_MAX=("Expensive model", "highest toll allowed"),
            N_INIT=("Bayesian equilibration", "initial runs (Latin hypercube)"),
            SEED=("Bayesian equilibration", "where the random initial runs go"),
            MAX_ITER=("Bayesian equilibration", "maximum number of iterations"),
            LENGTH=("Bayesian equilibration", "GP length-scale ℓ (toll units)"),
            DELTA=("Bayesian equilibration", "nudge size δ of the check"),
            EPS=("Bayesian equilibration", "tolerance ε on a player's gain"),
            N_SEEDS=("Bayesian equilibration", "1 = one run, more = compare seeds"))

# ----------------------------------------------------------------------
# 1. The "expensive model" (here cheap, but we treat it as a black box)
# ----------------------------------------------------------------------
N, C_MAX, TAU_MAX = P["N"], P["C_MAX"], P["TAU_MAX"]
T0 = np.array(P["T0"], dtype=float)   # free-flow travel time of each road
A = np.array(P["A"], dtype=float)     # congestion: extra minutes per vehicle
# Cost of using road i:      c_i = T0_i + A_i * x_i + tau_i   (x_i = its flow)
# Total demand for trips:    x_1 + x_2 = N * (1 - c / C_MAX)
# Route choice (Wardrop):    every used road costs the same c, and an
#                            unused road would cost more than c.


def flows(tau):
    """User equilibrium with elastic demand, for the toll pair tau.

    Returns the flows x (one per road) and the common trip cost c.
    """
    free = T0 + np.asarray(tau, dtype=float)   # cost of each road when empty
    used = free < C_MAX                         # roads anyone might take
    while used.any():
        # On used roads x_i = (c - free_i) / A_i. Summing and setting equal
        # to the demand N (1 - c / C_MAX) gives one linear equation in c.
        c = (N + np.sum(free[used] / A[used])) / (np.sum(1 / A[used]) + N / C_MAX)
        x = np.where(used, (c - free) / A, 0.0)
        if (x >= 0).all():
            return x, c
        used &= x > 0      # a road with negative flow is too expensive: drop it
    return np.zeros(2), C_MAX


def objectives(tau):
    """Both players' objectives at once: toll revenue R_i = tau_i * x_i.

    This is the function BES treats as expensive (a real traffic model run
    takes an hour). One run gives the result for BOTH players.
    """
    return np.asarray(tau, dtype=float) * flows(tau)[0]


# ----------------------------------------------------------------------
# 2. Reference: the exact Nash equilibrium
# ----------------------------------------------------------------------
# While both roads are used, each flow is linear in the tolls:
#     x_i = alpha_i + beta_i1 * tau_1 + beta_i2 * tau_2
# so R_i = tau_i * x_i is a parabola in the player's own toll. Setting
# dR_i/dtau_i = alpha_i + 2 beta_ii tau_i + beta_ij tau_j = 0 for both
# players gives a 2x2 linear system: its solution is the Nash equilibrium.
def exact_nash():
    S = np.sum(1 / A) + N / C_MAX
    alpha = ((N + np.sum(T0 / A)) / S - T0) / A
    beta = np.outer(1 / A, 1 / (A * S))     # effect of tau_j on x_i through c
    beta[np.diag_indices(2)] -= 1 / A       # plus the direct effect of own toll
    first_order = beta + np.diag(np.diag(beta))
    return np.linalg.solve(first_order, -alpha)


# ======================================================================
# PART 3: One Gaussian Process per player, over both tolls
# ======================================================================
# Exactly the GP of toll_bo.py, with one change: the inputs are toll PAIRS
# (tau1, tau2), so the kernel uses the squared distance between two pairs.
# GP 1 learns R_1(tau1, tau2) and GP 2 learns R_2(tau1, tau2), from the
# same runs of the expensive model.
# ----------------------------------------------------------------------

def rbf(a, b, length, sigma_f=1.0):
    """RBF kernel between toll pairs a (shape (n, 2)) and b (shape (m, 2))."""
    squared_distance = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)
    return sigma_f**2 * np.exp(-0.5 * squared_distance / length**2)


def gp_mean(x_train, y_train, x_test, length, sigma_f=1.0, noise=1e-6):
    """GP posterior mean k*^T K^-1 y (see gp_posterior in toll_bo.py).

    The game is solved on the mean only, so the variance is not computed.
    """
    y_mean, y_std = y_train.mean(), y_train.std() + 1e-12
    y = (y_train - y_mean) / y_std
    K = rbf(x_train, x_train, length, sigma_f)
    # The check runs can repeat earlier points; then K is (almost)
    # singular, and a little more noise on the diagonal fixes that.
    while True:
        try:
            L = np.linalg.cholesky(K + noise * np.eye(len(x_train)))
            break
        except np.linalg.LinAlgError:
            noise *= 10
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    return rbf(x_train, x_test, length, sigma_f).T @ alpha * y_std + y_mean


# ======================================================================
# PART 4: Bayesian equilibration
# ======================================================================
# Candidate tolls: a 121 x 121 grid of toll pairs. GRID2 lists all pairs,
# with GRID2[i * 121 + j] = (G[i], G[j]).
G = np.linspace(0, TAU_MAX, 121)
GRID2 = np.array([(t1, t2) for t1 in G for t2 in G])


def best_responses(R1, R2):
    """Best-response curves on the grid, for surfaces R_p[i, j] = R_p(G[i], G[j]).

    br1[j] = index of player 1's best toll when player 2 charges G[j]
    br2[i] = index of player 2's best toll when player 1 charges G[i]
    """
    return R1.argmax(axis=0), R2.argmax(axis=1)


def solve_game(br1, br2, start):
    """Nash equilibrium by alternating best responses ("diagonalisation").

    Each player in turn picks its best toll given the other's, until
    neither moves. On the GPs this is free; on the real model every best
    response would need many expensive runs. Returns grid indices (i, j).
    """
    i, j = start
    seen = set()
    for _ in range(500):
        i_new = br1[j]
        j_new = br2[i_new]
        if (i_new, j_new) == (i, j) or (i_new, j_new) in seen:
            return i_new, j_new     # converged, or cycling: stop here
        seen.add((i, j))
        i, j = i_new, j_new
    return i, j


def check(candidate):
    """Test a candidate with the real model: each player nudges its own toll.

    Runs the model at the candidate and at 4 nudged points (own toll
    +- DELTA). gain_i is the most player i can win by such a nudge. If no
    player gains more than EPS, the candidate is an eps-local Nash
    equilibrium, the same test as in the MBE paper.
    """
    points = [candidate]
    for i in range(2):
        for d in (-P["DELTA"], P["DELTA"]):
            t = candidate.copy()
            t[i] = np.clip(t[i] + d, 0, TAU_MAX)
            points.append(t)
    points = np.array(points)
    values = np.array([objectives(t) for t in points])   # 5 expensive runs
    gains = np.array([values[1 + 2 * i: 3 + 2 * i, i].max() - values[0, i] for i in range(2)])
    return points, values, gains


# --- Reference (only possible because this toy model is cheap) -----------
nash = exact_nash()
if not ((nash >= 0).all() and (nash <= TAU_MAX).all() and (flows(nash)[0] > 0).all()):
    raise ValueError("With these parameters the Nash equilibrium is not interior "
                     "(a road would be empty or a toll outside 0..TAU_MAX). "
                     "Try other T0, A or TAU_MAX.")
print(f"Exact Nash equilibrium: tau* = ({nash[0]:.2f}, {nash[1]:.2f}), "
      f"R* = ({objectives(nash)[0]:.0f}, {objectives(nash)[1]:.0f})")


def run_bes(seed, verbose=True):
    """One complete BES run, starting from the initial design of `seed`.

    Returns the snapshots (one per iteration, for the figure), the final
    candidate, whether it passed the check, and the number of expensive runs.
    """
    # --- Initial design: a Latin hypercube ------------------------------
    # One point in each row and each column of an N_INIT x N_INIT grid over
    # the toll square, so the first runs are spread out in both tolls. The
    # seed decides where exactly; it is the only thing that differs between
    # seeds.
    rng = np.random.default_rng(seed)
    n0 = P["N_INIT"]
    X = TAU_MAX * np.column_stack([(rng.permutation(n0) + rng.random(n0)) / n0 for _ in range(2)])
    Y = np.array([objectives(t) for t in X])     # Y[:, p] = objective of player p

    # --- The BES loop ----------------------------------------------------
    start = (len(G) // 2, len(G) // 2)
    snapshots = []
    for it in range(1, P["MAX_ITER"] + 1):
        # (a) Fit one GP per player to all runs so far; predict on the grid.
        R_gp = [gp_mean(X, Y[:, p], GRID2, P["LENGTH"]).reshape(len(G), len(G))
                for p in range(2)]

        # (b) Solve the game on the cheap GPs: a candidate equilibrium.
        br1, br2 = best_responses(*R_gp)
        i, j = solve_game(br1, br2, start)
        candidate = np.array([G[i], G[j]])
        start = (i, j)

        # (c) Check the candidate with the real, expensive model (5 runs).
        new_X, new_Y, gains = check(candidate)
        done = (gains < P["EPS"]).all()
        snapshots.append(dict(X=X.copy(), R_gp=R_gp, br=(br1, br2), cand=candidate,
                              new_X=new_X, gains=gains, done=done))
        if verbose:
            print(f"iter {it}: candidate tau = ({candidate[0]:.2f}, {candidate[1]:.2f})  "
                  f"gains = ({gains[0]:+.2f}, {gains[1]:+.2f})  "
                  f"{'eps-local Nash' if done else 'not yet'}")

        # (d) Add the check runs: both GPs improve around the candidate.
        X = np.vstack([X, new_X])
        Y = np.vstack([Y, new_Y])
        if done:
            break
    return snapshots, candidate, done, len(X)


if P["N_SEEDS"] == 1:
    # One run, shown step by step in the figure
    print(f"{P['N_INIT']} initial runs\n")
    snapshots, candidate, done, n_runs = run_bes(P["SEED"])
    print(f"\nBES result: tau = ({candidate[0]:.2f}, {candidate[1]:.2f}) after "
          f"{n_runs} runs of the expensive model"
          + ("" if done else f" (no eps-local Nash within {P['MAX_ITER']} iterations)"))
    print(f"Exact:      tau = ({nash[0]:.2f}, {nash[1]:.2f}), "
          f"distance {np.linalg.norm(candidate - nash):.3f} "
          f"(grid step {G[1] - G[0]:.3f})")
else:
    # Several runs that differ only in their random initial design
    seeds = range(P["SEED"], P["SEED"] + P["N_SEEDS"])
    print(f"{P['N_SEEDS']} seeds ({seeds[0]} to {seeds[-1]}), "
          f"{P['N_INIT']} initial runs each\n")
    results = []
    for seed in seeds:
        snaps, cand, ok, n_runs = run_bes(seed, verbose=False)
        dist_nash = np.linalg.norm(cand - nash)
        results.append(dict(seed=seed, cand=cand, done=ok, runs=n_runs,
                            iters=len(snaps), dist=dist_nash))
        print(f"seed {seed:3d}: {len(snaps):2d} iterations, {n_runs:3d} runs, "
              f"candidate ({cand[0]:.2f}, {cand[1]:.2f}), distance {dist_nash:.3f}"
              + ("" if ok else "  NOT CONVERGED"))
    runs = np.array([r["runs"] for r in results])
    dists = np.array([r["dist"] for r in results])
    n_ok = sum(r["done"] for r in results)
    print(f"\nReached an eps-local Nash: {n_ok} of {len(results)} seeds")
    print(f"Model runs:        median {np.median(runs):.0f}, mean {runs.mean():.1f}, "
          f"range {runs.min()} to {runs.max()}")
    print(f"Distance to Nash:  median {np.median(dists):.3f}, max {dists.max():.3f} "
          f"(grid step {G[1] - G[0]:.3f})")

# ----------------------------------------------------------------------
# 5. Interactive figure: step through the iterations with a slider
# ----------------------------------------------------------------------
# True surfaces and best responses, for reference lines only (BES never
# sees them; computing them takes 121^2 runs of the "expensive" model).
R_true = np.array([objectives(t) for t in GRID2]).reshape(len(G), len(G), 2)
true_br1, true_br2 = best_responses(R_true[..., 0], R_true[..., 1])

C1, C2 = "#1f77b4", "#ff7f0e"
GREY = [[0, "#f8fafc"], [1, "#8d9cad"]]
SHOW = slice(None, None, 2)     # heatmaps drawn on every 2nd grid point


def frame_data(k):
    """All traces for iteration k (1-based), in a fixed order (plotly needs that)."""
    s = snapshots[k - 1]
    traces = []
    for p in range(2):
        legend = p == 0
        traces += [
            go.Heatmap(x=G[SHOW], y=G[SHOW], z=np.round(s["R_gp"][p][SHOW, SHOW].T),
                       colorscale=GREY, showscale=False,
                       hovertemplate="τ₁=%{x:.2f}, τ₂=%{y:.2f}<br>GP mean R=%{z}<extra></extra>"),
            go.Scatter(x=G[true_br1], y=G, name="BR₁ true", showlegend=legend,
                       line=dict(color=C1, dash="dash", width=1.5)),
            go.Scatter(x=G, y=G[true_br2], name="BR₂ true", showlegend=legend,
                       line=dict(color=C2, dash="dash", width=1.5)),
            go.Scatter(x=G[s["br"][0]], y=G, name="BR₁ on GP", showlegend=legend,
                       line=dict(color=C1, width=2.5)),
            go.Scatter(x=G, y=G[s["br"][1]], name="BR₂ on GP", showlegend=legend,
                       line=dict(color=C2, width=2.5)),
            go.Scatter(x=s["X"][:, 0], y=s["X"][:, 1], mode="markers", name="runs so far",
                       showlegend=legend, marker=dict(color="#1b2430", size=6)),
            go.Scatter(x=s["X"][:P["N_INIT"], 0], y=s["X"][:P["N_INIT"], 1], mode="markers",
                       name=f"initial runs (seed {P['SEED']})", showlegend=legend,
                       marker=dict(color="#1b2430", size=12, symbol="square-open",
                                   line=dict(width=2))),
            go.Scatter(x=s["new_X"][:, 0], y=s["new_X"][:, 1], mode="markers",
                       name="check runs", showlegend=legend,
                       marker=dict(color="#d62728", size=9, symbol="circle-open", line=dict(width=2))),
            go.Scatter(x=[s["cand"][0]], y=[s["cand"][1]], mode="markers", name="candidate",
                       showlegend=legend, marker=dict(color="#d62728", size=12, symbol="diamond")),
            go.Scatter(x=[nash[0]], y=[nash[1]], mode="markers", name="exact Nash",
                       showlegend=legend, marker=dict(color="#2ca02c", size=15, symbol="star",
                                                      line=dict(color="white", width=1))),
        ]
    g = np.array([sn["gains"] for sn in snapshots[:k]])
    iters = np.arange(1, k + 1)
    dist = [max(np.linalg.norm(sn["cand"] - nash), 1e-3) for sn in snapshots[:k]]
    traces += [
        go.Scatter(x=iters, y=dist, mode="lines+markers", showlegend=False,
                   line=dict(color="#d62728")),
        go.Scatter(x=iters, y=g[:, 0], mode="lines+markers", name="gain player 1",
                   line=dict(color=C1)),
        go.Scatter(x=iters, y=g[:, 1], mode="lines+markers", name="gain player 2",
                   line=dict(color=C2)),
        go.Scatter(x=[0.5, len(snapshots) + 0.5], y=[P["EPS"]] * 2, mode="lines",
                   name="ε", line=dict(color="#5d6b78", dash="dot")),
    ]
    return traces + road_traces(s["cand"])


# The two roads drawn as arcs from A to B: road 1 above, road 2 below.
ARC = np.linspace(0, 1, 41)


def road_traces(tau):
    """Network diagram at the toll pair tau: line width shows each road's flow."""
    x, c = flows(tau)
    traces = []
    for i, (sign, color) in enumerate([(1, C1), (-1, C2)]):
        traces.append(go.Scatter(x=10 * ARC, y=sign * 1.4 * np.sin(np.pi * ARC), mode="lines",
                                 showlegend=False, hoverinfo="skip",
                                 line=dict(color=color, width=1 + 18 * x[i] / N)))
    road_time = T0 + A * x     # travel time without the toll
    labels = [f"<b>Road 1</b>: toll {tau[0]:.2f}, {x[0]:.0f} veh<br>"
              f"time {road_time[0]:.1f} min, R₁ = {tau[0] * x[0]:.0f}",
              f"<b>Road 2</b>: toll {tau[1]:.2f}, {x[1]:.0f} veh<br>"
              f"time {road_time[1]:.1f} min, R₂ = {tau[1] * x[1]:.0f}",
              f"trip cost {c:.1f} min<br>{N - x.sum():.0f} of {N:g} stay home"]
    traces.append(go.Scatter(x=[5, 5, 5], y=[2.4, -2.2, -4.0], mode="text", text=labels,
                             showlegend=False, hoverinfo="skip",
                             textfont=dict(size=10, color="#1b2430")))
    traces.append(go.Scatter(x=[0, 10], y=[0, 0], mode="markers+text", text=["A", "B"],
                             showlegend=False, hoverinfo="skip",
                             marker=dict(size=20, color="#1b2430"),
                             textfont=dict(color="white", size=12)))
    return traces


def frame_title(k):
    s = snapshots[k - 1]
    verdict = "ε-local Nash ✓" if s["done"] else "not yet an equilibrium"
    return (f"<b>Iteration {k}</b>: candidate τ̂ = ({s['cand'][0]:.2f}, {s['cand'][1]:.2f}), "
            f"gains ({s['gains'][0]:+.1f}, {s['gains'][1]:+.1f}) vs ε = {P['EPS']:g}: {verdict}"
            f"<br>Seed {P['SEED']} · {len(s['X']) + 5} runs of the expensive model so far · "
            f"exact Nash τ* = ({nash[0]:.2f}, {nash[1]:.2f})")


def run_figure():
    """One BES run, with a slider over its iterations."""
    # Top row: the two heatmaps (3 grid columns each). Bottom row: the roads,
    # the distance to Nash and the gains (2 grid columns each).
    fig = make_subplots(rows=2, cols=6, row_heights=[0.64, 0.36],
                        vertical_spacing=0.13, horizontal_spacing=0.07,
                        specs=[[{"colspan": 3}, None, None, {"colspan": 3}, None, None],
                               [{"colspan": 2}, None, {"colspan": 2}, None, {"colspan": 2}, None]],
                        subplot_titles=("Player 1: GP mean of R₁(τ₁, τ₂)",
                                        "Player 2: GP mean of R₂(τ₁, τ₂)",
                                        "Roads at the candidate tolls",
                                        "Distance to exact Nash",
                                        "Nudge gains (need < ε)"))
    cells = [(1, 1)] * 10 + [(1, 4)] * 10 + [(2, 3), (2, 5), (2, 5), (2, 5)] + [(2, 1)] * 4
    for trace, (r, c) in zip(frame_data(1), cells):
        fig.add_trace(trace, row=r, col=c)

    K = len(snapshots)
    fig.frames = [go.Frame(name=str(k), data=frame_data(k),
                           layout=dict(title_text=frame_title(k))) for k in range(1, K + 1)]

    for c in (1, 4):
        fig.update_xaxes(range=[0, TAU_MAX], title_text="τ₁ (road 1)", title_standoff=2, row=1, col=c)
        fig.update_yaxes(range=[0, TAU_MAX], title_text="τ₂ (road 2)" if c == 1 else None,
                         title_standoff=2, row=1, col=c)
    for c in (3, 5):
        fig.update_xaxes(range=[0.5, K + 0.5], dtick=1, title_text="iteration",
                         title_standoff=2, row=2, col=c)
    fig.update_xaxes(range=[-0.8, 10.8], visible=False, row=2, col=1)
    fig.update_yaxes(range=[-4.8, 3.1], visible=False, row=2, col=1)
    all_gains = np.array([s["gains"] for s in snapshots])
    fig.update_yaxes(type="log", range=[-3.2, np.log10(TAU_MAX) + 0.3], row=2, col=3)
    fig.update_yaxes(range=[min(all_gains.min(), 0) * 1.1 - 1, max(all_gains.max(), P["EPS"]) * 1.1],
                     row=2, col=5)

    step_args = dict(frame=dict(duration=0, redraw=True), mode="immediate")
    fig.update_layout(
        # No fixed height: the web page sizes the figure to fit the screen
        title=dict(text=frame_title(1), font=dict(size=14), x=0.06, y=0.985),
        template="plotly_white", font=dict(family="Arial, Helvetica, sans-serif"),
        margin=dict(t=62, b=105, l=60, r=10),
        legend=dict(font=dict(size=11)),
        updatemenus=[dict(type="buttons", x=0, y=-0.115, xanchor="left", yanchor="top",
                          direction="left", pad=dict(t=0, r=10),
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(frame=dict(duration=1500, redraw=True),
                                                         fromcurrent=True)]),
                                   dict(label="❚❚ Pause", method="animate",
                                        args=[[None], step_args])])],
        sliders=[dict(x=0.24, y=-0.075, len=0.76, yanchor="top", pad=dict(t=0, b=0),
                      currentvalue=dict(prefix="BES iteration: ", font=dict(size=14)),
                      steps=[dict(label=str(k), method="animate", args=[[str(k)], step_args])
                             for k in range(1, K + 1)])],
    )
    return fig


def seeds_figure():
    """Summary of several BES runs that differ only in their seed."""
    fig = make_subplots(rows=2, cols=2, column_widths=[0.55, 0.45],
                        vertical_spacing=0.16, horizontal_spacing=0.1,
                        specs=[[{"rowspan": 2}, {}], [None, {}]],
                        subplot_titles=("Final candidates (zoom around the exact Nash)",
                                        "Model runs needed per seed",
                                        "Cost against accuracy, per seed"))
    # Several seeds often end on the same grid point: draw each point once,
    # sized by how many seeds ended there.
    for ok, name, color, symbol in [(True, "passed the check", "#d62728", "diamond"),
                                    (False, "not converged", "#7f8c8d", "x")]:
        group = {}
        for r in results:
            if r["done"] == ok:
                group.setdefault(tuple(np.round(r["cand"], 4)), []).append(r["seed"])
        if not group:
            continue
        pts = np.array(list(group))
        counts = np.array([len(v) for v in group.values()])
        fig.add_trace(go.Scatter(
            x=pts[:, 0], y=pts[:, 1], mode="markers", name=name,
            marker=dict(color=color, symbol=symbol, size=8 + 4 * np.sqrt(counts),
                        line=dict(color="white", width=1)),
            text=[f"{len(v)} seed(s): {', '.join(map(str, v))}" for v in group.values()],
            hovertemplate="τ₁=%{x:.2f}, τ₂=%{y:.2f}<br>%{text}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=G[true_br1], y=G, name="BR₁ true",
                             line=dict(color=C1, dash="dash", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=G, y=G[true_br2], name="BR₂ true",
                             line=dict(color=C2, dash="dash", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[nash[0]], y=[nash[1]], mode="markers", name="exact Nash",
                             marker=dict(color="#2ca02c", size=16, symbol="star",
                                         line=dict(color="white", width=1))), row=1, col=1)
    # Ring the first seed: the single run shown step by step when N_SEEDS = 1
    first = results[0]
    ring = dict(color="#1b2430", size=20, symbol="circle-open", line=dict(width=2))
    fig.add_trace(go.Scatter(x=[first["cand"][0]], y=[first["cand"][1]], mode="markers",
                             name=f"seed {first['seed']} (the single run)", marker=ring),
                  row=1, col=1)

    values, counts = np.unique(runs, return_counts=True)
    fig.add_trace(go.Bar(x=values, y=counts, marker_color="#1b2430", showlegend=False,
                         hovertemplate="%{x} runs: %{y} seed(s)<extra></extra>"), row=1, col=2)
    colors = ["#d62728" if r["done"] else "#7f8c8d" for r in results]
    fig.add_trace(go.Scatter(x=runs, y=dists, mode="markers", showlegend=False,
                             marker=dict(color=colors, size=8, opacity=0.7),
                             text=[f"seed {r['seed']}" for r in results],
                             hovertemplate="%{text}: %{x} runs, distance %{y:.3f}<extra></extra>"),
                  row=2, col=2)
    fig.add_trace(go.Scatter(x=[first["runs"]], y=[first["dist"]], mode="markers",
                             showlegend=False, marker=ring, hoverinfo="skip"), row=2, col=2)

    zoom = max(1.0, 1.15 * dists.max())
    fig.update_xaxes(range=[nash[0] - zoom, nash[0] + zoom], title_text="τ₁ (road 1)",
                     title_standoff=2, row=1, col=1)
    fig.update_yaxes(range=[nash[1] - zoom, nash[1] + zoom], title_text="τ₂ (road 2)",
                     title_standoff=2, scaleanchor="x", row=1, col=1)
    # Runs always come in steps of 5 (one check per iteration): tick those
    # values, with what they would cost at one hour per traffic-model run.
    hours = [f"{v}<br>≈ {v} h" for v in values]
    fig.update_xaxes(title_text="model runs (≈ hours of a real traffic model)", tickvals=values,
                     ticktext=hours, title_standoff=2, row=1, col=2)
    fig.update_yaxes(title_text="seeds", row=1, col=2)
    fig.update_xaxes(title_text="model runs", tickvals=values, title_standoff=2, row=2, col=2)
    fig.update_yaxes(title_text="distance to Nash", rangemode="tozero", row=2, col=2)
    fig.update_layout(
        title=dict(text=f"<b>Same method, {len(results)} different random starts</b> "
                        f"(seeds {results[0]['seed']} to {results[-1]['seed']}): "
                        f"{n_ok} of {len(results)} reached an ε-local Nash"
                        f"<br>At 1 h per model run: {runs.min()} h (luckiest seed) to "
                        f"{runs.max()} h (unluckiest), median {np.median(runs):.0f} h · "
                        f"distance to Nash {dists.min():.2f} to {dists.max():.2f}",
                   font=dict(size=14), x=0.06, y=0.985),
        template="plotly_white", font=dict(family="Arial, Helvetica, sans-serif"),
        margin=dict(t=70, b=50, l=60, r=10), legend=dict(font=dict(size=11)))
    return fig


fig = run_figure() if P["N_SEEDS"] == 1 else seeds_figure()

# ----------------------------------------------------------------------
# 6. Web page: toll_bes.html, built around this file's own source
# ----------------------------------------------------------------------
GUIDE_RUN = """
<ul>
<li><b>Heatmaps.</b> Each player's GP estimate of its revenue (darker =
higher). Dots: model runs so far. Red circles: this iteration's check runs.</li>
<li><b>Best-response curves.</b> Blue: player 1's best τ₁ for each τ₂. Orange:
player 2's best τ₂ for each τ₁. Solid = from the GPs, dashed = true (BES never
sees these).</li>
<li><b>Equilibrium</b> is where the curves cross: no player wants to change its
toll. Red diamond = BES candidate, green star = exact answer. Over the
iterations the solid curves move onto the dashed ones.</li>
<li><b>Roads.</b> Traffic at the candidate tolls (thicker = more vehicles). A
cheaper road draws traffic from the other.</li>
<li><b>Nudge gains.</b> The most each player wins by moving its own toll ±δ in
the real model. Both below ε (dotted) = ε-local Nash equilibrium: BES stops.
The distance to the exact Nash is only known because this toy model can be
solved exactly.</li>
<li><b>Seed.</b> BES starts from N_INIT random toll pairs (open squares); SEED
decides where. Everything after follows from them, so another seed gives a
different path, cost and answer. This run may be lucky: set N_SEEDS to 50 to
see the spread.</li>
</ul>
"""

GUIDE_SEEDS = """
<ul>
<li><b>What a seed is.</b> BES starts by running the model at N_INIT toll pairs
placed at random. The seed fixes that randomness: same seed, same run. Here
BES ran once per seed; nothing else changed.</li>
<li><b>What is at stake.</b> One run can be lucky. Each run of a real traffic
model takes about an hour, so the spread in runs is a spread in hours of
computing, and the spread in distance is how wrong the answer can be.
Reporting a single seed hides both.</li>
<li><b>Left.</b> Where each seed's final answer landed, around the exact Nash
(star). Bigger diamond = more seeds; hover to see which. Grey cross = never
passed the check.</li>
<li><b>Right.</b> Top: runs (hours) each seed needed. Bottom: cost against
accuracy. The ring marks the seed you step through when N_SEEDS = 1.</li>
<li><b>Verdict.</b> Diamonds close to the star and a narrow spread of runs:
BES is reliable for this game.</li>
</ul>
"""
GUIDE = GUIDE_RUN if P["N_SEEDS"] == 1 else GUIDE_SEEDS

if not IN_BROWSER:
    from webpage import write_page
    write_page(__file__, "toll_bes.html",
               title="Bayesian equilibration: two roads, two toll authorities",
               guide=GUIDE,
               params=P, param_help=HELP, fig=fig)

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
         N_INIT=4, SEED=4, MAX_ITER=10, LENGTH=6.0, DELTA=0.5, EPS=2.0)
P.update(globals().get("WEB_PARAMS", {}))
for key in ("N_INIT", "SEED", "MAX_ITER"):
    P[key] = int(P[key])
if len(P["T0"]) != 2 or len(P["A"]) != 2:
    raise ValueError("T0 and A need exactly 2 values, one per road.")
if min(P["A"]) <= 0 or P["N_INIT"] < 2 or P["LENGTH"] <= 0 or P["DELTA"] <= 0:
    raise ValueError("A, LENGTH and DELTA must be positive, and N_INIT at least 2.")

# Labels for the web page form: name -> (group, short description)
HELP = dict(N=("Expensive model", "trips if travel were free"),
            C_MAX=("Expensive model", "trip cost at which nobody travels"),
            T0=("Expensive model", "free-flow time of road 1, road 2"),
            A=("Expensive model", "extra minutes per vehicle, road 1, road 2"),
            TAU_MAX=("Expensive model", "highest toll allowed"),
            N_INIT=("Bayesian equilibration", "initial runs (Latin hypercube)"),
            SEED=("Bayesian equilibration", "random seed of the initial runs"),
            MAX_ITER=("Bayesian equilibration", "maximum number of iterations"),
            LENGTH=("Bayesian equilibration", "GP length-scale ℓ (toll units)"),
            DELTA=("Bayesian equilibration", "nudge size δ of the check"),
            EPS=("Bayesian equilibration", "tolerance ε on a player's gain"))

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


# --- Initial design: a Latin hypercube ----------------------------------
# One point in each row and each column of an N_INIT x N_INIT grid over the
# toll square, so the first runs are spread out in both tolls.
rng = np.random.default_rng(P["SEED"])
n0 = P["N_INIT"]
X = TAU_MAX * np.column_stack([(rng.permutation(n0) + rng.random(n0)) / n0 for _ in range(2)])
Y = np.array([objectives(t) for t in X])     # Y[:, p] = objective of player p

# --- Reference (only possible because this toy model is cheap) -----------
nash = exact_nash()
if not ((nash >= 0).all() and (nash <= TAU_MAX).all() and (flows(nash)[0] > 0).all()):
    raise ValueError("With these parameters the Nash equilibrium is not interior "
                     "(a road would be empty or a toll outside 0..TAU_MAX). "
                     "Try other T0, A or TAU_MAX.")
print(f"Exact Nash equilibrium: tau* = ({nash[0]:.2f}, {nash[1]:.2f}), "
      f"R* = ({objectives(nash)[0]:.0f}, {objectives(nash)[1]:.0f})")
print(f"{n0} initial runs\n")

# --- The BES loop ----------------------------------------------------------
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
    print(f"iter {it}: candidate tau = ({candidate[0]:.2f}, {candidate[1]:.2f})  "
          f"gains = ({gains[0]:+.2f}, {gains[1]:+.2f})  "
          f"{'eps-local Nash' if done else 'not yet'}")

    # (d) Add the check runs to the data: both GPs improve around the candidate.
    X = np.vstack([X, new_X])
    Y = np.vstack([Y, new_Y])
    if done:
        break

print(f"\nBES result: tau = ({candidate[0]:.2f}, {candidate[1]:.2f}) after "
      f"{len(X)} runs of the expensive model"
      + ("" if done else f" (no eps-local Nash within {P['MAX_ITER']} iterations)"))
print(f"Exact:      tau = ({nash[0]:.2f}, {nash[1]:.2f}), "
      f"distance {np.linalg.norm(candidate - nash):.3f} "
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
iters = np.arange(1, len(snapshots) + 1)
dist = [max(np.linalg.norm(s["cand"] - nash), 1e-3) for s in snapshots]


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
    traces += [
        go.Scatter(x=iters[:k], y=dist[:k], mode="lines+markers", showlegend=False,
                   line=dict(color="#d62728")),
        go.Scatter(x=iters[:k], y=g[:, 0], mode="lines+markers", name="gain player 1",
                   line=dict(color=C1)),
        go.Scatter(x=iters[:k], y=g[:, 1], mode="lines+markers", name="gain player 2",
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
            f"<br>{len(s['X']) + 5} runs of the expensive model so far · "
            f"exact Nash τ* = ({nash[0]:.2f}, {nash[1]:.2f})")


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
cells = [(1, 1)] * 9 + [(1, 4)] * 9 + [(2, 3), (2, 5), (2, 5), (2, 5)] + [(2, 1)] * 4
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

# ----------------------------------------------------------------------
# 6. Web page: toll_bes.html, built around this file's own source
# ----------------------------------------------------------------------
GUIDE = """
<ul>
<li><b>Heatmaps.</b> Each player's GP estimate of its revenue for every pair of
tolls (darker = higher). Black dots: model runs so far. Red circles: this
iteration's 5 check runs.</li>
<li><b>Best-response curves.</b> Blue: player 1's best τ₁ for each τ₂. Orange:
player 2's best τ₂ for each τ₁. Solid = from the GPs, dashed = true (BES never
sees these).</li>
<li><b>Equilibrium</b> is where the curves cross: no player wants to change its
toll. Red diamond = BES candidate, green star = exact answer. Over the
iterations the solid curves move onto the dashed ones.</li>
<li><b>Roads.</b> Traffic at the candidate tolls (thicker = more vehicles). A
cheaper road draws traffic from the other, so each best toll depends on the
other authority's toll.</li>
<li><b>Distance to exact Nash.</b> Only known because this toy model can be
solved exactly.</li>
<li><b>Nudge gains.</b> The most each player wins by moving its own toll ±δ in
the real model. Both below ε (dotted) = ε-local Nash equilibrium: BES stops.
Precision is limited by δ, ε and the toll grid, so the candidate can still be
a few tenths from the exact answer.</li>
</ul>
"""

if not IN_BROWSER:
    from webpage import write_page
    write_page(__file__, "toll_bes.html",
               title="Bayesian equilibration: two roads, two toll authorities",
               guide=GUIDE,
               params=P, param_help=HELP, fig=fig)

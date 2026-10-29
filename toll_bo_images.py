"""
Single road, single toll operator: revenue maximisation.

Part 1: exact solution (very doable by hand on paper as well)
Part 2: Bayesian Optimisation with a hand-written Gaussian Process
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
import imageio.v2 as imageio

# ----------------------------------------------------------------------
# 1. The "expensive model" (here cheap, but we treat it as a black box)
# ----------------------------------------------------------------------
N, TAU_MAX = 1000, 10.0
#tau = toll level
#N = total amount of travelers (if tau = 0)
#TAU_MAX = toll level for which no travelers use the road

def demand(tau):
    """Number of drivers that still use the road at toll tau."""
    return N * (1 - tau / TAU_MAX)


def revenue(tau):
    """Operator's objective: R(tau) = tau * D(tau)."""
    return tau * demand(tau)


# Analytical optimum for R = N*tau*(1 - tau/tau_max): tau* = tau_max / 2
tau_star = TAU_MAX / 2
print(f"Exact solution: tau* = {tau_star:.2f}, R* = {revenue(tau_star):.1f}")


# ======================================================================
# PART 2: The Gaussian Process (GP)
# ======================================================================
# A GP is a probability distribution over functions. Given a few observed
# points (toll, revenue), it answers for ANY toll tau:
#     "My best guess for R(tau) is `mean`, and I am uncertain by `std`."
# Close to observed points the std is small; far from them it is large.
#
# Two ingredients define a GP:
#   1. A kernel: says how strongly the revenue at two tolls is related.
#   2. The data: the (toll, revenue) pairs evaluated so far.
# ----------------------------------------------------------------------
 
def rbf(a, b, length, sigma_f):
    """RBF (squared-exponential) kernel: similarity between two sets of tolls.
 
    For every pair (a_i, b_j) it returns
        k(a_i, b_j) = sigma_f^2 * exp( -(a_i - b_j)^2 / (2 * length^2) )
 
    Interpretation:
      * Equal tolls (a_i == b_j)  ->  k = sigma_f^2, the maximum: the two
        revenues are perfectly correlated (they are the same quantity).
      * Tolls close together      ->  k close to sigma_f^2: similar revenue.
      * Tolls far apart           ->  k close to 0: knowing one tells you
        almost nothing about the other.
 
    Parameters
    ----------
    a, b    : 1-D arrays of tolls (lengths n and m)
    length  : length-scale. How far apart two tolls can be and still be
              "related". Small = wiggly functions, large = smooth ones.
    sigma_f : prior standard deviation (vertical scale) of the function.
 
    Returns
    -------
    (n, m) matrix whose entry (i, j) is k(a_i, b_j).
    """
    # a[:, None] has shape (n, 1) and b[None, :] has shape (1, m).
    # Subtracting broadcasts to an (n, m) matrix of all pairwise differences.
    squared_distance = (a[:, None] - b[None, :]) ** 2
    return sigma_f**2 * np.exp(-0.5 * squared_distance / length**2)
 
 
def gp_posterior(x_train, y_train, x_test, length=2.0, sigma_f=1.0, noise=1e-6):
    """Predict the revenue at new tolls `x_test` from the evaluations so far.
 
    This is the GP "posterior": the prior (kernel) updated with the data.
    The formulas, with K = kernel matrix of the training tolls,
    k* = kernel between training tolls and a test toll, y = training revenues:
 
        mean(x*) = k*^T  K^-1  y
        var(x*)  = k(x*, x*) - k*^T  K^-1  k*
 
    Reading the formulas:
      * mean: a weighted sum of the observed revenues. Observations at tolls
        close to x* get the biggest weights (that is what k* encodes).
      * var : starts at the prior uncertainty k(x*, x*) = sigma_f^2 and
        SHRINKS by an amount that depends on how much the data tells us
        about x*.  Near data -> large reduction -> small uncertainty.
 
    Parameters
    ----------
    x_train : tolls evaluated so far, shape (n,)
    y_train : revenues observed at those tolls, shape (n,)
    x_test  : tolls at which we want a prediction, shape (m,)
    length, sigma_f : kernel hyperparameters (see `rbf`). Fixed here;
              a full implementation would fit them to the data.
    noise   : tiny value added to the diagonal of K. It (a) models
              measurement noise (almost none here) and (b) keeps K
              invertible numerically.
 
    Returns
    -------
    mean, std : arrays of shape (m,), the prediction and its uncertainty
                at every test toll, in the original revenue units.
    """
    # --- Step 1: standardise the revenues --------------------------------
    # The kernel assumes a function with typical size ~ sigma_f = 1, but
    # revenues are in the thousands. Rescaling y to zero mean / unit spread
    # makes sigma_f = 1 a sensible choice. We undo this at the very end.
    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-12          # +1e-12 avoids dividing by zero
    y = (y_train - y_mean) / y_std
 
    # --- Step 2: build the kernel matrices -------------------------------
    # K  : (n, n) similarity between all pairs of evaluated tolls
    # Ks : (n, m) similarity between evaluated tolls and test tolls (= k*)
    K = rbf(x_train, x_train, length, sigma_f) + noise * np.eye(len(x_train))
    Ks = rbf(x_train, x_test, length, sigma_f)
 
    # --- Step 3: solve K^-1 y without explicitly inverting K -------------
    # Cholesky: K = L L^T with L lower-triangular. Two cheap triangular
    # solves then give alpha = K^-1 y. This is faster and numerically more
    # stable than np.linalg.inv(K).
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
 
    # --- Step 4: posterior mean = k*^T K^-1 y ----------------------------
    mean = Ks.T @ alpha
 
    # --- Step 5: posterior variance = k(x*,x*) - k*^T K^-1 k* ------------
    # With v = L^-1 k*, we have k*^T K^-1 k* = v^T v, so the second term is
    # the column-wise sum of squares of v.
    prior_var = sigma_f**2 * np.ones(len(x_test))   # k(x*, x*) = sigma_f^2
    v = np.linalg.solve(L, Ks)
    var = prior_var - np.sum(v**2, axis=0)
    var = np.clip(var, 1e-12, None)   # rounding can give tiny negatives
 
    # --- Step 6: undo the standardisation --------------------------------
    return mean * y_std + y_mean, np.sqrt(var) * y_std
 
 
# ======================================================================
# PART 3: Bayesian Optimisation
# ======================================================================
# The GP tells us what we believe about R(tau) everywhere. The question now:
# WHERE should we spend our next expensive evaluation?
#
# Two competing wishes:
#   * EXPLOITATION: go where the GP predicts a high revenue.
#   * EXPLORATION : go where the GP is very uncertain; the revenue there
#                   might be surprisingly high.
# An "acquisition function" scores every candidate toll by balancing both.
# We use Expected Improvement (EI).
# ----------------------------------------------------------------------
 
def expected_improvement(mean, std, best, xi=0.01):
    """Expected Improvement (EI) for MAXIMISATION.
 
    For a candidate toll the GP says R ~ Normal(mean, std^2). EI is the
    expected amount by which R beats the best revenue seen so far
    (`best`), counting only improvements (a worse result is worth 0, not
    negative, because we keep the best point anyway).
 
    Closed form, with z = (mean - best - xi) / std:
 
        EI = (mean - best - xi) * Phi(z)  +  std * phi(z)
             \\_________ exploitation __/    \\__ exploration __/
 
    where Phi and phi are the standard normal CDF and density.
      * First term: large when the predicted mean already beats `best`.
      * Second term: large when `std` is large (uncertain = possible upside).
 
    Parameters
    ----------
    mean, std : GP prediction at the candidate tolls (arrays)
    best      : highest revenue observed so far (a number)
    xi        : small margin; an improvement must exceed `best` by about xi
                to count. Discourages pointless tiny refinements.
 
    Returns
    -------
    EI at every candidate toll (array). Higher = more worth evaluating.
    """
    z = (mean - best - xi) / std
    return (mean - best - xi) * norm.cdf(z) + std * norm.pdf(z)
 
 
# --- Setup of the optimisation -----------------------------------------
# Candidate tolls: the GP and EI are cheap, so we can simply check a fine
# grid of 501 tolls between 0 and TAU_MAX and pick the best one.
grid = np.linspace(0, TAU_MAX, 501)
 
# Initial design: BO needs a few evaluations to start. These 3 spread-out
# tolls are the only "free" information; every later one is chosen by EI.
x_obs = np.array([1.0, 4.0, 8.0])     # tolls evaluated so far
y_obs = revenue(x_obs)                # their revenues (3 expensive calls)
N_ITER = 6                            # number of BO iterations
 
snapshots = []   # store each iteration's state, only used for plotting
 
# --- The BO loop ---------------------------------------------------------
for it in range(N_ITER):
    # (a) Fit the GP to all data so far and predict on the whole grid.
    #     Cheap: only linear algebra on a tiny matrix.
    mean, std = gp_posterior(x_obs, y_obs, grid)
 
    # (b) Score every candidate toll with EI, relative to the best
    #     revenue observed so far.
    ei = expected_improvement(mean, std, y_obs.max())
 
    # (c) Pick the toll with the highest EI: the most promising place
    #     to spend one expensive evaluation.
    x_next = grid[np.argmax(ei)]
 
    snapshots.append((x_obs.copy(), y_obs.copy(), mean, std, ei, x_next))
    print(f"iter {it}: evaluate tau = {x_next:.2f}  "
          f"(best so far: tau = {x_obs[np.argmax(y_obs)]:.2f}, "
          f"R = {y_obs.max():.1f})")
 
    # (d) Evaluate the real, expensive model at x_next (the ONLY expensive
    #     call per iteration) and add the result to the data.
    x_obs = np.append(x_obs, x_next)
    y_obs = np.append(y_obs, revenue(x_next))
    # Back to (a): the GP now includes this new information.
 
# --- Result ----------------------------------------------------------------
best = np.argmax(y_obs)
print(f"\nBO result: tau = {x_obs[best]:.3f}, R = {y_obs[best]:.1f} "
      f"after {len(x_obs)} evaluations")
print(f"Exact:     tau = {tau_star:.3f}, R = {revenue(tau_star):.1f}")
 
# Why does it stop making progress? Once a point near the optimum is known,
# the GP is confident nearby and EI is tiny everywhere, so argmax keeps
# returning (almost) the same toll. Repeated picks signal convergence.

# ----------------------------------------------------------------------
# 4. Plot every iteration + summary figure + GIF
# ----------------------------------------------------------------------

# Global limits for smooth animation
# all_ei_max = max(np.max(snap[4]) for snap in snapshots)

Y_MIN = -0.05 * revenue(tau_star)
Y_MAX = 1.15 * revenue(tau_star)

EI_MAX = max(np.max(snap[4]) for snap in snapshots)

gif_frames = []

for k in range(N_ITER):

    xo, yo, m, s, ei, xn = snapshots[k]

    fig_iter, (ax, ax_ei) = plt.subplots(
        2, 1,
        figsize=(8, 6),
        dpi=150,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # GP panel
    ax.set_xlim(0, TAU_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)

    # EI panel
    ax_ei.set_yscale("log")
    ax_ei.set_xlim(0, TAU_MAX)
    ax_ei.set_ylim(1e-4, EI_MAX)

    # GP fit
    ax.plot(grid, revenue(grid), "k--", lw=2, label="true R(τ)")
    ax.plot(grid, m, "C0", lw=2, label="GP mean")

    ax.fill_between(
        grid,
        m - 2 * s,
        m + 2 * s,
        color="C0",
        alpha=0.2,
        label="±2σ"
    )

    ax.scatter(
        xo,
        yo,
        c="C3",
        s=60,
        zorder=5,
        label="evaluations"
    )

    ax.axvline(
        xn,
        color="C2",
        ls=":",
        lw=2,
        label=f"next τ = {xn:.2f}"
    )

    ax.set_ylabel("Revenue")
    ax.set_title(
        f"Bayesian Optimisation Iteration {k}\n"
        f"{len(xo)} evaluations"
    )

    ax.legend(fontsize=8, loc="upper left")

    # EI
    ei_plot = np.maximum(ei, 1e-4)

    ax_ei.plot(grid, ei_plot, "C2", lw=2)
    ax_ei.fill_between(grid, 1e-4, ei_plot,
                    color="C2", alpha=0.25)
    # ax_ei.plot(grid, ei, "C2", lw=2)
    # ax_ei.fill_between(grid, 0, ei, color="C2", alpha=0.25)

    ax_ei.axvline(
        xn,
        color="C2",
        ls=":"
    )

    ax_ei.set_ylabel("EI")
    ax_ei.set_xlabel("Toll τ")

    plt.tight_layout()

    fname = f"vis/bo_iteration_{k:02d}.png"

    plt.savefig(
        fname,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig_iter)

    gif_frames.append(imageio.imread(fname))

# ----------------------------------------------------------------------
# Create animated GIF
# ----------------------------------------------------------------------

imageio.mimsave(
    "vis/toll_bo_evolution.gif",
    gif_frames,
    fps=1.5,
    loop=0
)

print(
    f"Saved {N_ITER} iteration figures "
    f"and toll_bo_evolution.gif"
)

# ----------------------------------------------------------------------
# Original summary figure (first, middle, last iteration)
# ----------------------------------------------------------------------

picks = [0, N_ITER // 2, N_ITER - 1]

fig, axes = plt.subplots(
    2,
    3,
    figsize=(14, 6),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]}
)

for col, k in enumerate(picks):

    xo, yo, m, s, ei, xn = snapshots[k]

    ax = axes[0, col]
    ax_ei = axes[1, col]

    ax.plot(grid, revenue(grid), "k--", label="true R(τ)")
    ax.plot(grid, m, "C0", label="GP mean")

    ax.fill_between(
        grid,
        m - 2 * s,
        m + 2 * s,
        color="C0",
        alpha=0.2,
        label="±2σ"
    )

    ax.scatter(
        xo,
        yo,
        c="C3",
        zorder=5,
        label="evaluations"
    )

    ax.axvline(
        xn,
        color="C2",
        ls=":",
        label="next τ"
    )

    ax.set_title(
        f"After {len(xo)} evaluations"
    )

    ax_ei.plot(
        grid,
        ei,
        "C2"
    )

    ax_ei.set_xlabel("toll τ")

    if col == 0:
        ax.set_ylabel("revenue")
        ax_ei.set_ylabel("EI")
        ax.legend(
            fontsize=8,
            loc="lower center"
        )

plt.tight_layout()
plt.savefig("vis/toll_bo.png", dpi=130)
plt.close()

print("Saved toll_bo.png")
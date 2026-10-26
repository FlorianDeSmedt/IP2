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


# ----------------------------------------------------------------------
# 2. Gaussian Process (RBF kernel), from scratch
# ----------------------------------------------------------------------
def rbf(a, b, length, sigma_f):
    d2 = (a[:, None] - b[None, :]) ** 2
    return sigma_f**2 * np.exp(-0.5 * d2 / length**2)


def gp_posterior(x_train, y_train, x_test, length=2.0, sigma_f=1.0, noise=1e-6):
    """Posterior mean and std at x_test (targets are standardised internally)."""
    mu_y, sd_y = y_train.mean(), y_train.std() + 1e-12
    y = (y_train - mu_y) / sd_y

    K = rbf(x_train, x_train, length, sigma_f) + noise * np.eye(len(x_train))
    Ks = rbf(x_train, x_test, length, sigma_f)
    Kss_diag = sigma_f**2 * np.ones(len(x_test))

    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    mean = Ks.T @ alpha
    v = np.linalg.solve(L, Ks)
    var = np.clip(Kss_diag - np.sum(v**2, axis=0), 1e-12, None)

    return mean * sd_y + mu_y, np.sqrt(var) * sd_y


def expected_improvement(mean, std, best, xi=0.01):
    """EI for maximisation."""
    z = (mean - best - xi) / std
    return (mean - best - xi) * norm.cdf(z) + std * norm.pdf(z)


# ----------------------------------------------------------------------
# 3. Bayesian Optimisation loop
# ----------------------------------------------------------------------
grid = np.linspace(0, TAU_MAX, 501)           # candidate tolls
x_obs = np.array([1.0, 4.0, 8.0])             # initial evaluations
y_obs = revenue(x_obs)
N_ITER = 6

snapshots = []
for it in range(N_ITER):
    mean, std = gp_posterior(x_obs, y_obs, grid)
    ei = expected_improvement(mean, std, y_obs.max())
    x_next = grid[np.argmax(ei)]
    snapshots.append((x_obs.copy(), y_obs.copy(), mean, std, ei, x_next))

    print(f"iter {it}: evaluate tau = {x_next:.2f}  "
          f"(best so far: tau = {x_obs[np.argmax(y_obs)]:.2f}, R = {y_obs.max():.1f})")

    x_obs = np.append(x_obs, x_next)           # one "expensive" evaluation
    y_obs = np.append(y_obs, revenue(x_next))

best = np.argmax(y_obs)
print(f"\nBO result: tau = {x_obs[best]:.3f}, R = {y_obs[best]:.1f} "
      f"after {len(x_obs)} evaluations")
print(f"Exact:     tau = {tau_star:.3f}, R = {revenue(tau_star):.1f}")

# ----------------------------------------------------------------------
# 4. Plot: GP fit and EI at first, middle and last iteration
# ----------------------------------------------------------------------
picks = [0, N_ITER // 2, N_ITER - 1]
fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1]})
for col, k in enumerate(picks):
    xo, yo, m, s, ei, xn = snapshots[k]
    ax, ax_ei = axes[0, col], axes[1, col]
    ax.plot(grid, revenue(grid), "k--", label="true R(τ)")
    ax.plot(grid, m, "C0", label="GP mean")
    ax.fill_between(grid, m - 2 * s, m + 2 * s, color="C0", alpha=0.2, label="±2σ")
    ax.scatter(xo, yo, c="C3", zorder=5, label="evaluations")
    ax.axvline(xn, color="C2", ls=":", label="next τ")
    ax.set_title(f"After {len(xo)} evaluations")
    ax_ei.plot(grid, ei, "C2")
    ax_ei.set_xlabel("toll τ")
    if col == 0:
        ax.set_ylabel("revenue")
        ax_ei.set_ylabel("EI")
        ax.legend(fontsize=8, loc="lower center")
plt.tight_layout()
plt.savefig("toll_bo.png", dpi=130)

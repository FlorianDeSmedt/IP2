import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel, Matern
)


# ==================================================
# 1. BLACK-BOX GAME
# ==================================================

class ToyGame:
    """
    The optimizer only sees this interface:
        evaluate(x, y) -> (payoff_player_1, payoff_player_2)

    Replace this class with your real simulator or
    expensive payoff evaluation when ready.
    """

    def evaluate(self, x, y):
        # Hidden game used only for this demonstration.
        u1 = -(x - (1.0 - y))**2
        u2 = -(y - (1.0 - x))**2
        return u1, u2


# ==================================================
# 2. BAYESIAN NASH OPTIMIZER
# ==================================================

class BayesianNashOptimizer:

    def __init__(
        self,
        game,
        n_initial=8,
        max_iterations=100,
        n_candidates=201,
        beta=1.5,
        tolerance=0.02,
        seed=7,
    ):
        self.game = game
        self.max_iterations = max_iterations
        self.n_candidates = n_candidates
        self.beta = beta
        self.tolerance = tolerance
        self.rng = np.random.default_rng(seed)

        # Joint actions and observed payoffs.
        self.X = self.rng.uniform(
            0.0, 1.0, size=(n_initial, 2)
        )
        self.y1 = []
        self.y2 = []

        # Initial black-box queries.
        for x, y in self.X:
            p1, p2 = self.game.evaluate(x, y)
            self.y1.append(p1)
            self.y2.append(p2)

        self.y1 = np.asarray(self.y1)
        self.y2 = np.asarray(self.y2)

        self.actions = np.linspace(
            0.0, 1.0, n_candidates
        )

    def make_gp(self):
        kernel = ConstantKernel(1.0) * Matern(
            length_scale=[0.3, 0.3], nu=2.5
        )

        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=1,
            random_state=7,
        )

    def fit_models(self):
        self.gp1 = self.make_gp()
        self.gp2 = self.make_gp()

        self.gp1.fit(self.X, self.y1)
        self.gp2.fit(self.X, self.y2)

    def query(self, x, y):
        """Make one expensive joint-action evaluation."""
        p1, p2 = self.game.evaluate(
            float(x), float(y)
        )

        self.X = np.vstack((self.X, [x, y]))
        self.y1 = np.append(self.y1, p1)
        self.y2 = np.append(self.y2, p2)

    def best_response(self, player, opponent_action):
        """
        GP-UCB best-response proposal.

        Player 1 optimizes x with y fixed.
        Player 2 optimizes y with x fixed.
        """

        a = self.actions

        if player == 1:
            Xcand = np.column_stack((
                a, np.full_like(a, opponent_action)
            ))
            gp = self.gp1
        else:
            Xcand = np.column_stack((
                np.full_like(a, opponent_action), a
            ))
            gp = self.gp2

        mean, std = gp.predict(
            Xcand, return_std=True
        )

        ucb = mean + self.beta * std

        # Prefer an unobserved candidate if numerical
        # ties occur; repeated queries are still allowed.
        return float(a[np.argmax(ucb)])

    def regret_upper_bounds(self, x, y):
        """
        Conservative posterior-based regret bounds.

        Best-response payoffs use mean + beta * std.
        Current payoffs use mean - beta * std.

        These are model-based uncertainty bounds,
        not guaranteed bounds without additional
        assumptions on the GP and true payoff functions.
        """

        a = self.actions

        # Player 1 deviations: vary x, hold y fixed.
        X1 = np.column_stack((
            a, np.full_like(a, y)
        ))

        # Player 2 deviations: vary y, hold x fixed.
        X2 = np.column_stack((
            np.full_like(a, x), a
        ))

        mean1, std1 = self.gp1.predict(
            X1, return_std=True
        )
        mean2, std2 = self.gp2.predict(
            X2, return_std=True
        )

        current = np.array([[x, y]])

        current1, current_std1 = self.gp1.predict(
            current, return_std=True
        )
        current2, current_std2 = self.gp2.predict(
            current, return_std=True
        )

        # Maximum possible improvement under the
        # selected posterior confidence multiplier.
        r1 = (
            np.max(mean1 + self.beta * std1)
            - (current1[0] - self.beta * current_std1[0])
        )

        r2 = (
            np.max(mean2 + self.beta * std2)
            - (current2[0] - self.beta * current_std2[0])
        )

        return max(0.0, r1), max(0.0, r2)

    def optimize(self):
        # Start from the best observed joint action
        # according to a simple payoff-independent rule.
        x, y = self.X[-1]

        for iteration in range(1, self.max_iterations + 1):
            self.fit_models()

            # Alternating Bayesian best responses.
            x_new = self.best_response(1, y)
            self.query(x_new, y)
            x = x_new

            # Refresh the models after observing new data.
            self.fit_models()

            y_new = self.best_response(2, x)
            self.query(x, y_new)
            y = y_new

            # Refit before testing convergence.
            self.fit_models()

            r1, r2 = self.regret_upper_bounds(x, y)
            max_regret = max(r1, r2)

            print(
                f"Iteration {iteration:03d} | "
                f"x={x:.4f} y={y:.4f} | "
                f"regret bounds=({r1:.4f}, {r2:.4f})"
            )

            if max_regret <= self.tolerance:
                print("Posterior convergence criterion met.")
                return {
                    "x": x,
                    "y": y,
                    "regret_bound_1": r1,
                    "regret_bound_2": r2,
                    "converged": True,
                    "evaluations": len(self.X),
                }

        print("Maximum iterations reached.")
        return {
            "x": x,
            "y": y,
            "regret_bound_1": r1,
            "regret_bound_2": r2,
            "converged": False,
            "evaluations": len(self.X),
        }


# ==================================================
# 3. RUN
# ==================================================

if __name__ == "__main__":
    game = ToyGame()

    optimizer = BayesianNashOptimizer(
        game=game,
        n_initial=8,
        max_iterations=100,
        n_candidates=201,
        beta=1.5,
        tolerance=0.02,
        seed=7,
    )

    result = optimizer.optimize()

    print("\nFinal result")
    for key, value in result.items():
        print(f"{key}: {value}")
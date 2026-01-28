"""Budget-Feasible Posted-Price UCB Algorithm.

This module implements the BP-UCB algorithm for learning optimal posted prices
under budget constraints when the distribution of worker costs is unknown.
"""

from dataclasses import dataclass
import math
import random

import numpy as np


@dataclass
class BPUCBResult:
    """Results from running BP-UCB algorithm."""

    utility: int
    price_indices: list[int]
    final_budget: float
    counts: np.ndarray


class BPUCB:
    """Budget-Feasible Posted-Price UCB Algorithm.

    Learns the optimal posted price for hiring workers under a budget constraint
    using the Upper Confidence Bound (UCB) approach.

    Parameters
    ----------
    budget : float
        Total budget available for hiring workers.
    n_workers : int
        Number of workers (horizon length).
    c_min : float
        Minimum possible worker cost.
    c_max : float
        Maximum possible worker cost.
    n_prices : int
        Number of discretized price levels (arms).
    alpha : float, optional
        Multiplicative factor for geometric price spacing. If None, uses
        linear spacing.

    Attributes
    ----------
    prices : np.ndarray
        Discretized price levels.
    p_star : float
        Optimal continuous price (computed after generating bids).
    p_prime : float
        Optimal discretized price.
    F : np.ndarray
        True CDF values at each price level.
    """

    def __init__(
        self,
        budget: float,
        n_workers: int,
        c_min: float,
        c_max: float,
        n_prices: int,
        alpha: float | None = None,
    ):
        self.budget = budget
        self.n_workers = n_workers
        self.c_min = c_min
        self.c_max = c_max
        self.n_prices = n_prices
        self.alpha = alpha

        self._setup_prices()
        self.bids: np.ndarray | None = None
        self.F: np.ndarray | None = None
        self.p_star: float | None = None
        self.F_star: float | None = None

    def _setup_prices(self) -> None:
        """Initialize discretized price levels."""
        if self.alpha is not None:
            # Geometric spacing
            prices = [self.c_min]
            while (1 + self.alpha) * prices[-1] <= self.c_max:
                prices.append((1 + self.alpha) * prices[-1])
            self.prices = np.array(prices)
            self.n_prices = len(self.prices)
        else:
            # Linear spacing
            self.prices = np.linspace(self.c_min, self.c_max, self.n_prices)
            # Avoid zero price
            if self.prices[0] == 0:
                self.prices[0] = self.c_min if self.c_min > 0 else 0.01

    def generate_bids(self, seed: int | None = None) -> np.ndarray:
        """Generate random worker bids from uniform distribution.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Array of worker bids (costs).
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.bids = np.random.uniform(self.c_min, self.c_max, self.n_workers)
        np.random.shuffle(self.bids)
        self._compute_true_cdf()
        self._compute_optimal_prices()
        return self.bids

    def set_bids(self, bids: np.ndarray) -> None:
        """Set worker bids directly.

        Parameters
        ----------
        bids : np.ndarray
            Array of worker bids (costs).
        """
        self.bids = bids.copy()
        self.n_workers = len(bids)
        self._compute_true_cdf()
        self._compute_optimal_prices()

    def _compute_true_cdf(self) -> None:
        """Compute the true CDF F at each price level."""
        self.F = np.array(
            [np.mean(self.bids <= p) for p in self.prices]
        )

    def _compute_optimal_prices(self) -> None:
        """Compute optimal continuous price p* and discretized price p'."""
        # Find p* by linear interpolation at the crossing point
        n = self.n_workers
        B = self.budget
        p = self.prices
        F = self.F

        # Find index where N*F(p) crosses B/p
        i = 0
        while i < len(p) - 1 and n * F[i] < B / p[i]:
            i += 1

        if i > 0:
            i -= 1

        # Linear interpolation to find p*
        if i < len(p) - 1:
            num = (p[i] ** 2) * (B - F[i + 1] * n * p[i + 1]) - B * (p[i + 1] ** 2) + F[i] * n * (p[i + 1] ** 2) * p[i]
            dem = p[i] * (B + F[i] * n * p[i + 1] - F[i + 1] * n * p[i + 1]) - B * p[i + 1]
            if dem != 0:
                self.p_star = num / dem
            else:
                self.p_star = p[i]
        else:
            self.p_star = p[i]

        # Compute F* at p*
        self.F_star = np.mean(self.bids <= self.p_star)

        # Find optimal discretized price p'
        utilities = np.minimum(n * F, B / p)
        self.i_prime = int(np.argmax(utilities))
        self.p_prime = p[self.i_prime]

    def run(self, shuffle: bool = False) -> BPUCBResult:
        """Run the BP-UCB algorithm.

        Parameters
        ----------
        shuffle : bool, optional
            Whether to shuffle bids before running.

        Returns
        -------
        BPUCBResult
            Results containing utility, price history, and statistics.

        Notes
        -----
        Uses arm correlation optimization from the paper: rejection at price p
        implies rejection for all cheaper prices, so we maintain a lower bound
        estimate and deactivate arms below it.
        """
        if self.bids is None:
            raise ValueError("Bids not set. Call generate_bids() or set_bids() first.")

        bids = self.bids.copy()
        if shuffle:
            np.random.shuffle(bids)

        K = self.n_prices
        n = self.n_workers
        B = self.budget
        p = self.prices

        t = 0
        budget_remaining = B
        F_estimate = np.zeros(K)
        counts = np.zeros(K)
        utility = 0
        price_indices = []

        # Track active arms - deactivate arms below estimated cost support
        # Keep one arm below the lower bound for exploration
        active = np.ones(K, dtype=bool)
        min_accepted_idx = K  # Index of lowest price that got accepted

        while budget_remaining > self.c_min and t < n:
            # Select arm
            unvisited = (counts == 0) & active
            if np.any(unvisited):
                i = int(np.argmax(unvisited))  # First unvisited active arm
            else:
                # UCB selection among active arms
                # V = min(UCB * N, B / p) - matches original notebook implementation
                ucb = np.zeros(K)
                for j in range(K):
                    if counts[j] > 0:
                        ucb[j] = F_estimate[j] + np.sqrt(2 * math.log(t) / counts[j])
                    else:
                        ucb[j] = 1.0  # Optimistic for unvisited

                V = np.minimum(ucb * n, B / p)

                # Mask out prices we can't afford or inactive
                V[p > budget_remaining] = -np.inf
                V[~active] = -np.inf
                i = int(np.argmax(V))

            price_indices.append(i)

            # Observe acceptance
            y = int(bids[t] <= p[i])
            counts[i] += 1
            F_estimate[i] += (y - F_estimate[i]) / counts[i]
            budget_remaining -= p[i] * y
            utility += y

            # Arm correlation: if rejected, all cheaper arms would also reject
            # Update lower bound estimate and deactivate arms
            if y == 1 and i < min_accepted_idx:
                min_accepted_idx = i
            elif y == 0 and i < min_accepted_idx:
                # Deactivate arms below this one, keeping one for exploration
                for j in range(i):
                    if counts[j] > 0:  # Only deactivate if already explored
                        active[j] = False

            t += 1

        return BPUCBResult(
            utility=utility,
            price_indices=price_indices,
            final_budget=budget_remaining,
            counts=counts,
        )

    def compute_regret(self, result: BPUCBResult) -> float:
        """Compute regret compared to optimal price.

        Parameters
        ----------
        result : BPUCBResult
            Results from running the algorithm.

        Returns
        -------
        float
            Regret (optimal utility - achieved utility).
        """
        if self.p_star is None:
            raise ValueError("Optimal price not computed. Generate bids first.")

        optimal_utility = self.budget / self.p_star
        return optimal_utility - result.utility

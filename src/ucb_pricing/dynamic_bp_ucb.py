"""Dynamic Budget-Feasible Posted-Price UCB Algorithm.

This module implements a time-varying extension of BP-UCB that handles
demand fluctuations across different time periods (e.g., 2-hour slices).
"""

import math

import numpy as np


class DynamicBPUCB:
    """Dynamic Budget-Feasible Posted-Price UCB Algorithm.

    Extends BP-UCB to handle time-varying demand by dividing the day into
    time slices with separate budget allocations.

    Parameters
    ----------
    prices : np.ndarray
        Discretized price levels.
    n_slices : int
        Number of time slices (default 12 for 2-hour slices in a day).
    """

    def __init__(
        self,
        prices: np.ndarray,
        n_slices: int = 12,
    ):
        self.prices = prices
        self.n_prices = len(prices)
        self.n_slices = n_slices

    @staticmethod
    def get_time_slice(minutes_from_midnight: int, slice_duration: int = 120) -> int:
        """Convert time to time slice index.

        Parameters
        ----------
        minutes_from_midnight : int
            Time in minutes from midnight.
        slice_duration : int
            Duration of each slice in minutes.

        Returns
        -------
        int
            Time slice index.
        """
        return minutes_from_midnight // slice_duration

    @staticmethod
    def estimate_total_requests(requests_per_slice: np.ndarray) -> float:
        """Estimate total requests per slice.

        Parameters
        ----------
        requests_per_slice : np.ndarray
            Number of requests in each time slice.

        Returns
        -------
        float
            Estimated average requests per slice.
        """
        return np.mean(requests_per_slice)

    def run(
        self,
        start_time: int,
        allocated_budgets: np.ndarray,
        requests_per_slice: np.ndarray,
        cost_distribution: callable | None = None,
    ) -> dict:
        """Run the Dynamic BP-UCB algorithm.

        Parameters
        ----------
        start_time : int
            Starting time in minutes from midnight.
        allocated_budgets : np.ndarray
            Budget allocated to each time slice.
        requests_per_slice : np.ndarray
            Number of requests in each time slice.
        cost_distribution : callable, optional
            Function that returns a random cost. Defaults to uniform[0, max_price].

        Returns
        -------
        dict
            Results containing price offers and statistics.
        """
        if cost_distribution is None:
            max_price = self.prices[-1]
            min_price = self.prices[0]

            def cost_distribution():
                return np.random.uniform(min_price, max_price)

        K = self.n_prices
        p = self.prices

        n = 2  # iteration
        h = self.get_time_slice(start_time)

        B = allocated_budgets[h]
        B_n = B
        N = 2 * self.estimate_total_requests(requests_per_slice)

        arm_counts = np.ones(K)
        F_estimate = np.zeros(K)

        price_offers = []
        utilities = []

        for h_index in range(self.n_slices):
            current_h = (h + h_index) % self.n_slices
            B_n += allocated_budgets[current_h]
            B = B_n
            N = 2 * self.estimate_total_requests(requests_per_slice)

            slice_utility = 0
            n_requests = int(requests_per_slice[current_h])

            for _ in range(n_requests):
                # Compute UCB estimates
                F_tilde = np.minimum(
                    1.0,
                    F_estimate + np.sqrt(2 * np.log(n) / arm_counts)
                )

                # Select optimal arm
                opt_i = self._select_arm(N, F_tilde, B, p, B_n)
                price = p[opt_i]
                price_offers.append(price)

                # Sample cost and observe acceptance
                cost = cost_distribution()
                accepted = cost <= price

                if accepted:
                    B_n -= price
                    slice_utility += 1

                # Update estimates
                F_estimate[opt_i] += (accepted - F_estimate[opt_i]) / (arm_counts[opt_i] + 1)
                arm_counts[opt_i] += 1
                n += 1

            utilities.append(slice_utility)

        return {
            "price_offers": np.array(price_offers),
            "utilities_per_slice": np.array(utilities),
            "total_utility": sum(utilities),
            "arm_counts": arm_counts,
        }

    def _select_arm(
        self,
        N: float,
        F_tilde: np.ndarray,
        B: float,
        p: np.ndarray,
        B_n: float,
    ) -> int:
        """Select the optimal arm given current estimates.

        Parameters
        ----------
        N : float
            Estimated number of workers.
        F_tilde : np.ndarray
            UCB estimates of CDF.
        B : float
            Current budget.
        p : np.ndarray
            Price levels.
        B_n : float
            Remaining budget.

        Returns
        -------
        int
            Index of selected arm.
        """
        opt_i = 0
        opt_value = -np.inf

        for i in range(len(p)):
            if p[i] <= B_n:
                value = min(N * F_tilde[i], B / p[i])
                if value > opt_value:
                    opt_value = value
                    opt_i = i

        return opt_i

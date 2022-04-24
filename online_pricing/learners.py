import numpy as np
import numpy.typing as npt
from typing import List


class Learner:
    """Generic learner class, for now it has no methods we may update it when the time comes.
    :param n_products: number of products in environment.
    :param n_prices: number of prices per products.
    """

    def __init__(self, n_products: int, n_prices: int) -> None:
        self.t: int = 0
        self.n_products: int = n_prices
        self.n_prices: int = n_products


class GreedyLearner(Learner):
    """Greedy learner class, implemented for solving "Step 2" of the project description.
        :param n_products: number of products in environment.
        :param n_prices: number of prices per products.
        :param initial_reward: initial reward with the configuration with minimum prices.
    """
    def __init__(self, n_products: int, n_prices: int, initial_reward: int) -> None:
        super().__init__(n_products, n_prices)
        self.done = False
        self.configs: npt.NDArray[np.int8] = np.diag(np.ones(self.n_products, dtype=np.int8))
        self.best_config: npt.NDArray[np.int8] = np.zeros(n_products)
        self.best_reward: int = initial_reward

    def pull_arm(self) -> npt.NDArray:
        """
        Return a numpy.ndarray with the `n_product`-configurations. If the algorithm has finished
        it returns the best configuration found.

        :return: array with configuration or best configuration found.
        """
        if self.done:
            return self.best_config
        return np.array([self.best_config + self.configs[i] for i in range(self.n_products)])

    def update(self, reward: List[int]) -> None:
        """
        Update the learner with a list of reward given by the environment as a response
        to the configuration of prices.
        :param reward: list reward retrieved from the environment (ordered).
        """
        self.t += 1

        if self.done:
            return

        idx = np.argmax(reward)
        max_reward = reward[idx]

        if max_reward > self.best_reward:
            self.best_config = self.best_config + self.configs[idx]
            self.best_reward = max_reward
        else:
            self.done = True

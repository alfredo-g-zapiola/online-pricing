import numpy as np


class Learner:
    """Base learner class for the project."""

    def __init__(self, n_arms: int, prices: list[float]):
        self.t = 0
        self.prices = sorted(prices)
        self.n_arms = n_arms
        self.rewards: list[int] = []
        self.rewards_per_arm: list[list[int]] = [[] for _ in range(n_arms)]
        self.pulled_arms: list[int] = []

    def reset(self) -> None:
        pass

    def act(self) -> int:
        pass

    def update(self, arm_pulled: int, reward: int) -> None:
        self.t += 1
        self.rewards.append(reward)
        self.rewards_per_arm[arm_pulled].append(reward)
        self.pulled_arms.append(arm_pulled)

    def get_arm(self, price: float) -> int:
        return self.prices.index(price)

    def sample_arm(self, arm_id):
        return np.random.uniform(0, 1)


class GreedyLearner(Learner):
    """Greedy learner.

    Works by pulling the same arm every time.
    This arm can be changed by the update method, which sets the next price in the list.
    """

    def __init__(self, n_arms: int, prices: list[int]):
        super().__init__(n_arms, prices)
        self.current_price = 0

    def act(self) -> float:
        return self.prices[self.current_price]

    def greedy_act(self) -> int | None:
        return (
            self.prices[self.current_price + 1] if self.current_price < (self.n_arms - 1) else None
        )

    def update(self, arm_pulled: int, reward: int) -> None:  # i don't think this method is useful
        super().update(arm_pulled, reward)
        self.current_price += 1 if self.current_price < (self.n_arms - 1) else 0


class Ucb(Learner):
    def __init__(self, n_arms: int, prices: list[int]):
        super().__init__(n_arms, prices)
        self.means = np.zeros(n_arms)
        self.widths = np.array([np.inf for _ in range(n_arms)])

    def act(self) -> int:
        idx = np.argmax((self.means + self.widths))
        return int(idx)

    def update(self, arm_pulled: int, reward: int) -> None:
        super().update(arm_pulled, reward)
        self.means[arm_pulled] = np.mean(self.rewards_per_arm[arm_pulled])
        for idx in range(self.n_arms):
            n = len(self.rewards_per_arm[idx])
            if n > 0:
                self.widths[idx] = np.sqrt(2 * np.max(self.prices) * np.log(self.t) / n)
            else:
                self.widths[idx] = np.inf


class TSLearner(Learner):
    def __init__(self, n_arms: int, prices: list[float]):
        super().__init__(n_arms, prices)
        self.beta_parameters = np.ones((n_arms, 2))

    def pull_arm(self) -> int:
        idx = np.argmax(
            np.array(self.prices)
            * np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1])
        )
        return int(idx)

    def update(self, arm_pulled: int, reward: int) -> None:
        super().update(arm_pulled, reward)
        self.beta_parameters[arm_pulled, 0] = self.beta_parameters[arm_pulled, 0] + reward
        self.beta_parameters[arm_pulled, 1] = self.beta_parameters[arm_pulled, 1] + 1.0 - reward

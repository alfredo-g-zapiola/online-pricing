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

    def sample_arm(self, arm_id) -> float:
        pass

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

    def sample_arm(self, arm_id):
        value = self.means[arm_id] + self.widths[arm_id]
        return value if value <= 1 else 1


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

    def sample_arm(self, arm_id) -> float:
        return np.random.beta(self.beta_parameters[arm_id, 0], self.beta_parameters[arm_id, 1])

    def mean_arm(self, arm_id) -> float:
        """
        Returns the mean of the beta distribution for such arm id.
        :param arm_id:
        :return:
        """
        return self.beta_parameters[arm_id, 0] / (self.beta_parameters[arm_id, 0] + self.beta_parameters[arm_id, 1] )
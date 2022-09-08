from abc import ABC, abstractmethod

import numpy as np


class Learner(ABC):
    """Base learner class for the project."""

    def __init__(self, n_arms: int, prices: list[float]):
        self.t = 0
        self.prices = sorted(prices)
        self.n_arms = n_arms
        self.rewards: list[int] = []
        self.rewards_per_arm: list[list[int]] = [[] for _ in range(n_arms)]
        self.pulled_arms: list[int] = []

    @property
    @abstractmethod
    def parameters(self) -> list[tuple[float, ...]]:
        pass

    @abstractmethod
    def update(self, arm_pulled: int, reward: int) -> None:
        self.t += 1
        self.rewards.append(reward)
        self.rewards_per_arm[arm_pulled].append(reward)
        self.pulled_arms.append(arm_pulled)

    def get_arm(self, price: float) -> int:
        return self.prices.index(price)

    @abstractmethod
    def sample_arm(self, arm_id: int) -> float:
        pass


class UCBLearner(Learner):
    def __init__(self, n_arms: int, prices: list[float]):
        super().__init__(n_arms, prices)
        self.means = [0.0] * n_arms
        self.widths = [np.inf] * n_arms

    def parameters(self) -> list[tuple[float, ...]]:
        return [(self.means[idx], self.widths[idx]) for idx in range(self.n_arms)]

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

    def sample_arm(self, arm_id: int) -> float:
        value = self.means[arm_id] + self.widths[arm_id]
        return value if value <= 1 else 1.0


class TSLearner(Learner):
    def __init__(self, n_arms: int, prices: list[float]):
        super().__init__(n_arms, prices)
        self.beta_parameters = np.ones(shape=(n_arms, 2))

    @property
    def parameters(self) -> list[tuple[float, ...]]:
        return [tuple(self.beta_parameters[i]) for i in range(self.n_arms)]

    def pull_arm(self) -> int:
        idx = np.argmax(np.array(self.prices) * np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]))
        return int(idx)

    def update(self, arm_pulled: int, reward: int) -> None:
        super().update(arm_pulled, reward)
        self.beta_parameters[arm_pulled, 0] = self.beta_parameters[arm_pulled, 0] + reward
        self.beta_parameters[arm_pulled, 1] = self.beta_parameters[arm_pulled, 1] + 1.0 - reward

    def sample_arm(self, arm_id: int) -> float:
        return float(np.random.beta(self.beta_parameters[arm_id, 0], self.beta_parameters[arm_id, 1]))

    def mean_arm(self, arm_id: int) -> float:
        """
        Returns the mean of the beta distribution for such arm id.
        :param arm_id: the arm id
        :return: the mean of the beta distribution
        """
        return float(self.beta_parameters[arm_id, 0] / (self.beta_parameters[arm_id, 0] + self.beta_parameters[arm_id, 1]))


class SWUCBLearner(UCBLearner):
    def __init__(self, n_arms: int, prices: list[float], window_size: int):
        super().__init__(n_arms, prices)
        self.window_size = window_size

    def update(self, arm_pulled: int, reward: int) -> None:
        self.t += 1
        self.rewards.append(reward)
        self.rewards_per_arm[arm_pulled].append(reward)
        self.pulled_arms.append(arm_pulled)

        self.means[arm_pulled] = np.mean(self.rewards_per_arm[arm_pulled][-self.window_size :])
        for idx in range(self.n_arms):
            n = len(self.rewards_per_arm[idx][-self.window_size :])
            if n > 0:
                self.widths[idx] = np.sqrt(2 * np.max(self.prices) * np.log(self.t) / n)
            else:
                self.widths[idx] = np.inf


# https://arxiv.org/pdf/1802.03692.pdf
class MUCB(UCBLearner):
    def __init__(self, n_arms: int, prices: list[float], w: int, beta: int, gamma: int) -> None:
        super().__init__(n_arms, prices)
        assert w > 0 & beta > 0 & gamma >= 0 & gamma <= 1
        self.w = w
        self.beta = beta
        self.gamma = gamma
        self.detections = [0]
        self.last_detection = 0

    def update(self, arm_pulled: int, reward: int) -> None:
        super(UCBLearner, self).update(arm_pulled, reward)

        self.means[arm_pulled] = np.mean(self.rewards_per_arm[arm_pulled])
        for idx in range(self.n_arms):
            n = len(self.rewards_per_arm[idx])
            if n > 0:
                self.widths[idx] = np.sqrt(2 * np.max(self.prices) * np.log(self.t - self.last_detection) / n)

            else:
                self.widths[idx] = np.inf

        if len(self.rewards_per_arm[arm_pulled]) > self.w:
            if self.change_detection(self.rewards_per_arm[arm_pulled]):
                self.last_detection = self.t
                self.detections.append(self.t)
                for idx in range(self.n_arms):
                    self.rewards_per_arm[idx] = []

    def change_detection(self, observations: list[int]) -> bool:
        if sum(observations[np.floor(-self.w / 2) + 1 :]) - sum(observations[: np.floor(self.w / 2)]) > self.beta:
            return True

        return False

    def act(self) -> int:
        idx = np.argmax((self.means + self.widths))
        return int(idx)

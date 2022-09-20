from abc import ABC, abstractmethod
from typing import Any, Type, cast

import numpy as np

from online_pricing.models.rewards_and_features import RewardsAndFeatures


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
    def parameters(self) -> list[tuple[float, ...]]:
        raise TypeError("Parameters not implemented for this learner.")

    @abstractmethod
    def update(self, arm_pulled: int, reward: int, *args: Any) -> None:
        self.t += 1
        self.rewards.append(reward)
        self.rewards_per_arm[arm_pulled].append(reward)
        self.pulled_arms.append(arm_pulled)

    def get_arm(self, price: float) -> int:
        return self.prices.index(price)

    @abstractmethod
    def sample_arm(self, arm_id: int, *args: Any) -> float:
        pass

    def mean_arm(self, arm_id: int) -> float:
        pass

    def new_day(self) -> None:
        self.t += 1


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

    def update(self, arm_pulled: int, reward: int, *args: Any) -> None:
        super().update(arm_pulled, reward)
        self.means[arm_pulled] = np.mean(self.rewards_per_arm[arm_pulled]).astype(float)
        for idx in range(self.n_arms):
            n = len(self.rewards_per_arm[idx])
            if n > 0:
                self.widths[idx] = np.sqrt(2 * np.max(self.prices) * np.log(self.t) / n)
            else:
                self.widths[idx] = np.inf

    def sample_arm(self, arm_id: int, *args: Any) -> float:
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

    def update(self, arm_pulled: int, reward: int, *args: Any) -> None:
        super().update(arm_pulled, reward)
        self.beta_parameters[arm_pulled, 0] = self.beta_parameters[arm_pulled, 0] + reward
        self.beta_parameters[arm_pulled, 1] = self.beta_parameters[arm_pulled, 1] + 1.0 - reward

    def sample_arm(self, arm_id: int, *args: Any) -> float:
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

    def update(self, arm_pulled: int, reward: int, *args: Any) -> None:
        self.t += 1
        self.rewards.append(reward)
        self.rewards_per_arm[arm_pulled].append(reward)
        self.pulled_arms.append(arm_pulled)

        self.means[arm_pulled] = np.mean(self.rewards_per_arm[arm_pulled][-self.window_size :]).astype(float)
        for idx in range(self.n_arms):
            n = len(self.rewards_per_arm[idx][-self.window_size :])
            if n > 0:
                self.widths[idx] = np.sqrt(2 * np.max(self.prices) * np.log(self.t) / n)
            else:
                self.widths[idx] = np.inf


# https://arxiv.org/pdf/1802.03692.pdf
class MUCBLearner(UCBLearner):
    def __init__(self, n_arms: int, prices: list[float], w: float, beta: float, gamma: float) -> None:
        super().__init__(n_arms, prices)

        self.w = w
        self.beta = beta
        self.gamma = gamma
        self.detections = [0]
        self.last_detection = 0

        assert self.w > 0 and self.beta > 0 and 0 <= self.gamma <= 1

    def update(self, arm_pulled: int, reward: int, *args: Any) -> None:
        super(UCBLearner, self).update(arm_pulled, reward)

        self.means[arm_pulled] = np.mean(self.rewards_per_arm[arm_pulled]).astype(float)
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
        if sum(observations[int(np.floor(-self.w / 2)) + 1 :]) - sum(observations[: int(np.floor(self.w / 2))]) > self.beta:
            return True

        return False

    def act(self) -> int:
        idx = np.argmax((self.means + self.widths))
        return int(idx)


class CGLearner(Learner):
    """
    Class that manages the context generation algorithm.

    It's characterized by instance learners that gets updated whenever there's a context split.
    In particular, having binary features this class will have 2^d learners, where d is the number of features.
    Each learner correspond to a set of features, and it's considered active if that particular features has been
    split. A feature is split after a condition is met, until then, all the non-split features are considered
    aggregated.
    """

    def __init__(self, n_arms: int, prices: list[float], context_window: int, n_features: int) -> None:
        """
        :param n_arms: the number of arms
        :param prices: the prices of the arms
        :param context_window: the size of the context window
        :param n_features: the number of features
        """
        super().__init__(n_arms, prices)
        self.n_features = n_features
        self.context_window = context_window
        self.features_count = np.zeros(shape=(2, self.n_features), dtype=np.int64)

        self.learners = self.initialize_learners()

        self.is_split_feature = np.zeros(n_features, dtype=np.int8)
        self.training_data: list[list[RewardsAndFeatures]] = [[] for _ in range(n_arms)]
        self.pre_split_data: list[list[RewardsAndFeatures]] = [[] for _ in range(n_arms)]

    def initialize_learners(self) -> Any:
        """Initialize learners."""
        learners = np.full(shape=[2] * self.n_features, fill_value=None, dtype=UCBLearner)
        initialize_learners = learners.flatten()
        for idx in range(len(initialize_learners)):
            initialize_learners[idx] = UCBLearner(self.n_arms, self.prices)
        return initialize_learners.reshape(learners.shape)

    def update(self, arm_pulled: int, reward: int, *args: Any) -> None:
        """
        Update the learners.

        :param arm_pulled: the arm pulled
        :param reward: the reward
        :param args: features must be added here
        """
        features: list[int] = args[0]
        if features is None:
            raise ValueError(f"Features cannot be None: {features}")
        self.learners[tuple(np.logical_and(features, self.is_split_feature).astype(np.int8))].update(
            arm_pulled, reward, features
        )

        for idx, feature in enumerate(features):
            self.features_count[idx, feature] += 1

        self.training_data[arm_pulled].append(RewardsAndFeatures(reward=reward, features=features))
        self.pre_split_data[arm_pulled].append(RewardsAndFeatures(reward=reward, features=features))

    def new_day(self) -> None:
        """New day, see if context split is needed."""
        features_probability = self.features_count / np.sum(self.features_count)
        context_rewards = self.get_context_rewards()

        self.t += 1
        if self.t % self.context_window == 0:
            self.analyze_context(features_probability, context_rewards)

            self.features_count = np.zeros(shape=(2, self.n_features), dtype=np.int64)
            self.pre_split_data = [[] for _ in range(self.n_arms)]

    def get_context_rewards(self) -> list[float]:
        """Get the context rewards."""
        rewards = np.zeros(shape=(2, self.n_features))
        for arm in range(self.n_arms):
            for reward_and_features in self.pre_split_data[arm]:
                for idx, feature in enumerate(reward_and_features.features):
                    rewards[idx, feature] += reward_and_features.reward

        return cast(list[float], rewards)

    def analyze_context(self, features_probability: Any, context_rewards: Any) -> None:
        """Make decision to split a feature or not"""

        for idx, feature_to_split in enumerate(self.is_split_feature):
            if self.do_we_split(features_probability, context_rewards, feature_to_split) and not feature_to_split:
                self.is_split_feature[idx] = 1
                self.train_learners()

    def do_we_split(self, features_probability: Any, context_rewards: Any, feature_to_split: int) -> bool:
        """Check if we should split a feature."""
        probability_0 = features_probability[feature_to_split, 0]
        probability_1 = features_probability[feature_to_split, 1]
        context_rewards_0 = context_rewards[feature_to_split, 0]
        context_rewards_1 = context_rewards[feature_to_split, 1]

        context_reward_aggregated = self.get_aggregated_probability(context_rewards)

        return bool(probability_0 * context_rewards_0 + probability_1 * context_rewards_1 > context_reward_aggregated)

    def train_learners(self) -> None:
        """Train new learners on the training data stored."""
        self.learners = self.initialize_learners()
        for arm, episode in enumerate(self.training_data):
            for reward_and_features in episode:
                reward, features = reward_and_features.reward, reward_and_features.features
                self.learners[tuple(np.logical_and(features, self.is_split_feature).astype(np.int8))].update(arm, reward)

    def sample_arm(self, arm_id: int, *args: Any) -> float:
        """
        Sample an arm.

        The choice of which learner to pull is based on split features. If a feature is not split, it's considered
        aggregated, so the learner chosen the aggregated learner of that feature.
        """
        features: list[int] = args[0]
        arm_sampled = self.learners[tuple(np.logical_and(features, self.is_split_feature).astype(np.int8))].sample_arm(
            arm_id
        )
        return cast(float, arm_sampled)

    def get_arm(self, price: float) -> int:
        return self.prices.index(price)

    def get_aggregated_probability(self, context_rewards: Any) -> float:
        """Get the aggregated probability of a feature."""
        reward: float = context_rewards[tuple(self.is_split_feature)]
        return reward


LEARNERS: dict[str, Type[Learner]] = {
    "UCB": UCBLearner,
    "TS": TSLearner,
    "SWUCB": SWUCBLearner,
    "MUCB": MUCBLearner,
    "CGUCB": CGLearner,
}


class LearnerFactory:
    def __init__(
        self,
        learner_class: str,
        window_size: int | None = None,
        w: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
        context_window: int | None = None,
        n_features: int | None = None,
    ) -> None:
        self._args: tuple[Any, ...] | None = None
        self.learner = LEARNERS[learner_class]
        self._specific_args: tuple[Any, ...]

        match learner_class:
            case "TS":
                self._specific_args = tuple()
            case "UCB":
                self._specific_args = tuple()
            case "CGUCB":
                self._specific_args = (
                    context_window,
                    n_features,
                )
            case "SWUCB":
                if window_size is None:
                    raise ValueError("window_size must be provided for SWUCB")
                self._specific_args = (window_size,)
            case "MUCB":
                if w is None or beta is None or gamma is None:
                    raise ValueError("w, beta and gamma must be provided for MUCB")
                self._specific_args = (w, beta, gamma)
            case _:
                raise ValueError("Unknown learner class")

    @property
    def args(self) -> tuple[Any, ...]:
        return self.base_args + self._specific_args

    @property
    def base_args(self) -> tuple[Any, ...]:
        if self._args is None:
            raise ValueError("args must be set before calling get_learner")
        return self._args

    @base_args.setter
    def base_args(self, value: tuple[Any, ...]) -> None:
        self._args = value

    def get_learner(self) -> Learner:
        return self.learner(*self.args)

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.gridspec import GridSpec
from scipy import interpolate


def moving_average(avg_rewards) -> list[float]:
    window = 7
    average = []
    for idx in range(len(avg_rewards) - window + 1):
        average.append(np.mean(avg_rewards[idx : idx + window]))
    for idx in range(window - 1):
        average.insert(idx, np.mean(avg_rewards[0 : 0 + idx]))

    return average


class Tracer:
    def __init__(self, n_sims: int, n_days: int) -> None:
        self.optimum_total: float

        self.avg_reward = list[float]()
        self.arm_data = list[list[list[float]]]()
        self.rewards_mat = np.zeros(shape=(n_sims, n_days))
        self.regrets_mat = np.zeros(shape=(n_sims, n_days))

    def add_avg_reward(self, avg_reward: float) -> None:
        self.avg_reward.append(avg_reward)

    def set_optimum_total(self, optimum_total: float) -> None:
        self.optimum_total = optimum_total

    def add_arm_data(self, arm_data: list[list[float]]) -> None:
        self.arm_data.append(arm_data)

    def add_daily_data(self, sample: int, rewards: list[float]) -> None:
        self.rewards_mat[sample, :] = rewards

    def plot_day(self) -> None:
        ma = moving_average(self.avg_reward)
        plt.figure()
        plt.plot(
            self.avg_reward,
            label="Mean Average Reward",
            color="blue",
            linewidth=0.75,
        )
        plt.plot(
            ma,
            label="Moving Average",
            color="red",
        )
        plt.xlabel("Days")
        plt.ylabel("Average Reward")
        plt.legend()
        plt.show()

        fig = plt.figure()
        spec = GridSpec(ncols=3, nrows=2)

        axes = list()
        axes.append(fig.add_subplot(spec[0, 0]))
        axes.append(fig.add_subplot(spec[0, 1]))
        axes.append(fig.add_subplot(spec[0, 2]))
        axes.append(fig.add_subplot(spec[1, 0]))
        axes.append(fig.add_subplot(spec[1, 2]))

        for idx, ax in enumerate(axes):
            for day in range(10, len(self.arm_data), 5):  # starting from day 10
                ax.scatter(
                    y=self.arm_data[day][idx],
                    x=range(len(self.arm_data[day][idx])),
                    marker=".",
                    linewidth=0.5,
                )

            n = len(self.arm_data[-1][idx])
            x_new = np.linspace(0, n - 1, 50)
            bspline = interpolate.make_interp_spline(range(n), self.arm_data[-1][idx])
            y_new = bspline(x_new)
            ax.plot(x_new, y_new)

            ax.set_title(f"Product {idx }")
            ax.set_xlabel("Days")
            ax.set_ylabel("Conversion Rate")
            ax.legend()

        for ax in fig.get_axes():
            ax.tick_params(bottom=False, labelbottom=False)

        plt.show()

    def plot_total(self) -> None:

        mean_rewards = self.rewards_mat.mean(axis=0)
        sdev_rewards = self.rewards_mat.std(axis=0)
        self.plot(mean_rewards, sdev_rewards, "Mean rewards")

        cum_regret = np.cumsum(self.regrets_mat, axis=1)
        mean_regrets = cum_regret.mean(axis=0)
        sdev_regret = cum_regret.std(axis=0)
        self.plot(mean_regrets, sdev_regret, "Mean cumulative regret")

    def plot(self, what: npt.NDArray[float], sdev: float, title: str) -> None:
        ma = moving_average(what)

        plt.figure()
        plt.plot(
            what,
            label="Mean {}".format(title),
            color="blue",
            linewidth=0.75,
        )
        plt.plot(what + 3 * sdev, label="Upper bound {}".format(title), color="pink", linewidth=0.75)
        plt.plot(what - 3 * sdev, label="Lower bound {}".format(title), color="pink", linewidth=0.75)
        plt.plot(
            ma,
            label="Moving Average",
            color="red",
        )
        plt.xlabel("Days")
        plt.ylabel("{}".format(title))
        plt.legend()
        plt.show()

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.gridspec import GridSpec
from scipy import interpolate

plt.rcParams["figure.figsize"] = (15, 10)


def moving_average(avg_rewards: list[float] | npt.NDArray[np.float32]) -> list[float]:
    window = 4
    average = []
    for idx in range(len(avg_rewards) - window + 1):
        average.append(np.mean(avg_rewards[idx : idx + window]))
    for idx in range(window - 1):
        average.insert(idx, np.mean(avg_rewards[0 : 0 + idx]))

    return cast(list[float], average)


class Tracer:
    def __init__(self, n_sims: int, n_days: int) -> None:
        self.regret = list[float]()
        self.avg_reward = list[float]()
        self.arm_data = list[list[list[float]]]()
        self.rewards_mat = np.zeros(shape=(n_sims, n_days))
        self.regrets_mat = np.zeros(shape=(n_sims, n_days))
        self.optimum_total = list[float]()
        self.splits = list[int | None]()

    def add_avg_reward(self, avg_reward: float) -> None:
        self.avg_reward.append(avg_reward)

    def add_regret(self, regret: float) -> None:
        self.regret.append(regret)

    def add_optimum_total(self, optimum_total: float) -> None:
        self.optimum_total.append(optimum_total)

    def add_arm_data(self, arm_data: list[list[float]]) -> None:
        self.arm_data.append(arm_data)

    def add_split(self, index: int | None) -> None:
        self.splits.append(index)

    def add_daily_data(self, sample: int, rewards: list[float], regrets: list[float] | None = None) -> None:
        self.rewards_mat[sample, :] = rewards
        self.regrets_mat[sample, :] = regrets

    def new_day(self) -> None:
        self.avg_reward = list[float]()
        self.arm_data = list[list[list[float]]]()
        self.regret = list[float]()
        self.optimum_total = list[float]()

    def plot_day(self) -> None:
        ma = moving_average(self.avg_reward)
        plt.figure()
        if any(self.splits):
            for split in set(self.splits):
                plt.axvline(x=split, color="black", linestyle="--", label="Split")
        plt.plot(
            self.avg_reward,
            label="Mean Average Reward",
            color="blue",
            linewidth=0.66,
        )

        plt.plot(
            self.optimum_total,
            color="black",
            label="Optimum",
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

        for ax in fig.get_axes():
            ax.tick_params(bottom=False, labelbottom=False)

        plt.show()

    def plot_total(self) -> None:

        mean_rewards = self.rewards_mat.mean(axis=0)
        sdev_rewards = self.rewards_mat.std(axis=0)

        self.plot(mean_rewards, sdev_rewards, "Mean rewards", optimum=True)

        # Plot regret
        cum_regret = np.cumsum(self.regrets_mat, axis=1)
        mean_regrets = cum_regret.mean(axis=0)
        sdev_regret = cum_regret.std(axis=0)
        self.plot(mean_regrets, sdev_regret, "Mean cumulative regret")

    def plot(self, data: npt.NDArray[np.float32] | list[float], sdev: float, title: str, optimum: bool = False) -> None:
        ma = moving_average(data)

        plt.figure()
        plt.plot(
            data,
            label="Mean {}".format(title),
            color="blue",
            linewidth=0.75,
        )
        plt.plot(data + np.multiply(3, sdev), label="Upper bound {}".format(title), color="pink", linewidth=0.75)
        plt.plot(data - np.multiply(3, sdev), label="Lower bound {}".format(title), color="pink", linewidth=0.75)
        plt.plot(
            ma,
            label="Moving Average",
            color="red",
        )
        if optimum:
            plt.plot(
                self.optimum_total,
                color="black",
                label="Optimum",
            )

        plt.xlabel("Days")
        plt.ylabel("{}".format(title))
        plt.legend()
        plt.show()

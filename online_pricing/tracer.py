from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy import interpolate


class Tracer:
    def __init__(self) -> None:
        self.avg_reward = list[float]()
        self.arm_data = list[list[list[float]]]()

    def add_measurement(self, avg_reward: float, arm_data: list[list[float]]) -> None:
        self.avg_reward.append(avg_reward)
        self.arm_data.append(arm_data)

    def plot(self) -> None:
        moving_average = self.moving_average()
        plt.figure()
        plt.plot(
            self.avg_reward,
            label="Mean Average Reward",
            color="blue",
            linewidth=0.75,
        )
        plt.plot(
            moving_average,
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

    def moving_average(self) -> list[float]:
        window = 7
        average = []
        for idx in range(len(self.avg_reward) - window + 1):
            average.append(np.mean(self.avg_reward[idx : idx + window]))
        for idx in range(window - 1):
            average.insert(idx, np.mean(self.avg_reward[0 : 0 + idx]))

        return average

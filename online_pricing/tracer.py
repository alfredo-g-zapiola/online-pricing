from typing import Any

import matplotlib.pyplot as plt
import numpy as np


class Tracer:
    def __init__(self) -> None:
        self.measurements = list[Any]()

    def add_measurement(self, measurement: Any) -> None:
        self.measurements.append(measurement)

    def plot(self) -> None:
        moving_average = self.moving_average()
        plt.figure()
        plt.plot(
            self.measurements,
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

    def moving_average(self) -> list[float]:
        window = 5
        average = []
        for idx in range(len(self.measurements) - window + 1):
            average.append(np.mean(self.measurements[idx : idx + window]))
        for idx in range(window - 1):
            average.insert(idx, np.mean(self.measurements[0 : 0 + idx]))

        return average

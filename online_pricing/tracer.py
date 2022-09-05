from typing import Any

import matplotlib.pyplot as plt


class Tracer:
    def __init__(self) -> None:
        self.measurements = list[Any]()

    def add_measurement(self, measurement: Any) -> None:
        self.measurements.append(measurement)

    def plot(self) -> None:
        plt.plot(self.measurements)
        plt.show()

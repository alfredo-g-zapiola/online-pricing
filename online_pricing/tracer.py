import matplotlib.pyplot as plt


class Tracer:
    def __init__(self):
        self.measurements = list()

    def add_measurement(self, measurement) -> None:
        self.measurements.append(measurement)

    def plot(self) -> None:
        plt.plot(self.measurements)
        plt.show()

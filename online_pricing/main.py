from online_pricing.simulator import Simulator


def main():
    simulator = Simulator()

    try:
        for i in range(1000):
            simulator.sim_one_day()
    except KeyboardInterrupt:
        print("Interrupted")

    simulator.tracer.plot()


if __name__ == "__main__":
    main()

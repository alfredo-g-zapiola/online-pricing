from online_pricing.simulator import Simulator


def main():
    simulator = Simulator()

    for i in range(1000):
        simulator.sim_one_day()



if __name__ == "__main__":
    main()

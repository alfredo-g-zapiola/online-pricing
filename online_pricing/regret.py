import matplotlib.pyplot as plt
import numpy as np
from online_pricing.tracer import Tracer, moving_average
import matplotlib.pyplot as plt


class Regret(Tracer):
    def __init__(self, optimum) -> None:
        super().__init__()
        self.optimum_total: float = optimum

    """
    Ricevere optimum e rewards giornaliero per calcolare il regret
    """

    def plot(self):
        ma = moving_average(self.avg_reward)
        plt.figure()
        plt.plot(
            self.avg_reward,
            label="Mean Average Regret",
            color="blue",
            linewidth=0.75,
        )
        plt.plot(
            ma,
            label="Moving Average",
            color="red",
        )
        plt.xlabel("Days")
        plt.ylabel("Average Regret")
        plt.legend()
        plt.show()
    # for i in range(n_phases):
    #  t_index = range(i * phases_len, (i + 1) * phases_len)
    # optimum_per_round[t_index] = opt_per_phases[i]
    # ts_istantaneous_regret[t_index] = opt_per_phases[i] - np.mean(ts_rewards_per_experiments, axis=0)[t_index]
    # def plot_rewards(self, rewards_avg_per_day, optimum_total, regret_instant):
    #     plt.figure(0)
    #     plt.plot(np.mean(rewards_avg_per_day, axis=0), "r")
    #     plt.plot(optimum_total, "k--")
    #     plt.legend(["REWARDS DAY BY DAY", "OPTIMUM"])
    #     plt.ylabel("Reward")
    #     plt.xlabel("t")
    #     plt.show()
    #
    # plt.figure(1)
    # # plt.plot(np.cumsum(ts_istantaneous_regret), "r")
    # # plt.plot(np.cumsum(swts_istantaneous_regret), "b")
    # plt.legend(["TS", "SWTS"])
    # plt.ylabel("Regret")
    # plt.xlabel("t")
    # plt.show()

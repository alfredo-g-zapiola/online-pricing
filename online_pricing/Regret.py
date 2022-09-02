import numpy as np
import matplotlib.pyplot as plt


class Regret:
    def __init__(self, optimum_total, optimum_per_group, rewards_total) -> None:
        self.rewards_avg_per_day: list[int] = []
        self.rewards_total: float = rewards_total
        self.regret: list[int] = []
        self.optimum_total: float = optimum_total
        self.optimum_per_group: list[int] = optimum_per_group

    """
    Ricevere optimum e rewards giornaliero per calcolare il regret
    """

    def accumulate_reward(self, reward_per_day):
        self.rewards_avg_per_day.append(reward_per_day)
        return self.rewards_avg_per_day  # a ogni simulazione aggiorna il reward totale

    def compute_regret(self, reward_per_day, optimum):
        return self.regret_instant.append(optimum - reward_per_day)

    # for i in range(n_phases):
    #  t_index = range(i * phases_len, (i + 1) * phases_len)
    # optimum_per_round[t_index] = opt_per_phases[i]
    # ts_istantaneous_regret[t_index] = opt_per_phases[i] - np.mean(ts_rewards_per_experiments, axis=0)[t_index]
    def plot_rewards(self, rewards_avg_per_day, optimum_total, regret_instant):
        plt.figure(0)
        plt.plot(np.mean(rewards_avg_per_day, axis=0), "r")
        plt.plot(optimum_total, "k--")
        plt.legend(["REWARDS DAY BY DAY", "OPTIMUM"])
        plt.ylabel("Reward")
        plt.xlabel("t")
        plt.show()

    plt.figure(1)
    plt.plot(np.cumsum(ts_istantaneous_regret), "r")
    plt.plot(np.cumsum(swts_istantaneous_regret), "b")
    plt.legend(["TS", "SWTS"])
    plt.ylabel("Regret")
    plt.xlabel("t")
    plt.show()

    def compute_asymptotics():
        pass

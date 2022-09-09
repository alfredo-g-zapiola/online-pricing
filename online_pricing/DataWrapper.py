from online_pricing.tracer import moving_average
import numpy as np
import matplotlib.pyplot as plt
class DataWrapper(object):
    def __init__(self, n_sims, n_days):
        self.rewards_mat = np.zeros(shape=(n_sims, n_days))
        self.regrets_mat = np.zeros(shape=(n_sims, n_days))

    def add_measurements(self, rewards, regrets, sample):
        self.rewards_mat[sample, :] = rewards
        self.regrets_mat[sample, :] = regrets

    def plot_all(self):
        mean_rewards = self.rewards_mat.mean(axis=0)
        sdev_rewards = self.rewards_mat.std(axis=0)
        self.plot(mean_rewards, sdev_rewards, "Mean rewards")
        cum_regret = np.cumsum(self.regrets_mat, axis=1)
        mean_regrets = cum_regret.mean(axis=0)
        sdev_regret = cum_regret.std(axis=0)
        self.plot(mean_regrets, sdev_regret, "Mean cumulative regret")

    def plot(self, what,sdev, title):
        ma = moving_average(what)

        plt.figure()
        plt.plot(
            what,
            label="Mean {}".format(title),
            color="blue",
            linewidth=0.75,
        )
        plt.plot(
            what + 3 * sdev,
            label="Upper bound {}".format(title),
            color="pink",
            linewidth=.75
        )
        plt.plot(
            what - 3 * sdev,
            label="Lower bound {}".format(title),
            color="pink",
            linewidth=.75
        )
        plt.plot(
            ma,
            label="Moving Average",
            color="red",
        )
        plt.xlabel("Days")
        plt.ylabel("{}".format(title))
        plt.legend()
        plt.show()




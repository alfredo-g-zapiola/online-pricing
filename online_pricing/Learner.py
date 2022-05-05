import numpy as np

class Learner:
    def __init__(self, n_arms):
        self.t = 0
        self.n_arms = n_arms
        self.rewards = []
        self.rewards_per_arm = [[] for _ in range(n_arms)]
        self.pulled_arms = []

    def reset(self):
        self.__init__(self.n_arms)

    def act(self):
        pass

    def update(self, arm_pulled, reward):
        self.t += 1
        self.rewards.append(reward)
        self.rewards_per_arm[arm_pulled].append(reward)
        self.pulled_arms.append(arm_pulled)



class Ucb(Learner):
    def __init__(self, n_arms, prices):
        super().__init__(n_arms)
        self.means = np.zeros(n_arms)
        self.widths = np.array([np.inf for _ in range(n_arms)])
        self.prices = prices

    def act(self):
        idx = np.argmax((self.means + self.widths))
        return idx

    def update(self, arm_pulled, reward):
        super().update(arm_pulled, reward)
        self.means[arm_pulled] = np.mean(self.rewards_per_arm[arm_pulled])
        for idx in range(self.n_arms):
            n = len(self.rewards_per_arm[idx])
            if n > 0:
                self.widths[idx] = np.sqrt(2*np.max(self.prices)*np.log(self.t)/n)
            else:
                self.widths[idx] = np.inf




class TS_Learner(Learner):
    def __init__(self, n_arms, prices):
        super().__init__(n_arms)
        self.beta_parameters = np.ones((n_arms, 2))
        self.prices = prices

    def pull_arm(self):
        idx = np.argmax(np.array(self.prices) * np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]))
        return idx

    def update(self, pulled_arm, reward):
        super().update(pulled_arm, reward)
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1.0 - reward



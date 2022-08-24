import numpy as np

class Regret:
    def __init__(self,optimum_total,optimum_per_group,rewards_total)->None:
        self.rewards_avg_per_day: list[int] = []
        self.rewards_total: float =rewards_total
        self.regret: list[int] = []
        self.optimum_total: float = optimum_total
        self.optimum_per_group: list[int] = optimum_per_group
    """
    Ricevere optimum e rewards giornaliero per calcolare il regret
    """

    def accumulate_reward(self, reward_per_day):
        rewards_avg_per_day.append(reward_per_day)


    def compute_regret(reward_per_day):
        rewards=reward_per_day(learner)
        optimum=
        regret_instantneous[i]=optimum[i]-rewards[i]
        opt_per_phases = p.max(axis=1)
        optimum_per_round = np.zeros(T)

for i in range(n_phases):
    t_index = range(i * phases_len, (i + 1) * phases_len)
    optimum_per_round[t_index] = opt_per_phases[i]
    ts_istantaneous_regret[t_index] = opt_per_phases[i] - np.mean(ts_rewards_per_experiments, axis=0)[t_index]
    swts_istantaneous_regret[t_index] = opt_per_phases[i] - np.mean(swts_rewards_per_experiments, axis=0)[t_index]

def plot():
  plt.figure(0)
    plt.plot(np.mean(ts_rewards_per_experiments,axis=0),'r')
    plt.plot(np.mean(swts_rewards_per_experiments,axis=0),'b')
    plt.plot(optimum_per_round,'k--')
    plt.legend(['TS','SWTS','OPTIMUM'])
    plt.ylabel('Reward')
    plt.xlabel('t')
    plt.show()

    plt.figure(1)
    plt.plot(np.cumsum(ts_istantaneous_regret),'r')
    plt.plot(np.cumsum(swts_istantaneous_regret),'b')
    plt.legend(['TS','SWTS'])
    plt.ylabel('Regret')
    plt.xlabel('t')
    plt.show()

  def compute_asymptotics():


#import matplotlib.pyplot as plt
#from scipy.stats import wishart, chi2
import numpy as np
from copy import copy

class Social_Influence:

  def __init__(self,prob_matrix):
    # self.simulator=Simulator(seed=41703192)
    self.current_customers=0
    self.n_steps_max=10 #da definire in base ai costumers
    self.prob_matrix=prob_matrix
    self.alpha_ratios=[]



  def simulate_episode(self, init_prob_matrix, n_steps_max):
    """
    Cosa si fa in questa funzine
    :param init_prob_matrix:
    :param n_steps_max:
    :return:
    """
  #simulazione active graph influenza con n step di influenza
    prob_matrix = init_prob_matrix.copy()
    current_customers = prob_matrix.shape[0]
    initial_active_customers = np.random.binomial(1, 0.1, size=current_customers)
    history = np.array([initial_active_customers])
    active_customers = initial_active_customers
    newly_active_customers = active_customers
    active_edges=np.zeros(prob_matrix.shape[0],prob_matrix.shape[1])

    t = 0
    graph=np.zeros((prob_matrix.shape[0],prob_matrix.shape[0]))
    for i in initial_active_customers:
      if (initial_active_customers[i]!=0 & graph[i,i]==0):
        graph[i,i]=+1 #diventa 1 la diag del cliente attivo

    #creazione active graph con n_steps_max iterazioni
    while t < n_steps_max and np.sum(newly_active_customers) > 0:
      p = (prob_matrix.T * active_customers).T
      #introdurre condizione per tracciare l'influenza tra i nodi
      #inserire ciclo di controllo indicizzato per salvare chi ha influenzato chi
      for i in p:
        influence_check=np.random.rand(p.shape[0], p.shape[0])
        for j in range (i, influence_check.shape[0]):
          active_edges[i,j] = p[i] >influence_check[i,j]  #servono if #cambiare dim np.rand
          if ( graph[i, i] == 0 & graph[i, j] == 0):
            graph[i, i] = +1
            graph[i, j] = +1  # diventa 1 la diag del cliente attivo

      prob_matrix = prob_matrix * ((p != 0) == active_edges)
      newly_active_customers = (np.sum(active_edges, axis=0) > 0) * (1 - active_customers)
      active_customers = np.array(active_customers + newly_active_customers)
      history = np.concatenate((history, [newly_active_customers]), axis=0)
      t += 1

    return history #registra active graph


  def simulate_influence(self, init_data):
    # init data e' viene come [array_client, array_quale_prodtto, ary_rating]
    # if rating is 5, add 20% to the random influence (in probability matrix)
    # if rating is 4, add 10%
    # if rating is 3, just the random one
    # if rating is 2 or 1, the initial client is no longer a seed
    newly_active_customers = [0, 1]
    products = [4, 5]
    return [newly_active_customers, products]

  def estimate_probabilities(self, dataset, node_index, n_nodes):
    #dataset = collezione delle history
    estimated_prob = np.ones(n_nodes) * 1.0 / (n_nodes - 1)
    credits = np.zeros(n_nodes)
    occurr_v_active = np.zeros(n_nodes)
    n_episodes = len(dataset) #numero di grafi simulati
    for episode in dataset:
      idx_w_active = np.argwhere(episode[:, node_index] == 1).reshape(-1)
      if len(idx_w_active) > 0 and idx_w_active > 0:
        active_nodes_in_prev_step = episode[idx_w_active - 1, :].reshape(-1)
        credits += active_nodes_in_prev_step / np.sum(active_nodes_in_prev_step)
      for v in range(0, n_nodes):
        if v != node_index:
          idx_v_active = np.argwhere(episode[:, v] == 1).reshape(-1)
          if len(idx_v_active) > 0 and (idx_v_active < idx_w_active or len(idx_w_active) == 0):
            occurr_v_active[v] += 1
    estimated_prob = credits / occurr_v_active
    estimated_prob = np.nan_to_num(estimated_prob)
    return estimated_prob


#n_nodes = 5
#n_episodes = 1000
#prob_matrix = np.random.uniform(0, 0.1, (n_nodes, n_nodes))
#node_index = 4
#dataset = []
#for i in range(0, n_episodes):
#  dataset.append(simulate_episode(init_prob_matrix=prob_matrix, n_steps_max=10))
#
#estimated_prob = estimate_probabilities(dataset=dataset, node_index=node_index, n_nodes=n_nodes)



from typing import cast

import numpy as np
import numpy.typing as npt

class SocialInfluence:
    def __init__(self, n_products=5, n_episodes: int = 5, n_steps_max: int = 10):
        self.n_episodes = n_episodes
        self.n_steps_max = n_steps_max  # da definire in base ai costumers
        self.dataset = list()

    def simulate_episode(self, init_prob_matrix: list[list[int]], n_step_max: int):
        init_prob_matrix = cast(npt.NDArray[int, int], np.array(init_prob_matrix))

        prob_matrix = init_prob_matrix.copy()
        n_products = prob_matrix.shape[0]

        initial_active_products = np.random.binomial(1, 0.1, size=n_products)
        history = np.array([initial_active_products])

        newly_active_products = active_products = initial_active_products

        for iteration in range(n_step_max):
            if np.sum(newly_active_products) > 0:
                break

            p = (prob_matrix.T * active_products).T

            active_edges = p > np.random.rand(p.shape[0], p.shape[1])
            prob_matrix = prob_matrix * ((p != 0) == active_edges)  # aggiornamento prob matrix

            newly_active_products = (np.sum(active_edges, axis=0) > 0) * (1 - active_products)
            active_products = np.array(active_products + newly_active_products)

            history = np.concatenate((history, [newly_active_products]), axis=0)

        return history  # storico nodi (prodotti) attivati

    def add_episode(self, episode: list[list[int]]) -> None:
        """
        Adds the episode of a user. Consumed at sim_one_user
        :param episode: a list of lists
        """
        self.dataset.append(episode)

    def estimate_probabilities(self, node_index: int, n_products: int) -> npt.NDArray[int]:
        """
        Stima la probabilità di influence dal prodotto node_index a tutti gli altri.

        :param node_index: il prodotto di partenza
        :return:
        """
        # estimated_prob = np.ones(n_products) * 1.0 / (n_products - 1) <- NOT USED
        credit = np.zeros(n_products)
        occurr_v_active = np.zeros(n_products)

        for episode in self.dataset:
            idx_w_active = np.argwhere(episode[:, node_index] == 1).reshape(-1)

            if len(idx_w_active) > 0 and idx_w_active > 0:
                active_products_in_prev_step = episode[idx_w_active - 1, :].reshape(-1)
                credit += active_products_in_prev_step / np.sum(active_products_in_prev_step)

            for v in range(0, n_products):
                if v != node_index:
                    idx_v_active = np.argwhere(episode[:, v] == 1).reshape(-1)
                    if len(idx_v_active) > 0 and (
                        idx_v_active < idx_w_active or len(idx_w_active) == 0
                    ):
                        occurr_v_active[v] += 1

        estimated_prob = credit / occurr_v_active
        estimated_prob = np.nan_to_num(estimated_prob)
        return estimated_prob
    #
    # def __test(self):
    #     n_products = 5
    #     n_episodes = 1000
    #     prob_matrix = np.random.uniform(
    #         0.0, 0.1, (n_products, n_products)
    #     )  # initial matrix provided by flavio (?)
    #     node_index = 4
    #     dataset = []
    #
    #     for e in range(0, n_episodes):
    #         dataset.append(self.simulate_episode(init_prob_matrix=prob_matrix, n_step_max=10))
    #
    #     estimate_prob = self.estimate_probabilities(node_index=2, n_products=n_products)
    #     final_list = [self.estimate_probabilities(dataset, i, n_products) for i in range(5)]
    #     # [0.30 0.18 0.25 0. ]
    #     print("True P matrix:   ", prob_matrix[:, 4])
    #     print("Estimated P matrix:  ", estimate_prob)

from typing import cast

import numpy as np
import numpy.typing as npt

from online_pricing.learner import Learner, TSLearner


class SocialInfluence:
    def __init__(self, n_products: int, secondaries: list[list[int]], lambda_param: float, n_episodes: int = 5) -> None:
        self.lambda_param = lambda_param
        self.n_episodes = n_episodes
        self.n_products = n_products
        self.dataset: list[npt.NDArray[int]] = list()
        self.secondaries = secondaries
        # number of edges (we have one learner for each one)
        self.learners: list[tuple[Learner, Learner]] = [(TSLearner(1, [0]), TSLearner(1, [0])) for i in range(n_products)]

    def add_episode(self, episode: list[list[int]]) -> None:
        """
        Adds the episode of a user. Consumed at sim_one_user
        :param episode: a list of lists
        """
        self.dataset.append(np.array(episode))

    def estimate_probabilities(self) -> list[list[int | float]]:
        """
        Look into all episodes of the day, and updates the beta distribution of each learner
        :return: the estimated probabilities of the edges
        """
        # print("Estimating today's probabilities")
        for episode in self.dataset:
            # print("Current episode ", episode)
            # print(episode)
            # print(np.argwhere(episode[0] != 0))
            prod = np.argwhere(episode[0] != 0)  # starting node
            if not len(prod) > 0:
                # we did not even buy the landing product
                # print("Did not buy the landing product")
                continue
            else:
                prod = prod[0][0]
            # print(self.secondaries[prod][0])
            # print(episode[1][self.secondaries[prod][0]])
            reward_first_sec = episode[1][self.secondaries[prod][0]]
            reward_sec_sec = episode[1][self.secondaries[prod][1]]
            # print("Bought product: ", prod, "\nThe rewards were: ", reward_first_sec, reward_sec_sec)
            # update plearner
            self.learners[prod][0].update(0, reward_first_sec)
            self.learners[prod][1].update(0, reward_sec_sec)
        self.dataset = list()  # we do not need today's data anymore

        # estimated edge probabilities. The edge proba is 0 if it is not in the secondary products
        estimated_edge_probas = [
            [
                min([1, self.learners[i][(0 if j == self.secondaries[i][0] else 1)].mean_arm(0) / 1])
                # (1 if j == self.secondaries[i][0] else self.lambda_param)])
                if j in self.secondaries[i] else 0
                for j in range(5)
            ]
            for i in range(self.n_products)
        ]

        return estimated_edge_probas

    def estimate_probabilities_old(self, node_index: int, n_products: int) -> list[float]:
        """
        Estimates the probabilities of the edges of the graph from node_index to the other nodes

        :param node_index: the primary product
        :param n_products: the number of products
        :return: a list of probabilities
        """
        # estimated_prob = np.ones(n_products) * 1.0 / (n_products - 1)
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
                    if len(idx_v_active) > 0 and (idx_v_active < idx_w_active or len(idx_w_active) == 0):
                        occurr_v_active[v] += 1
        estimated_prob = credit / occurr_v_active
        estimated_prob = np.nan_to_num(estimated_prob)
        return cast(list[float], estimated_prob.tolist())

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

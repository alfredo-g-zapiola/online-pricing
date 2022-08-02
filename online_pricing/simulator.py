import random
from typing import Any, Type, TypeVar, Union

import numpy as np

from online_pricing.environment import EnvironmentBase, GreedyEnvironment
from online_pricing.learner import Learner, TSLearner
from online_pricing.social_influence import SocialInfluence


class Simulator(object):
    def __init__(self, environment: Type[EnvironmentBase] = EnvironmentBase, seed: int = 41703192):
        self.seed = seed
        self.groups = range(3)
        self.__SocialInfluence = None
        self.environment = environment()
        self.prices = [19.99, 16.7, 12.15, 8.0, 35.0]
        # lambda to go to second secondary product
        self._lambda = 0.5
        # daily data
        self._daily_data = dict()
        self._users_data = dict()
        self.current_learner: Learner | None = None

    def _init_r(self):
        pass

    def sim_one_day(self) -> None:
        """
        Simulate what happens in one day.

        This function simulates what would happen in a real world scenario.
        Clients interact with a primary product. Each client belongs to a group which has
        different probability distributions - meaning behaviours - that determines the outcome of a
        visit. After each client interaction, the current learner (belonging to the current
        configuration of prices) is updated. Then, the cumulative sold products array is updated
        along with the empirical influence matrix that records the jumps to secondary products.
        """
        direct_clients = self.environment.get_direct_clients()
        products_sold: list[int, int] = [0] * self.environment.n_products
        influence_matrix = [[0] * self.environment.n_products] * self.environment.n_products

        for group in self.groups:
            for client_id, primary_product in direct_clients[f"group_{group}"]:
                buys, influenced = self.sim_one_user(
                    group=group,
                    client_id=client_id,
                    product_id=primary_product,
                    product_graph=self.environment.distributions_parameters["product_graph"].copy(),
                    prices=self.prices,
                )
                self.update_learner(buys)
                products_sold = sum_by_element(products_sold, buys)
                influence_matrix = sum_by_element(influence_matrix, influenced)

    def sim_buy(self, group: int, product_id: int, price: float) -> int:
        """
        Simulate the buy of a product for a user belonging to a group.

        If the willing_price of a user is higher than the price of the product, the user will buy
        it. Next, the quantity of units bought will be decided, independently of the
        willing_price of a user, by sampling a probability distribution.

        :param group: group of the user
        :param product_id: product id
        :param price: price of the product
        :return: number of units bought
        """
        willing_price = self.environment.sample_demand_curve(group=group, prod_id=product_id)
        n_units = 0
        if price < willing_price:
            n_units = self.environment.sample_quantity_bought(group)

        return n_units

    def sim_one_user(
        self, group: int, client_id: int, product_id: int, product_graph: Any, prices: list[float]
    ) -> tuple[list[int], list[list[int]]]:
        """
        Function to simulate the behavior of a single user.

        A user may start from a primary product and then buy secondary products. Next, for each
        secondary product, he may buy a number of units and interact with other secondary products.
        This is done recursively while updating the product_graph matrix to remove cycles.
        The secondary products are chosen based on the product_graph matrix, using the most
        probable product to be bought. This is product is fixed and selected clairvoyantly by an
        external company. In the end, an array with the number of units bought per product is
        returned, along with the influence matrix.

        :param group: group of the user
        :param client_id: id of the user
        :param product_id: product to simulate the behavior of the user
        :param product_graph: secondary product probability graph
        :param prices: prices of the products
        :return: list of number of units bought per product and the influencing matrix
        """
        # Instantiate buys and influence matrix
        buys: list[int] = [
            0 if idx != product_id else self.sim_buy(group, product_id, prices[product_id])
            for idx in range(self.environment.n_products)
        ]
        influence_matrix = [[0] * self.environment.n_products] * self.environment.n_products

        # If user didn't buy primary_product, return
        if not buys[product_id]:
            return buys, influence_matrix

        # Remove cycles
        product_graph[:, product_id] = 0

        # Argmax of probabilities
        first_advised = np.argsort(product_graph[product_id])[-1]
        if random.random() < product_graph[product_id][first_advised]:
            # Update influence matrix with the current jump
            influence_matrix[product_id][first_advised] += 1
            # Simulate recursively
            following_buys, following_influence = self.sim_one_user(
                group, client_id, first_advised, product_graph, prices
            )
            # Update buys and influence matrix from recursive call
            buys = sum_by_element(buys, following_buys)
            influence_matrix = sum_by_element(influence_matrix, following_influence)

        # Do the same with the second secondary product
        second_advised = np.argsort(product_graph[product_id])[-2]
        if random.random() < product_graph[product_id][second_advised] * self._lambda:
            influence_matrix[product_id][second_advised] += 1
            following_buys, following_influence = self.sim_one_user(
                group, client_id, second_advised, product_graph, prices
            )
            buys = sum_by_element(buys, following_buys)
            influence_matrix = sum_by_element(influence_matrix, following_influence)

        return buys, influence_matrix

    # TODO: might be per group
    def update_learner(self, buys: list[int]) -> None:
        """
        Update current learner with the buys of the user.

        The reward is 1 if the user bought a product, 0 otherwise.

        :param buys: list of number of units bought per product
        """
        did_buy = [int(buy > 0) for buy in buys]

        for idx, bought in enumerate(did_buy):
            self.current_learner.update(arm_pulled=idx, reward=bought)

    def formula(self, alpha, conversion_rates, margins, influence_probability):
        s1 = 0
        s2 = 0
        sum = 0
        for i in range(0,5):
            s1 += conversion_rates[i]*margins[i]
            for j in range(0,5):
                if i != j:
                    s2 += influence_probability[i][j]*conversion_rates[j]*margins[j]
            sum = sum + alpha[i]*(s1 + s2)
            s2 = 0
            s1 = 0
        return sum

    def __wrap_influence_probability(self):
        """
        Adjust for conversion rates and for the lambda (in case of second seconary product)
        for every starting node in influence_probability
        :return:
        """
        pass

    def greedy_algorithm(self, alpha, conversion_rates, margins, influence_probability):        #greedy alg without considering groups. Alpha il a list of 5 elements,
        prices = [0, 0,0,0,0]                                                                    #conversion_rates and margins are matrix 5x4 (products x prices)
        max = self.formula(alpha, conversion_rates[:,0], margins[:,0], influence_probability)   #influence_probability is a matrix 5x5 (products x products) where cell ij is the
        while True:                                                                             #probability to go from i to j
            changed = False
            best = prices
            for i in range(0,5):
                temp = prices
                cr = []
                mr = []
                temp[i] += 1
                if temp[i] > 3:
                    temp[i] = 3
                for j in range(0,5):
                    cr.append(conversion_rates[j, temp[j]])
                    mr.append(margins[j, temp[j]])
                n = self.formula(alpha, cr, mr, influence_probability)
                if n > max:
                    max = n
                    best = temp
                    changed = True
            if not changed:
                return best
            if changed:
                prices = best


    def formula_with_groups(self, list_alpha, conversion_rates, margins, influence_probability):
        s1 = 0
        s2 = 0
        sum = 0
        for g in range (0,3):
            for i in range(0,5):
                s1 += conversion_rates[g][i]*margins[i]
                for j in range(0,5):
                    if i != j:
                        s2 += influence_probability[i][j]*conversion_rates[g][j]*margins[j]
                sum = sum + list_alpha[g][i]*(s1 + s2)
                s2 = 0
                s1 = 0
        return sum

    def greedy_algorithm_with_groups(self, list_alpha, list_conversion_rates, margins, influence_probability):      #greedy algorithm considering groups. Same as before except for:
        prices = [0,0,0,0,0]                                                                                        #list_alpha is a list of 3 lists where each list contains 5 elements
        conversion_rates = []                                                                                       #list_conversion_rates is a list of 3 matrices where each matrix is 5x4 (products x prices)
        for g in range(0,3):
            conversion_rates.append(list_conversion_rates[g][:,0])
        max = self.formula_with_groups(list_alpha, conversion_rates, margins[:,0], influence_probability)
        while True:
            changed = False
            best = prices
            for i in range(0,5):
                temp = prices
                cr = [[0 for v in range(5)] for w in range(3)]
                mr = []
                temp[i] += 1
                if temp[i] > 3:
                    temp[i] = 3
                for g in range(0,3):
                    for j in range(0,5):
                        cr[g][j] = list_conversion_rates[g][j, temp[j]]
                for k in range (0,5):
                    mr.append(margins[k, temp[k]])
                n = self.formula_with_groups(list_alpha, cr, mr, influence_probability)
                if n > max:
                    max = n
                    best = temp
                    changed = True
            if not changed:
                return best
            if changed:
                prices = best





    # TODO: here goes the greedy part of the simulation
    # def sim_one_day_greedy(self):
    #     # TODO define well social influence build matrix HERE
    #     self.__SocialInfluence = SocialInfluence(self.environment.sample_n_users())
    #     direct_clients = self.environment.get_direct_clients()
    #
    #     self._users_data = dict()
    #     # Secondary products probability graph
    #     product_graph = self.environment.distributions_parameters["product_graph"]
    #     # Initial configuration
    #     base_prices = [learner.act() for learner in self.environment.Learners]
    #
    #     # Users n_units bought per product
    #     buys_dict = dict()
    #
    #     # Simulate buys for direct clients
    #     for g in range(3):  # for each group
    #         for client, product in direct_clients["group_" + str(g)]:  # for each product
    #             buys_dict[client] = self.sim_one_user(
    #                 group=g,
    #                 client=client,
    #                 product=product,
    #                 product_graph=product_graph,
    #                 prices=base_prices,
    #             )
    #
    #     # We now see how these customers have influenced their contacts and simulate what happens to those
    #     # they brought to our product websites
    #     influenced_clients_prods: dict[str, int] = self.__SocialInfluence.simulate_influence(
    #         self._users_data
    #     )
    #
    #     # Simulate buys for indirect clients
    #     for g in range(3):
    #         for client, product in influenced_clients_prods.items():
    #             buys_dict[client] = self.sim_one_user(
    #                 group=g,
    #                 client=client,
    #                 product=product,
    #                 product_graph=product_graph,
    #                 prices=base_prices,
    #             )
    #
    #     # Calculate reward for configuration based on n_units sold
    #     max_cumulative_expected_margin = get_buys_reward(buys_dict, margins=[])
    #
    #     # New configurations
    #     n_learners = 5
    #     next_prices = [
    #         [learner.act() for learner in self.environment.Learners] for _ in range(n_products)
    #     ]
    #     # Update only one learner per configuration
    #     for new_idx in range(n_learners):
    #         next_prices[new_idx][new_idx] = self.environment.Learners[new_idx].greedy_act()
    #
    #     # Index of bes configuration, if None terminate
    #     best_update: int | None = None
    #
    #     # For each learner, re-test daily user
    #     for idx in range(n_learners):
    #         buys_dict = dict()
    #         for g in range(3):
    #             for client, product in direct_clients["group_" + str(g)]:  # for each webpage
    #                 buys_dict[client] = self.sim_one_user(
    #                     group=g,
    #                     client=client,
    #                     product=product,
    #                     product_graph=product_graph,
    #                     prices=base_prices,
    #                 )
    #             for client, product in influenced_clients_prods.items():
    #                 buys_dict[client] = self.sim_one_user(
    #                     group=g,
    #                     client=client,
    #                     product=product,
    #                     product_graph=product_graph,
    #                     prices=next_prices[idx],
    #                 )
    #
    #         # Calculate reward
    #         cumulative_expected_margin = get_buys_reward(buys_dict, margins=[])
    #
    #         # If better than initial_config/next_best_config
    #         if cumulative_expected_margin > max_cumulative_expected_margin:
    #             # Update best config
    #             best_update = idx
    #             max_cumulative_expected_margin = cumulative_expected_margin
    #
    #     # Return best config == learner index and reward
    #     return best_update, max_cumulative_expected_margin
    #
    #
    #
    # def greedy_simulation(self, n_iterations: int = 365) -> None:
    #     greedy_environment = GreedyEnvironment
    #     simulator = Simulator(greedy_environment)
    #     max_cumulative_expected_margin = 0
    #
    #     for t in range(n_iterations):
    #         best_update, max_cumulative_expected_margin = simulator.sim_one_day_greedy()
    #
    #         if best_update is None:
    #             return
    #
    #         greedy_environment.round(best_update, max_cumulative_expected_margin)


def sum_by_element(array_1: list[Any], array_2: list[Any]) -> list[Any]:
    """Sum lists - or matrices - by element."""
    if type(array_1) is not type(array_2):
        raise TypeError(f"Arrays must be of the same type, got {type(array_1)} and {type(array_2)}")

    if isinstance(array_1[0], list):
        return [sum_by_element(a1, a2) for a1, a2 in zip(array_1, array_2)]

    return [sum(items) for items in zip(array_1, array_2)]


def get_buys_reward(buys_dict: dict[str, dict[int, int]], margins: Any) -> int:
    """Calculate reward from margin detail and number of buys"""
    pass

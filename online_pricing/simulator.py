import random
from typing import Any, Type

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

    def sim_buy(self, group: int, product_id: int, price: float) -> int:
        """
        Simulate a buy of a product for a user belonging into a group.

        If the willing_price of a user is higher than the price of the product, the user will buy
        it. Then, the quantity of the number of units bought will be decided, independently of the
        willing_price of a user, tby sampling a probability distribution.

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
        self, group: int, client_id: str, product_id: int, product_graph: Any, prices: list[float]
    ) -> list[int]:
        """
        Function to simulate the behavior of a single user.

        Each entry of the returned array corresponds to a product and its quantity bought.
        A user may start from a primary product and then buy secondary products. Then, for each
        secondary product, he may buy a number of units and interact with other secondary products.
        This is done recursively while updating the product_graph matrix to remove cycles.
        The secondary products are chosen based on the product_graph matrix, by choosing the most
        probable product to be bought. This is handled clairvoyantly by an external company.

        :param group: group of the user
        :param client_id: id of the user
        :param product_id: product to simulate the behavior of the user
        :param product_graph: secondary product probability graph
        :param prices: prices of the products
        :return: list of number of units bought per products
        """
        buys: list[int] = [
            0 if idx != product_id else self.sim_buy(group, product_id, prices[product_id])
            for idx in range(self.environment.n_products)
        ]
        if not buys[product_id]:
            return buys

        # Remove cycles
        product_graph[:, product_id] = 0
        # Argmax of probabilities
        first_advised = np.argsort(product_graph[product_id])[-1]
        if random.random() < product_graph[product_id][first_advised]:
            # This is a sum by element of a pair of arrays
            buys = [
                sum(products)
                # The second entry is a recursion over the advised product
                for products in zip(
                    buys, self.sim_one_user(group, client_id, first_advised, product_graph, prices)
                )
            ]

        second_advised = np.argsort(product_graph[product_id])[-2]
        if random.random() < product_graph[product_id][second_advised] * self._lambda:
            buys = [
                sum(products)
                for products in zip(
                    buys, self.sim_one_user(group, client_id, second_advised, product_graph, prices)
                )
            ]

        return buys

    def update_learners(self, learner: Learner, buys: list[int]) -> None:
        """
        Update current learner with the buys of the user.

        The reward is 1 if the user bought a product, 0 otherwise.

        :param learner: learner to update
        :param buys: list of number of units bought per product
        """
        did_buy = [int(buy > 0) for buy in buys]

        for idx, bought in enumerate(did_buy):
            learner.update(arm_pulled=idx, reward=bought)

    def sim_one_day(self):
        direct_clients = self.environment.get_direct_clients()
        self._users_data = dict()
        products_sold: list[int, int] = [0 for _ in range(self.environment.n_products)]

        for group in self.groups:  # for each group
            for client_id, first_product in direct_clients[f"group_{group}"]:  # for each webpage
                buys = self.sim_one_user(
                    group=group,
                    client_id=client_id,
                    product_id=first_product,
                    product_graph=self.environment.distributions_parameters["product_graph"].copy(),
                    prices=self.prices,
                )
                self.update_learners(self.current_learner, buys)
                products_sold = [sum(products) for products in zip(buys, products_sold)]

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


def get_buys_reward(buys_dict: dict[str, dict[int, int]], margins: Any) -> int:
    """Calculate reward from margin detail and number of buys"""
    pass

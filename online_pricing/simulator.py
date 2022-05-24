import random
from typing import Any

import numpy as np

from online_pricing.environment import EnvironmentBase, GreedyEnvironment
from online_pricing.social_influence import SocialInfluence


class Simulator(object):
    def __init__(self, environment=GreedyEnvironment, seed=41703192):
        self.seed = seed
        self.__SocialInfluence = None
        self.environment = environment
        self.prices = [19.99, 16.7, 12.15, 8.0, 35.0]
        # lambda to go to second secondary product
        self.__lambda = 0.5
        # daily data
        self.__daily_data = dict()
        self.__users_data = dict()

    def __init_R(self):
        pass

    def sim_one_day(self):
        # TODO define well social influence build matrix HERE
        self.__SocialInfluence = SocialInfluence(self.environment.sample_n_users())
        direct_clients = self.environment.get_direct_clients()

        self.__users_data = dict()

        for g in range(3):  # for each group
            for client, site in direct_clients["group_" + str(g)]:  # for each webpage
                self.sim_one_user(group=g, client=client, first_product=site)

        # We now see how these customers have influenced their contacts and simulate what happens to those
        # they brought to our product websites
        influenced_clients_prods = self.__SocialInfluence.simulate_influence(self.__users_data)

        for client, product in influenced_clients_prods:
            self.sim_one_user(group=g, client=client, first_product=site)
            # Filippo & Flavio TODO send feedback to Learner

        # Filippo & Flavio  TODO make learner set prices for tomorrow

    def sim_one_day_greedy(self):
        # TODO define well social influence build matrix HERE
        self.__SocialInfluence = SocialInfluence(self.environment.sample_n_users())
        direct_clients = self.environment.get_direct_clients()

        self.__users_data = dict()
        # Secondary products probability graph
        product_graph = self.environment.__distributions_parameters["product_graph"]
        # Initial configuration
        base_prices = [learner.act() for learner in self.environment.Learners]

        # Users n_units bought per product
        buys_dict = dict()

        # Simulate buys for direct clients
        for g in range(3):  # for each group
            for client, product in direct_clients["group_" + str(g)]:  # for each product
                buys_dict[client] = self.sim_one_user(
                    group=g,
                    client=client,
                    product=product,
                    product_graph=product_graph,
                    prices=base_prices,
                )

        # We now see how these customers have influenced their contacts and simulate what happens to those
        # they brought to our product websites
        influenced_clients_prods: dict[str, int] = self.__SocialInfluence.simulate_influence(
            self.__users_data
        )

        # Simulate buys for indirect clients
        for g in range(3):
            for client, product in influenced_clients_prods.items():
                buys_dict[client] = self.sim_one_user(
                    group=g,
                    client=client,
                    product=product,
                    product_graph=product_graph,
                    prices=base_prices,
                )

        # Calculate reward for configuration based on n_units sold
        max_cumulative_expected_margin = get_buys_reward(buys_dict, margins=[])

        # New configurations
        n_learners = 5
        next_prices = [
            [learner.act() for learner in self.environment.Learners] for _ in range(n_products)
        ]
        # Update only one learner per configuration
        for new_idx in range(n_learners):
            next_prices[new_idx][new_idx] = self.environment.Learners[new_idx].greedy_act()

        # Index of bes configuration, if None terminate
        best_update: int | None = None

        # For each learner, re-test daily user
        for idx in range(n_learners):
            buys_dict = dict()
            for g in range(3):
                for client, product in direct_clients["group_" + str(g)]:  # for each webpage
                    buys_dict[client] = self.sim_one_user(
                        group=g,
                        client=client,
                        product=product,
                        product_graph=product_graph,
                        prices=base_prices,
                    )
                for client, product in influenced_clients_prods.items():
                    buys_dict[client] = self.sim_one_user(
                        group=g,
                        client=client,
                        product=product,
                        product_graph=product_graph,
                        prices=next_prices[idx],
                    )

            # Calculate reward
            cumulative_expected_margin = get_buys_reward(buys_dict, margins=[])

            # If better than initial_config/next_best_config
            if cumulative_expected_margin > max_cumulative_expected_margin:
                # Update best config
                best_update = idx
                max_cumulative_expected_margin = cumulative_expected_margin

        # Return best config == learner index and reward
        return best_update, max_cumulative_expected_margin

    def sim_buy(self, group: int, prod_id: int, price: int) -> int:
        willing_price = self.environment.sample_demand_curve(group=group, prod_id=prod_id)
        n_units = 0
        if price < willing_price:
            n_units = self.environment.sample_quantity_bought(group)

        return n_units

    def sim_one_user(
        self, group: int, client: str, product: int, product_graph: Any, prices: list[int]
    ) -> dict[int, int]:
        buys: dict[int, int] = {product: self.sim_buy(group, product, prices[product])}
        if not buys[product]:
            return buys

        # Remove cycles
        product_graph[:, product] = 0

        # Argmax of probabilities
        first_advised = np.argsort(product_graph[product])[-1]
        if random.random() < product_graph[product][first_advised]:
            # Recursion on next product
            buys.update(self.sim_one_user(group, client, first_advised, product_graph, prices))

        second_advised = np.argsort(product_graph[product])[-2]
        if random.random() < product_graph[product][second_advised] * self.__lambda:
            buys.update(self.sim_one_user(group, client, second_advised, product_graph, prices))

        return buys

    def greedy_simulation(self, n_iterations: int = 365) -> None:
        greedy_environment = GreedyEnvironment
        simulator = Simulator(greedy_environment)
        max_cumulative_expected_margin = 0

        for t in range(n_iterations):
            best_update, max_cumulative_expected_margin = simulator.sim_one_day_greedy()

            if best_update is None:
                return

            greedy_environment.round(best_update, max_cumulative_expected_margin)


def get_buys_reward(buys_dict: dict[str, dict[int, int]], margins: Any) -> int:
    """Calculate reward from margin detail and number of buys"""
    pass

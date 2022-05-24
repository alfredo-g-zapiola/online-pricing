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

    def sim_one_day_greedy(self, first_iteration: bool, max_cumulative_expected_margin: int):
        # TODO define well social influence build matrix HERE
        self.__SocialInfluence = SocialInfluence(self.environment.sample_n_users())
        direct_clients = self.environment.get_direct_clients()

        self.__users_data = dict()

        buys_dict = dict()
        for g in range(3):  # for each group
            for client, site in direct_clients["group_" + str(g)]:  # for each webpage
                buys_dict[client] = self.sim_one_user(group=g, client=client, first_product=site)

        # We now see how these customers have influenced their contacts and simulate what happens to those
        # they brought to our product websites
        influenced_clients_prods = self.__SocialInfluence.simulate_influence(self.__users_data)

        best_configuration = 0
        change = False
        if first_iteration:  # first iteration just compute the margin of configuration [0,0,0,0,0]
            for client, product in influenced_clients_prods:
                max_cumulative_expected_margin = (
                    self.sim_one_user_greedy
                )  # it's a function that computes cumulative margin given people and a configuration of prices
        else:
            for i in range(
                5
            ):  # i need to evaluate the best new configuration between the 5 computed in the environment
                for client, product in influenced_clients_prods:
                    cumulative_expected_margin = self.sim_one_user_greedy
                    if cumulative_expected_margin > max_cumulative_expected_margin:
                        best_configuration = i
                        max_cumulative_expected_margin = cumulative_expected_margin
                        change = True

        return change, best_configuration, max_cumulative_expected_margin

    def sim_buy(self, group, prod_id) -> int:
        willing_price = self.environment.sample_demand_curve(group=group, prod_id=prod_id)
        n_units = 0
        if self.environment.prices_and_margins[f"product_{prod_id}"] > willing_price:
            n_units = self.environment.sample_quantity_bought(group)

        return n_units

    def sim_one_user(
        self, group: str, client: str, product: int, product_graph: Any
    ) -> dict[int, int]:
        buys: dict[int, int] = {product: self.sim_buy(group, product)}
        if not buys[product]:
            return buys

        # Remove cycles
        product_graph[:, product] = 0

        # Argmax of probabilities
        first_advised = np.argsort(product_graph[product])[-1]
        if random.random() < product_graph[product][first_advised]:
            buys.update(self.sim_one_user(group, client, first_advised, product_graph))

        second_advised = np.argsort(product_graph[product])[-2]
        if random.random() < product_graph[product][second_advised] * self.__lambda:
            buys.update(self.sim_one_user(group, client, second_advised, product_graph))

        return buys

    def send_data_to_learner(self):
        pass

    def greedy_simulation(self):
        greedy_environment = GreedyEnvironment
        simulator = Simulator(greedy_environment)
        max_cumulative_expected_margin = 0
        T = 365

        for t in range(T):
            if t == 0:  # first iteration just compute the margin of configuration [0,0,0,0,0]
                change, best_conf, max_cumulative_expected_margin = simulator.sim_one_day_greedy(
                    True, 0
                )
                greedy_environment.round(greedy_environment.Learners)
            else:
                change, best_conf, max_cumulative_expected_margin = simulator.sim_one_day_greedy(
                    False, max_cumulative_expected_margin
                )
                if change == False:
                    return  # if there isn't a new best configuration, return
                else:
                    greedy_environment.round(
                        greedy_environment.configurations[best_conf]
                    )  # else the environment will compute the 5 new configuratoion

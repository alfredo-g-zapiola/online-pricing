import numpy as np

from online_pricing.environment import GreedyEnvironment
from online_pricing.social_influence import SocialInfluence
from online_pricing.environment import EnvironmentBase, GreedyEnvironment


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

        for g in range(3):  # for each group
            for client, site in direct_clients["group_" + str(g)]:  # for each webpage
                self.sim_one_user(group=g, client=client, first_product=site)

        # We now see how these customers have influenced their contacts and simulate what happens to those
        # they brought to our product websites
        influenced_clients_prods = self.__SocialInfluence.simulate_influence(self.__users_data)

        best_configuration = 0
        change = False
        if first_iteration:                                                 #first iteration just compute the margin of configuration [0,0,0,0,0]
            for client, product in influenced_clients_prods:
                max_cumulative_expected_margin = self.sim_one_user_greedy   #it's a function that computes cumulative margin given people and a configuration of prices
        else:
            for i in range(5):                                              #i need to evaluate the best new configuration between the 5 computed in the environment
                for client, product in influenced_clients_prods:
                    cumulative_expected_margin = self.sim_one_user_greedy
                    if cumulative_expected_margin > max_cumulative_expected_margin:
                        best_configuration = i
                        max_cumulative_expected_margin = cumulative_expected_margin
                        change = True

        return change, best_configuration, max_cumulative_expected_margin

    def sim_one_user(self, group, client, first_product):
        """
        :param group:
        :param client:
        :return:
        """

        def sim_buy(group, prod_id):
            willing_price = self.environment.sample_demand_curve(group=group, prod_id=prod_id)
            bought = False
            if self.environment.prices_and_margins["product_{}".format(first_product)] > willing_price:
                qty = self.environment.sample_quantity_bought(group)
                # send data to learner
                self.environment.Learners[prod_id].update(self.prices[prod_id],
                                                          qty *
                                                          self.environment.prices_and_margins["product_" +
                                                                                              str(prod_id)])
                bought = False
            else:
                # send data to learner
                self.environment.Learners[prod_id].update(self.prices[prod_id], 0)  # 0 since we did not sell

                bought = True
            return bought

        bought_first = sim_buy(group, first_product)
        if bought_first:
            first_secondary = None  # TODO sample first secondary
            goes_to_first_second = True  # TODO flip coin and compare with probability of affinity
            if goes_to_first_second:
                bought_second = sim_buy(group, first_secondary)
                goes_to_second_second = self.__lambda * 0  # TODO sample
                second_secondary = 3  # TODO sample
                if goes_to_second_second:
                    bought_third = sim_buy(group, second_secondary)
                else:
                    print(
                        "Bought the first product, the first secondary but not the second secondary"
                    )
            else:
                print("Bought the first product but did not go to the first secondary")
        else:
            print("Did not buy the first product")

        self.__users_data["client_" + str(client)] = {
            "group": group,
            "quantity_first": None,
            "quantity_second": None,
        }

    def send_data_to_learner(self):
        pass

    def greedy_simulation(self):
        greedy_environment = GreedyEnvironment
        simulator = Simulator(greedy_environment)
        max_cumulative_expected_margin = 0
        T = 365

        for t in range(T):
            if t == 0:                                                #first iteration just compute the margin of configuration [0,0,0,0,0]
                change, best_conf, max_cumulative_expected_margin = simulator.sim_one_day_greedy(True, 0)
                greedy_environment.round(greedy_environment.Learners)
            else:
                change, best_conf, max_cumulative_expected_margin = simulator.sim_one_day_greedy(False, max_cumulative_expected_margin)
                if change == False:
                    return                                  #if there isn't a new best configuration, return
                else:
                    greedy_environment.round(greedy_environment.configurations[best_conf])  #else the environment will compute the 5 new configuratoion

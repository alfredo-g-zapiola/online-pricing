import numpy as np
from Social_Influence import Social_Influence


class Simulator(object):
    def __init__(self, seed=41703192):
        self.seed = seed
        self.__SocialInfluence = None
        self.prices = {
            "product_1": 19.99,
            "product_2": 16.7,
            "product_3": 12.15,
            "product_4": 8.,
            "product_5": 35.,
        }
        # lambda to go to second secondary product
        self.__lambda = 0.5
        # daily data
        self.__daily_data = dict()
        self.__users_data = dict()

    def __init_R(self):
        pass

    def sim_one_day(self):
        # how many new users of each group arriving today
        ng1, ng2, ng3 = self.__sample_n_users()
        # total potential clients today
        n_tot = ng1 + ng2 + ng3
        # TODO define well social influence build matrix HERE
        self.__SocialInfluence = Social_Influence([ng1, ng2, ng3])

        direct_clients = {
            "group_1": np.random.choice(range(ng1), size=np.random.uniform(0, ng1)),  # TODO not uniform
            "group_2": np.random.choice(range(ng1, ng2), size=np.random.uniform(0, ng2)),
            "group_3": np.random.choice(range(ng2, ng3), size=np.random.uniform(0, ng3))
        }
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

    def sim_one_user(self, group, client, first_product):
        """
        :param group:
        :param client:
        :return:
        """
        def sim_buy(group, prod_id):
            willing_price = self.__sample_demand_curve(group, prod_id=prod_id)
            bought = False
            if self.__prices_and_margins["product_{}".format(first_product)] > willing_price:
                # TODO update dictionary
                bought = False
            else:
                # TODO
                bought = True
            return bought

        bought_first = sim_buy(group, first_product)
        if bought_first:
            first_secondary = None  # TODO sample first secondary
            goes_to_first_second = True  # TODO flip coin and compare with probability of affinity
            if goes_to_first_second:
                bought_second = sim_buy(group, first_secondary)
                goes_to_second_second = self.__lambda * 0 # TODO sample
                second_secondary = 3  # TODO sample
                if goes_to_second_second:
                    bought_third = sim_buy(group, second_secondary)
                else:
                    print("Bought the first product, the first secondary but not the second secondary")
            else:
                print("Bought the first product but did not go to the first secondary")
        else:
            print("Did not buy the first product")

        self.__users_data["client_" + str(client)] = {
            "group": group,
            "quantity_first": None,
            "quantity_second": None,
        }



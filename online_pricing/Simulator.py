import numpy as np


class Simulator(object):
    def __init__(self, seed=41703192):
        # we set a maximum total population so we can have a fixed influence graph
        # for the number of daily potential
        self.__population_max = 200
        self.seed = seed
        self.n_groups = 3
        # TODO big dictionaries here, move it to a json file
        self.__prices_and_margins = {
            "product_1": {
                25: 2,
                12: 4,
                14: 5,
                15: 2
            },
            "product_2": {
                25: 2,
                12: 4,
                14: 5,
                15: 2
            },
            "product_3": {
                25: 2,
                12: 4,
                14: 5,
                15: 2
            },
            "product_4": {
                25: 2,
                12: 4,
                14: 5,
                15: 2
            },
        }
        self.prices = {
            "product_1": 19.99,
            "product_2": 16.7,
            "product_3": 12.15,
            "product_4": 8.,
            "product_5": 35.,
        }

        # function parameters (can also be opened with a json)
        self.__distributions_parameters = {
            "n_people_params": {
                "group_0": 50,
                "group_1": 20,
                "group_2": 70
            },
            "dirichlet_params": {  # TODO dobbiamo giustificare le scelte qui
                "group 0": (7.65579946, 10.28353546,  5.16981654,  9.36425095,  9.26960117),
                "group 1": (14.54449788,  6.60476974, 11.29606424,  6.1703656,  8.9336728),
                "group 3": (12.89094056, 11.09866667,  9.96773461,  9.15999453,  7.7894984)
            },
            # for the quantity chosen daily we have a ... distribution
            "quantity_demanded_params": {
                "group 0": {},
                "group 1": {},
                "group 3": {}
            },
            # product graph probabilities
            # A Wishart distribution is assumed for the product graph probabilities
            "product_graph_params": {
                "group 0": {
                    "nu": 10,  # higher degrees of freedom, ...
                    "matrix": np.array(
                        [
                           [None, 0.1, .2, .15, .76],
                           [.2, None, .1, .15, .91]

                        ])
                },
                "group 1": {
                    "nu": 9,  # higher degrees of freedom, ...
                    "matrix": np.array(
                        [
                            [None, 0.1, .2, .15, .76],
                            [.2, None, .1, .15, .91]

                        ])
                },
                "group 2": {
                    "nu": 9,  # higher degrees of freedom, ...
                    "matrix": np.array(
                        [
                            [None, 0.1, .2, .15, .76],
                            [.2, None, .1, .15, .91]

                        ])
                }
            }
            # N.B. client graph probabilities are included in the Social Influece class
        }

        # function covariance parameters
        # higher alpha, higher ...
        self.__alpha_demand = 0
        # higher beta, higher ...
        self.__beta_demand = 0

        # initialise R session
        self.__init_R()

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
        # TODO social influence build matrix HERE
        daily_clients = {
            "group_1": np.random.choice(range(ng1), size=np.random.uniform(0, ng1)),  # TODO not uniform
            "group_2": np.random.choice(range(ng1, ng2), size=np.random.uniform(0, ng2)),
            "group_3": np.random.choice(range(ng2, ng3), size=np.random.uniform(0, ng3))
        }
        # TODO: social influence send seeds? HERE
        self.__users_data = dict()
        for g in range(3):  # for each group
            for client, site in daily_clients["group_" + str(g)]:  # for each webpage
                self.sim_one_user(group=g, client=client, first_product=site)
        # We now see how these customers have influenced their contacts and simulate what happens to those
        # they brought to our product websites
        influenced_clients = [[]]  # TODO social influence send and start here
        for client, site, g in influenced_clients:
            self.sim_one_user(group=g, client=client, first_product=site)

    def sim_one_user(self, group, client, first_product):
        """
        :param group:
        :param client:
        :return:
        """
        willing_price = self.__sample_demand_curve(group, prod_id=first_product)
        if self.__prices_and_margins["product_{}".format(first_product)] > willing_price:
            print("Client not buying anything")
        else:
            print("Price is OK for client")
            # quantity bought
            # GOTO secondary product
            goes_to_scnd, scnd_id = self.__sample_affinity(prod_id=first_product, group=group, first=True)
            if  goes_to_scnd:
                if self.__sample_demand_curve(group=group)



        self.__users_data["client_" + str(client)] = {
            "group": group,
            "quantity_first": None,
            "quantity_second": None,
        }

    def __sample_demand_curve(self, group, prod_id):
        pass  # TODO

    def __sample_n_users(self):
        """

        :return:
        """
        return [np.random.poisson(self.__distributions_parameters["n_people_params"]["group_"+str(i)], 1)
                for i in range(3)]

    def __sample_affinity(self, prod_id, group, first=True):
        return np.random.uniform(0, 1)

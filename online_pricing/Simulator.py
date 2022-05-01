import numpy as np
from Social_Influence import *

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
                "group_0": None,
                "group_1": None,
                "group_2": None
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

    def __init_R(self):
        pass

    def sim_one_day(self):
        #invocazione class social influence per avere i customer iniziali
        current_customer = current_customer + np.random.binomial(1, 0.1, size=1)
        # operazioni modifica initial prob matrix
        history=simulate_episode(init_prob_matrix, n_steps_max)

        pass

    def sim_one_user(self):
        pass

    def __sample_demand_curve(self, group):
        pass

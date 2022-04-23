import numpy as np


class Simulator(object):
    def __init__(self, seed=41703192):
        self.n_groups = 3
        self.prices = {}
        self.seed = seed

        # dirichlet
        self.__dirichlet_params = tuple()

        # function parameters (can also be opened with a json)
        self.__distributions_parameters = {
            "n_people_params": {

            },
            "alpha_ratios_params": {
                "group 0": {},
                "group 1": {},
                "group 3": {}

            },
            # for the quantity chosen daily we have a ... distribution
            "quantity_demanded_params": {
                "group 0": {},
                "group 1": {},
                "group 3": {}
            },
            # product graph probabilities
            # client graph probabilities are included in the Social Influece class

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
        pass

    def sim_one_user(self):
        pass

    def __sample_demand_curve(self, group):
        pass

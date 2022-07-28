from abc import abstractmethod

import numpy as np

# from scipy.stats import wishart
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

from online_pricing.learner import GreedyLearner


class EnvironmentBase:
    def __init__(self) -> None:
        self.n_products = 5
        self.n_groups = 3
        self.prices_and_margins = {
            "echo_dot": {34: 0.05, 12: 4, 14: 5, 15: 2},  # TODO look at plot
            "ring_chime": {36: 2, 12: 4, 14: 5, 15: 2},
            "product_3": {25: 2, 12: 4, 14: 5, 15: 2},
            "product_4": {25: 2, 12: 4, 14: 5, 15: 2},
        }

        # function parameters (can also be opened with a json)
        self.distributions_parameters = {
            "n_people_params": {"group_0": 50, "group_1": 20, "group_2": 70},
            "dirichlet_params": {  # TODO dobbiamo giustificare le scelte qui
                "group 0": (7.65579946, 10.28353546, 5.16981654, 9.36425095, 9.26960117),
                "group 1": (14.54449788, 6.60476974, 11.29606424, 6.1703656, 8.9336728),
                "group 3": (12.89094056, 11.09866667, 9.96773461, 9.15999453, 7.7894984),
            },
            # for the quantity chosen daily we have a ... distribution
            "quantity_demanded_params": {"group 0": 1, "group 1": 2, "group 3": 3},
            # product graph probabilities
            "product_graph": np.array(
                [
                    [0.0, 0.1, 0.2, 0.2, 0.0],
                    [0.1, 0.0, 0.2, 0.3, 0.0],
                    [0.2, 0.2, 0.0, 0.1, 0.0],
                    [0.2, 0.3, 0.1, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
            # A Wishart distribution is assumed for the product graph probabilities
            "product_graph_params": {
                "group 0": {
                    "nu": 10,  # higher degrees of freedom, ...
                    "matrix": np.array(
                        [[None, 0.1, 0.2, 0.15, 0.76], [0.2, None, 0.1, 0.15, 0.91]]
                    ),
                },
                "group 1": {
                    "nu": 9,  # higher degrees of freedom, ...
                    "matrix": np.array(
                        [[None, 0.1, 0.2, 0.15, 0.76], [0.2, None, 0.1, 0.15, 0.91]]
                    ),
                },
                "group 2": {
                    "nu": 9,  # higher degrees of freedom, ...
                    "matrix": np.array(
                        [[None, 0.1, 0.2, 0.15, 0.76], [0.2, None, 0.1, 0.15, 0.91]]
                    ),
                },
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

        # have the number of users
        self.__n_users = (10, 20, 5)

    def __init_R(self):
        """
        start R and download the roahd package, define the functions of the demand curves
        :return: void
        """
        # Install the roahd package
        utils = importr('utils')
        utils.chooseCRANmirror(ind=1)  # select the first mirror in the list
        utils.install_packages("roahd")
        with open('online_pricing/initialise_R.R', 'r') as file:
            code = file.read().rstrip()
            robjects.r(code)



    def sample_n_users(self):
        """
        Samples from the poisson distribution how many new potential clients of each group arrive on the current day

        :return: a list with the number of potential clients of each group
        """
        self.__n_users = [
            np.random.poisson(
                self.distributions_parameters["n_people_params"]["group_" + str(i)], 1
            )
            for i in range(3)
        ]
        return self.__n_users

    def sample_affinity(self, prod_id, group, first=True):
        return np.random.uniform(0, 1)

    def sample_demand_curve(self, group, prod_id, price, uncertain=False):
        """

        :param group: the id of the group
        :param prod_id: the id of the product
        :param price: the price at which we want to sample
        :return: a price the client is willing to pay
        """
        def get_fname(prod_id, group):

            fname = str()
            prodname = str()

            # python 3.10 for match
            match prod_id:
                case 0:
                    prodname = "echo_dot"
                case 1:
                    prodname = "ring_chime"
                case 2:
                    prodname = "ring_f"
                case 3:
                    prodname = "ring_v"
                case 4:
                    prodname = "echo_show"
                case _:
                    print("This should never happen")

            match group:
                case 0:
                    fname = prodname + "_poor"
                case 1:
                    fname = prodname
                case 2:
                    fname = prodname + "_rich"
                case _:
                    print("This should never happen")
            return fname
        fname = get_fname(prod_id, group)
        print(fname)
        if uncertain:
            d = robjects.r("""
                d <- sample.demand({}, {}, 0, 200 )
            """.format(fname, price))
            return robjects.r("d")[0]
        else:
            curve_f = robjects.r["{}".format(fname)]
            return curve_f(price)[0]



    def get_direct_clients(self) -> dict[str, list[tuple[int, int]]]:
        """
        Get all direct clients, for each group, with their respective primary product.

        This function return a dictionary with an entry for each group. For each entry, a list of
        tuples that represent (client_id, primary_product_id).

        :return: the direct clients with their primary product.
        """
        cumsum_clients = [0, *np.cumsum(self.__n_users)]
        # TODO: not uniform
        n_direct_clients = [int(np.random.uniform(0, n_group)) for n_group in self.__n_users]

        direct_clients = {
            f"group_{idx}": list(
                zip(
                    np.random.choice(
                        range(cumsum_clients[idx], cumsum_clients[idx + 1]),
                        size=n_direct_clients[idx],
                        replace=False,
                    ),
                    np.random.choice(range(self.n_products), n_direct_clients[idx]),
                )
            )
            for idx, n_group in enumerate(self.__n_users)
        }
        return direct_clients

    def sample_quantity_bought(self, group, uncertain=False):
        """

        :param group:
        :return:
        """
        m = self.distributions_parameters["quantity_demanded_params"]["group_" + str(group)]
        if uncertain:
            return np.random.poisson(m)
        else:
            return m


class GreedyEnvironment(EnvironmentBase):
    def __init__(self) -> None:
        super().__init__()
        # create the list of learners, one for each product
        self.learners = [
            GreedyLearner(
                n_arms=4, prices=list(self.prices_and_margins["product_" + str(i)].values())
            )
            for i in range(self.n_products)
        ]  # first configuration [0, 0, 0, 0, 0]

    def round(self, best_update: int, reward: int):

        self.learners[best_update].update(self.learners[best_update].act(), reward)


class EnvironmentStep4(EnvironmentBase):
    def sample_alpha_ratios(self):
        """
        Override this method since alpha ratios become uncertain: we thus instead of just dint hte expected value to the
        simulator we sample from the dirichlet the distribution.

        :return: the realisations of the alpha ratios (dirichlet distribution9
        """
        # todo implement

    def sample_items_sold(self):
        """
        The quantity of items sold becomes uncertain so now we sample instead of just returning the
        expected value.
        :return: realisations of the
        """
        # todo implement

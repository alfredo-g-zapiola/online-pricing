import itertools
from typing import Any, Optional, cast

import numpy as np
import numpy.typing as npt

# from scipy.stats import wishart # for step 5: uncertain graph weights
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

from online_pricing.influence_function import InfluenceFunctor
from online_pricing.Wishart import WishartHandler


class EnvironmentBase:
    def __init__(self, n_products: int = 5, n_groups: int = 3, hyperparameters: dict[str, Any] | None = None) -> None:

        if hyperparameters is None:
            hyperparameters = dict()

        self.n_products = n_products
        self.n_groups = n_groups

        # Hyperparameters
        self.fully_connected = hyperparameters.get("fully_connected", True)
        self.context_generation = hyperparameters.get("context_generation", False)
        self.uncertain_alpha = hyperparameters.get("uncertain_alpha", False)
        self.group_unknown = hyperparameters.get("group_unknown", True)
        self._lambda = hyperparameters.get("lambda", 0.5)
        self.uncertain_demand_curve = hyperparameters.get("uncertain_demand_curve", False)
        self.uncertain_quantity_bought = hyperparameters.get("uncertain_quantity_bought", False)
        self.uncertain_graph_weights = hyperparameters.get("uncertain_graph_weights", False)
        self.wishart_df = hyperparameters.get(
            "wishart_df", 20
        )  # higher df, less uncertainty. Cannot be lower than n_products
        self.shifting_demand_curve = hyperparameters.get("shifting_demand_curve", False)

        """
        Prices and margins: taken from the demand_curves.R we have the prices.
        We assume the cost to be the 40% of the standard price, so that when there is 40% discount,
        we break even (it would hardly make sense otherwise)
        Note the margin decreases linearly with the price
        """
        self.prices_and_margins: dict[str, list[tuple[float, float]]] = {
            "echo_dot": [(13, 0), (27, 14), (32, 19), (34, 21)],
            "ring_chime": [
                (14.4, 0),
                (28.8, 14.4),
                (34.2, 19.8),
                (36, 21.6),
            ],
            "ring_f": [(80, 0), (160, 80), (190, 110), (200, 120)],
            "ring_v": [(24, 0), (48, 24), (57, 23), (60, 26)],
            "echo_show": [
                (38.4, 0),
                (76.8, 38.4),
                (91.2, 52.8),
                (96, 57.6),
            ],
        }
        self.n_prices = len(self.prices_and_margins[list(self.prices_and_margins.keys())[0]])
        self.product_id_map = {0: "echo_dot", 1: "ring_chime", 2: "ring_f", 3: "ring_v", 4: "echo_show"}

        # function parameters (can also be  opened with a json)
        self.distributions_parameters: dict[str, Any] = {
            "n_people_params": [70, 50, 20],  # we have more poor people than rich people
            "dirichlet_params": [  # alpha ratios
                np.asarray([15, 10, 6, 5, 4, 6]),
                np.asarray([12, 9, 6, 4, 3, 4]),
                np.asarray([5, 5, 9, 7, 7, 8]),
            ],
            # for the quantity chosen daily we have a ... distribution
            "quantity_demanded_params": [1, 1.2, 1.8],
            # product graph probabilities, see the other file
            "product_graph": [
                self.product_matrix(
                    size=n_products, fully_connected=self.fully_connected, unif_params=(0.2 + g * 0.1, 0.8 + g * 0.1)
                )
                for g in range(self.n_groups)
            ],  # higher the groud id, richer it is, higher edge probas
            # end of product_graph matrices list
            # N.B. client graph probabilities are included in the Social Influece class
        }
        self.mean_product_graph = None
        self._influence_functor = InfluenceFunctor(self.yield_first_secondaries(), self._lambda)

        # TODO: use this
        # function covariance parameters
        # higher alpha, higher ...
        # self.__alpha_demand = 0
        # higher beta, higher ...
        # self.__beta_demand = 0

        # initialise R session
        self._init_r()

        # value the objective function
        self.rewards = dict()
        self.clairvoyant = {}

    @staticmethod
    def _init_r() -> None:
        """
        start R and download the roahd package, define the functions of the demand curves
        :return: void
        """
        # Install the roahd package
        utils = importr("utils")
        utils.chooseCRANmirror(ind=1)  # select the first mirror in the list
        utils.install_packages("roahd")
        with open("online_pricing/initialise_R.R", "r") as file:
            code = file.read().rstrip()
            robjects.r(code)

    def sample_n_users(self) -> tuple[int, ...]:
        """
        Samples from the poisson distribution how many new potential clients of each group arrive on the current day

        :return: a list with the number of potential clients of each group
        """
        n_users = tuple(
            int(np.random.poisson(self.distributions_parameters["n_people_params"][i], 1)) for i in range(self.n_groups)
        )
        return n_users

    # TODO: use this
    # def sample_affinity(self, prod_id, group, first=True):
    #     return np.random.uniform(0, 1)

    def sample_demand_curve(self, group: int, prod_id: int, price: float) -> float:
        """

        :param group: the id of the group
        :param prod_id: the id of the product
        :param price: the price at which we want to sample
        :return: a price the client is willing to pay
        """

        # python 3.10 for match
        match prod_id:
            case 0:
                prod_name = "echo_dot"
            case 1:
                prod_name = "ring_chime"
            case 2:
                prod_name = "ring_f"
            case 3:
                prod_name = "ring_v"
            case 4:
                prod_name = "echo_show"
            case _:
                raise ValueError("Invalid product id")

        match group:
            case 0:
                f_name = prod_name + "_poor"
            case 1:
                f_name = prod_name
            case 2:
                f_name = prod_name + "_rich"
            case _:
                raise ValueError("Invalid group id")

        if self.uncertain_demand_curve:
            robjects.r(
                """
                d <- sample.demand({}, {}, 0, 200 )
            """.format(
                    f_name, price
                )
            )
            return float(robjects.r("d")[0])
        else:
            curve_f = robjects.r["{}".format(f_name)]
            clipper = robjects.r["clipper.f"]
            return float(clipper(curve_f(price)[0])[0])

    def get_direct_clients(self) -> dict[str, list[tuple[int, int]]]:
        """
        Get all direct clients, for each group, with their respective primary product.

        This function return a dictionary with an entry for each group. For each entry, a list of
        tuples that represent (client_id, primary_product_id).

        :param: uncertain_alpha. False if we take the expected value of the alpha_ratios (mean of
        dirichlet distribution),
        True to sample from the dirichlet distribution

        :return: -> Return this { "group_id": [primary_product_id, ...],  }
             -> primary_product_id in {-1, 0, .., n_products-1}
        """
        n_user = self.sample_n_users()
        cumsum_clients = [0, *np.cumsum(n_user)]
        if self.uncertain_alpha:
            dirichlet_sample = [
                np.random.dirichlet(self.distributions_parameters["dirichlet_params"][g]) for g in range(self.n_groups)
            ]
        else:  # compute expected value
            # we just take each element of the vector and divide it by the sum to get the expected value
            dirichlet_sample = [
                self.distributions_parameters["dirichlet_params"][g]
                / self.distributions_parameters["dirichlet_params"][g].sum()
                for g in range(self.n_groups)
            ]

        # take out clients going away (i.e. alpha_0)
        n_direct_clients = [int(n_user[g] * (1 - dirichlet_sample[g][0])) for g in range(self.n_groups)]

        direct_clients = {
            f"group_{idx}": list(
                zip(
                    np.random.choice(
                        range(cumsum_clients[idx], cumsum_clients[idx + 1]),
                        size=n_direct_clients[idx],
                        replace=False,
                    ),
                    np.random.choice(
                        list(range(self.n_products)),
                        n_direct_clients[idx],
                        # take out element 0, re-normalise
                        p=dirichlet_sample[idx][1:] / dirichlet_sample[idx][1:].sum(),
                    ),
                )
            )
            for idx, n_group in enumerate(n_user)
        }
        return direct_clients

    def sample_quantity_bought(self, group: int) -> int:
        """
        Sample the quantity of products bought by a client of a given group

        :param group: the id of the group
        :return: the quantity of products bought by a client of a given group
        """
        m: int = self.distributions_parameters["quantity_demanded_params"][group]
        if self.uncertain_quantity_bought:
            return int(np.random.poisson(m))

        return m

    def yield_first_secondaries(self) -> list[npt.NDArray[int]]:
        """
        Sends to the simulator the two best products to be the secondaries.

        :return: A list of n_products where for each product we have the two secondaries
        """
        # first we have a weighed mean of the means of the product graphs
        weighted_mean_p_graph = np.zeros((self.n_products, self.n_products))
        for g in range(self.n_groups):
            weighted_mean_p_graph += (
                self.distributions_parameters["product_graph"][g].mean * self.distributions_parameters["n_people_params"][g]
            )

        weighted_mean_p_graph /= sum([self.distributions_parameters["n_people_params"][g] for g in range(self.n_groups)])
        self.mean_product_graph = weighted_mean_p_graph
        return [np.flip(np.argsort(weighted_mean_p_graph[i]))[:2].astype(int, copy=False) for i in range(self.n_products)]

    def yield_expected_alpha(self) -> list[float] | list[list[float]]:
        """
        It is assumed the simulator knows the expected values of the alpha ratios.

        :return: an array of the expected values of the five alpha ratios. If we assume a unique group,
        the weighted mean (according to the mean daily customers) is obtained
        """
        if not self.context_generation:
            # return the weighted mean (according to the number of people in the group) of the alpha ratios
            return cast(
                list[float],
                sum(
                    np.asarray(
                        [
                            self.distributions_parameters["n_people_params"][i]
                            * self.distributions_parameters["dirichlet_params"][i]
                            / self.distributions_parameters["dirichlet_params"][i].sum()
                            for i in range(self.n_groups)
                        ]
                    )
                )
                / sum([self.distributions_parameters["n_people_params"][i] for i in range(self.n_groups)]),
            )

        return [alphae / alphae.sum() for alphae in self.distributions_parameters["dirichlet_params"]]

    def product_matrix(self, size: int, fully_connected: bool = True, unif_params=(0.1, 1)) -> WishartHandler:
        """
        Generate random product matrix.
        Note the higher the group number, the richer the population, and hence higher the edges.

        We assume the product matrix to be a wishart distribution.
        This is the distribution of a covariance matrix, which can easily be transformed into a correlation
        matrix, so that all its elements are <= 1
        The diagonal elements (which are all 1) will then be set to 0 as there is no influence of a product to
        itself.
        Cfr. https://stats.stackexchange.com/questions/22050/generating-correlation-matrices-using-wishart-distribution


        :param size: size of the matrix, i.e. (n_products, n_products)
        :param fully_connected: if True, the matrix is fully connected, i.e. all products are connected to all other products
        :param group: bigger the group id, the richer the population, the higher in mean the product graph
        :param unif_params
        :return: a random product matrix
        """

        return WishartHandler(
            size=size,
            df=self.wishart_df,
            unif_params=unif_params,
            uncertain=self.uncertain_graph_weights,
            fully_connected=fully_connected,
            seed=2200337,
        )

    def compute_clairvoyant(self):
        """
        For every price combination (so it is a carthesian product of the possible prices with itself)
        , obtain the expected mean margin.
        I.e. we compute a grid search working with expected values
        Note it was tested:
            if the margins are all 1, and there is 1 user per group then (0,0,0,0,0) is the best one (we maximise influence function),
            and moreover its clairvoyant is below 1 (probability measure)
        :return:
        """
        push_uncertain = self.uncertain_demand_curve  # save config
        self.uncertain_demand_curve = False  # so we take the mean value
        push_alpha_context = self.context_generation
        self.context_generation = True
        expected_alpha_r = cast(list[list[float]], self.yield_expected_alpha())

        rewards = {}
        maximum = 0.0
        max_arm = ""

        # explore the carthesian product of the possible prices (5 values) with itself 5 times
        for price_config in itertools.product(list(range(self.n_prices)), repeat=self.n_products):
            print("Current price config: ", str(price_config))
            cur_reward = 0.0
            price_and_margin = lambda p: self.prices_and_margins[self.product_id_map[p]][price_config[p]]

            def c_rate(product: int) -> float:
                """Compute the conversion rate fixed with fixed group and prices.
                This is why the function is redefined daily"""
                return self.sample_demand_curve(group=g, prod_id=product, price=price_and_margin(product)[0])

            for g in range(self.n_groups):  # for each group
                quantity = self.distributions_parameters["quantity_demanded_params"][g]
                influence_function = np.zeros(shape=(self.n_products, self.n_products))

                # compute the influence function values for this group at this price
                for p1 in range(self.n_products):
                    for p2 in [i for i in range(self.n_products) if i != p1]:
                        influence_function[p1, p2] = self._influence_functor(
                            p1, p2, c_rate, self.distributions_parameters["product_graph"][g].mean
                        )

                g_reward = sum(
                    [
                        expected_alpha_r[g][product_id + 1]
                        * (  # plus one since 0 corresponds to leaving the website
                            c_rate(product_id) * price_and_margin(product_id)[1] * quantity
                            + sum(
                                [
                                    influence_function[product_id, secondary_product]
                                    * c_rate(secondary_product)
                                    * price_and_margin(secondary_product)[1]
                                    * quantity
                                    for secondary_product in range(self.n_products)
                                    if secondary_product != product_id
                                ]
                            )
                        )
                        for product_id in range(self.n_products)
                    ]
                )

                cur_reward += (
                    g_reward * self.distributions_parameters["n_people_params"][g]
                )  # weight it according to number of people
            cur_reward /= sum(
                [self.distributions_parameters["n_people_params"][g] for g in range(self.n_groups)]
            )  # normalise

            rewards[str(price_config)] = cur_reward
            if cur_reward > maximum:
                maximum = cur_reward
                max_arm = str(price_config)

        # reset data members
        self.uncertain_demand_curve = push_uncertain
        self.context_generation = push_alpha_context

        self.rewards = rewards
        self.clairvoyant = {max_arm: maximum}

        return rewards, max_arm

    def yield_clairvoyant(self):
        """
        Clairvoyant is hard-coded since it takes 30 mins to compute it
        :return:
        """
        if not self.shifting_demand_curve:
            return 19.08163728705  # with arm (1,1,3,1,1)
        else:
            return (19.08163728705, None, None)
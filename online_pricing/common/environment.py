import itertools
from typing import Any

import numpy as np
import numpy.typing as npt
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

from online_pricing.common.influence_function import InfluenceFunctor
from online_pricing.common.learner import TSLearner
from online_pricing.helpers.utils import flatten, suppress_output
from online_pricing.helpers.wishart import WishartHandler
from online_pricing.models.user import User


class EnvironmentBase:
    def __init__(self, n_products: int = 5, n_groups: int = 3, hyperparameters: dict[str, Any] | None = None) -> None:

        self._init_r()

        if hyperparameters is None:
            hyperparameters = dict()

        self.n_products = n_products
        self.n_groups = n_groups

        # Hyperparameters
        self.fully_connected = hyperparameters.get("fully_connected", True)
        self.learner_class = hyperparameters.get("learner_class", TSLearner)
        self.context_generation = hyperparameters.get("context_generation", False)
        self.uncertain_alpha = hyperparameters.get("uncertain_alpha", False)
        self.group_unknown = hyperparameters.get("group_unknown", True)
        self._lambda: float = hyperparameters.get("lambda", 0.5)
        self.uncertain_demand_curve = hyperparameters.get("uncertain_demand_curve", False)
        self.uncertain_quantity_bought = hyperparameters.get("uncertain_quantity_bought", False)
        self.uncertain_graph_weights = hyperparameters.get("uncertain_graph_weights", False)
        self.wishart_df = hyperparameters.get("wishart_df", 20)
        self.shifting_demand_curve = hyperparameters.get("shifting_demand_curve", False)
        self.unknown_demand_curve = hyperparameters.get("unknown_demand_curve", True)
        self.unknown_quantity_bought = hyperparameters.get("unknown_quantity_bought", True)
        self.unknown_product_weights = hyperparameters.get("unknown_product_weights", True)

        # Prices and margins: taken from the demand_curves.R we have the prices.
        # We assume the cost to be the 40% of the standard price, so that when there is 40% discount,
        # we break even (it would hardly make sense otherwise)
        # Note the margin decreases linearly with the price
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

        self.distributions_parameters: dict[str, Any] = {
            "n_people_params": [300, 450, 700],
            "dirichlet_params": [
                np.asarray([15, 10, 6, 5, 4, 6]),
                np.asarray([12, 9, 6, 4, 3, 4]),
                np.asarray([5, 5, 9, 7, 7, 8]),
            ],
            "quantity_demanded_params": [1, 1.2, 1.8],
            "product_graph": [
                self.product_matrix(
                    size=n_products, fully_connected=self.fully_connected, unif_params=(0.2 + g * 0.1, 0.8 + g * 0.1)
                )
                for g in range(self.n_groups)
            ],
        }
        self.mean_product_graph = list[list[float]]()
        self.influence_functor = InfluenceFunctor(self.yield_first_secondaries(), self._lambda)

        # initialise R session
        self.group_proportions = list(range(self.n_groups))
        for g in range(self.n_groups):
            self.group_proportions[g] = self.distributions_parameters["n_people_params"][g] / sum(
                self.distributions_parameters["n_people_params"]
            )

        # value the objective function
        self.rewards = dict[str, float]()
        self.clairvoyant = dict[str, float]()
        self.expected_demand_curve: list[list[list[float]]] = list()
        self.compute_expected_demand_curve()

    def get_lambda(self) -> float:
        """
        Get the lambda parameter.

        :return: lambda parameter
        """
        return self._lambda

    @staticmethod
    @suppress_output
    def _init_r() -> None:
        """Initialise the R session."""

        # Install the roahd package
        utils = importr("utils")
        utils.chooseCRANmirror(ind=1)
        utils.install_packages("roahd")
        with open("online_pricing/r/initialise_R.R", "r") as file:
            code = file.read().rstrip()
            robjects.r(code)

    def sample_n_users(self) -> tuple[int, ...]:
        """
        Sample the number of users for each group.

        Samples from the poisson distribution how many new potential clients of each group arrive at the current day
        :return: a list with the number of potential clients of each group
        """
        n_users = tuple(
            int(np.random.poisson(self.distributions_parameters["n_people_params"][i], 1)) for i in range(self.n_groups)
        )
        return n_users

    def sample_demand_curve(self, group: int, prod_id: int, price: float, n_day: int = 0) -> float:
        """
        Samples the demand curve for a given group, product and price. The day is used when the demand curve is shifting

        :param group: the id of the group
        :param prod_id: the id of the product
        :param price: the price at which we want to sample
        :param n_day: the day of the simulation
        :return: the probability of buying the product at the given price
        """

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
                f_name = prod_name + "_casual"
            case 1:
                f_name = prod_name + "_adjust"
            case 2:
                f_name = prod_name + "_organic"
            case _:
                raise ValueError("Invalid group id")

        def _apply_shift(c_rate: float) -> float:
            if not self.shifting_demand_curve:
                return c_rate
            else:
                if 15 < n_day <= 30:
                    c_rate = np.clip(c_rate * 1.5, 0, 1)
                elif n_day > 30:
                    c_rate = np.clip(c_rate * 0.5, 0, 1)
                return c_rate

        if self.uncertain_demand_curve:
            robjects.r("d <- sample.demand({}, {}, 0, 200 )".format(f_name, price))
            return _apply_shift((robjects.r("d")[0]))
        else:
            curve_f = robjects.r["{}".format(f_name)]
            clipper = robjects.r["clipper.f"]
            return _apply_shift(float(clipper(curve_f(price)[0])[0]))

    def get_direct_clients(self) -> list[User]:
        """
        Get all direct clients, for each group, with their respective primary product.

        This function return a dictionary with an entry for each group. For each entry, a list of
        tuples that represent (client_id, primary_product_id).

        :return:the direct clients for each group
        """
        n_user = self.sample_n_users()
        if self.uncertain_alpha:
            dirichlet_sample = [
                np.random.dirichlet(self.distributions_parameters["dirichlet_params"][g]) for g in range(self.n_groups)
            ]
        else:

            dirichlet_sample = [
                self.distributions_parameters["dirichlet_params"][g]
                / self.distributions_parameters["dirichlet_params"][g].sum()
                for g in range(self.n_groups)
            ]

        # take out clients going away (i.e. alpha_0)
        n_direct_clients = [int(n_user[g] * (1 - dirichlet_sample[g][0])) for g in range(self.n_groups)]
        products = list(range(self.n_products))
        direct_clients = [
            [
                User(
                    group=group,
                    landing_product=np.random.choice(
                        products,
                        p=dirichlet_sample[group][1:] / dirichlet_sample[group][1:].sum(),
                    ),
                )
                for _ in range(n_direct_clients[group])
            ]
            for group in range(self.n_groups)
        ]

        return flatten(direct_clients)

    def sample_quantity_bought(self, group: int) -> int:
        """
        Sample the quantity of products bought by a client of a given group

        :param group: the id of the group
        :return: the quantity of products bought by a client of a given group
        """
        m: int = self.distributions_parameters["quantity_demanded_params"][group]
        if self.uncertain_quantity_bought:
            return np.random.poisson(m) * 2

        return m * 2

    def yield_first_secondaries(self) -> list[list[int]]:
        """
        Sends to the simulator the two best products to be the secondaries.

        :return: A list of n_products where for each product we have the two secondaries
        """
        weighted_mean_p_graph: npt.NDArray[np.float32] = np.zeros((self.n_products, self.n_products), dtype=np.float32)
        for g in range(self.n_groups):
            weighted_mean_p_graph += (
                self.distributions_parameters["product_graph"][g].mean * self.distributions_parameters["n_people_params"][g]
            )

        weighted_mean_p_graph /= sum([self.distributions_parameters["n_people_params"][g] for g in range(self.n_groups)])
        self.mean_product_graph = weighted_mean_p_graph.tolist()
        return [
            np.flip(np.argsort(weighted_mean_p_graph[i]))[:2].astype(int, copy=False).tolist()
            for i in range(self.n_products)
        ]

    def yield_expected_alpha(self) -> list[list[float]]:
        """
        It is assumed the simulator knows the expected values of the alpha ratios.

        :return: an array of the expected values of the five alpha ratios. If we assume a unique group,
        the weighted mean (according to the mean daily customers) is obtained
        """
        if not self.context_generation:
            # return the weighted mean (according to the number of people in the group) of the alpha ratios
            return [
                sum(  # type: ignore [list-item]
                    np.asarray(
                        [
                            self.distributions_parameters["n_people_params"][i]
                            * self.distributions_parameters["dirichlet_params"][i]
                            / self.distributions_parameters["dirichlet_params"][i].sum()
                            for i in range(self.n_groups)
                        ],
                        dtype=np.float32,
                    ),
                )
                / sum([self.distributions_parameters["n_people_params"][i] for i in range(self.n_groups)])
            ] * self.n_groups

        return [alphae / alphae.sum() for alphae in self.distributions_parameters["dirichlet_params"]]

    def product_matrix(
        self, size: int, fully_connected: bool = True, unif_params: tuple[float, float] = (0.1, 1.0)
    ) -> WishartHandler:
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
        :param unif_params: the parameters of the uniform distribution used to generate the wishart distribution
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

    def compute_clairvoyant(self, n_day: int) -> tuple[dict[str, float], str]:
        """
        For every price combination (so it is a carthesian product of the possible prices with itself)
        , obtain the expected mean margin.
        I.e. we compute a grid search working with expected values
        Note:
        it was tested, if the margins are all 1, and there is 1 user per group then (0,0,0,0,0) is the best one
        (we maximise influence function), and moreover its clairvoyant is below 1 (probability measure)

        :return: the best price combination and its clairvoyant
        """

        rewards = {}
        maximum = 0.0
        max_arm = ""

        save_shift = self.shifting_demand_curve
        self.shifting_demand_curve = True
        uncertain_config = self.uncertain_demand_curve  # save config
        self.uncertain_demand_curve = False  # so we take the mean value
        alpha_context_config = self.context_generation
        self.context_generation = True
        expected_alpha_r = self.yield_expected_alpha()

        # explore the carthesian product of the possible prices (5 values) with itself 5 times
        for price_config in itertools.product(list(range(self.n_prices)), repeat=self.n_products):
            print("Price config: ", price_config)
            cur_reward = 0.0

            for g in range(self.n_groups):  # for each group
                quantity = self.distributions_parameters["quantity_demanded_params"][g]
                influence_function = np.zeros(shape=(self.n_products, self.n_products))

                # margin changes according to today's price config
                def price_and_margin(product: int) -> tuple[float, float]:
                    return self.prices_and_margins[self.product_id_map[product]][price_config[product]]

                # conversion rate changes according to group and selected price (given by price_config)
                def c_rate(product: int) -> float:
                    """Compute the conversion rate fixed with fixed group and prices.
                    This is why the function is redefined daily"""
                    return self.sample_demand_curve(
                        group=g, prod_id=product, price=price_and_margin(product)[0], n_day=n_day
                    )

                # compute the influence function values for this group at this price
                for p1 in range(self.n_products):
                    for p2 in [i for i in range(self.n_products) if i != p1]:
                        influence_function[p1, p2] = self.influence_functor(
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
                # weight it according to number of people
                # print(g_reward)
                cur_reward += g_reward * self.group_proportions[g]

            rewards[str(price_config)] = cur_reward
            if cur_reward > maximum:
                maximum = cur_reward
                max_arm = str(price_config)

        # reset data members
        self.uncertain_demand_curve = uncertain_config
        self.context_generation = alpha_context_config

        self.rewards = rewards
        self.clairvoyant = {max_arm: maximum}
        self.shifting_demand_curve = save_shift
        return rewards, max_arm

    def yield_clairvoyant(self, n_day: int) -> float:
        """
        Clairvoyant is hard-coded since it takes 30 mins to compute it
        :return:
        """
        if self.context_generation:
            computed_groups_clairvoyants = (2.833058712786084, 57.0369702202745, 298.7954434333168)

            return sum(computed_groups_clairvoyants[g] * self.group_proportions[g] for g in range(self.n_groups))
        else:
            # best arms: (2, 1, 3, 1, 2), (2, 2, 3, 1, 2),(2, 2, 3, 1, 2)
            computed_clairvoyants = (143.95453453, 287.49863215086856, 85.87646858816036)
            # computed_clairvoyants = (8.993181445527823, 13.29717560887341, 16.13083207278911)
            if not self.shifting_demand_curve:
                return computed_clairvoyants[0]  # with arm (1,1,3,1,1)
            else:

                if n_day <= 15:
                    return computed_clairvoyants[0]
                elif 15 < n_day <= 30:
                    return computed_clairvoyants[1]
                else:
                    return computed_clairvoyants[2]

    def compute_expected_demand_curve(self) -> None:
        # note we have to take the average conversion rate, so we weight the c_rate of each group
        # save old value
        save = self.uncertain_demand_curve
        self.uncertain_demand_curve = False  # since we need expected value
        save_2 = self.shifting_demand_curve
        self.shifting_demand_curve = True
        for t in [0, 20, 35]:  # three different demand curves
            current = np.zeros(shape=(self.n_products, 4))  # 4 different ṕrices
            for p in range(self.n_products):
                price_id = 0
                for price, margin in self.prices_and_margins[self.product_id_map[p]]:
                    current[p, price_id] = sum(
                        [
                            self.sample_demand_curve(group=g, prod_id=p, price=price, n_day=t) * self.group_proportions[g]
                            for g in range(self.n_groups)
                        ]
                    )
                    price_id += 1
            self.expected_demand_curve.append(current.astype(float).tolist())

        self.uncertain_demand_curve = save
        self.shifting_demand_curve = save_2

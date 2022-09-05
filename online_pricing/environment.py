from typing import Any, cast

import numpy as np
import numpy.typing as npt

# from scipy.stats import wishart # for step 5: uncertain graph weights
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

from online_pricing.influence_function import InfluenceFunctor


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
        self.group_unknown = hyperparameters.get("group_unknown", False)
        self._lambda = hyperparameters.get("lambda", 0.5)

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

        # function parameters (can also be  opened with a json)
        self.distributions_parameters: dict[str, Any] = {
            "n_people_params": [70, 50, 20],  # we have more poor people than rich people
            "dirichlet_params": [
                np.asarray([15, 10, 6, 5, 4, 6]),
                np.asarray([12, 9, 6, 4, 3, 4]),
                np.asarray([5, 5, 9, 7, 7, 8]),
            ],
            # for the quantity chosen daily we have a ... distribution
            "quantity_demanded_params": [1, 2, 3],
            # product graph probabilities
            "product_graph": [random_product_matrix(size=(n_products, n_products), fully_connected=self.fully_connected)]
            * n_groups,  # end of product_graph matrices list
            # A Wishart distribution is assumed for the product graph probabilities
            "product_graph_params": {
                "group 0": {
                    "nu": 10,  # higher degrees of freedom, ...
                    "matrix": np.array([[None, 0.1, 0.2, 0.15, 0.76], [0.2, None, 0.1, 0.15, 0.91]]),
                },
                "group 1": {
                    "nu": 9,  # higher degrees of freedom, ...
                    "matrix": np.array([[None, 0.1, 0.2, 0.15, 0.76], [0.2, None, 0.1, 0.15, 0.91]]),
                },
                "group 2": {
                    "nu": 9,  # higher degrees of freedom, ...
                    "matrix": np.array([[None, 0.1, 0.2, 0.15, 0.76], [0.2, None, 0.1, 0.15, 0.91]]),
                },
            }
            # N.B. client graph probabilities are included in the Social Influece class
        }

        self._influence_functor = InfluenceFunctor(self.yield_first_secondaries(), self._lambda)
        # TODO: use this
        # function covariance parameters
        # higher alpha, higher ...
        # self.__alpha_demand = 0
        # higher beta, higher ...
        # self.__beta_demand = 0

        # initialise R session
        self._init_r()

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

    @staticmethod
    def sample_demand_curve(group: int, prod_id: int, price: float, uncertain: bool = False) -> float:
        """

        :param group: the id of the group
        :param prod_id: the id of the product
        :param price: the price at which we want to sample
        :param uncertain: whether we want to sample from an uncertain demand curve
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

        if uncertain:
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
            return float(curve_f(price)[0])

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
        else:
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

    def sample_quantity_bought(self, group: int, uncertain: bool = False) -> int:
        """
        Sample the quantity of products bought by a client of a given group

        :param group: the id of the group
        :param uncertain: whether we want to sample from an uncertain demand curve
        :return: the quantity of products bought by a client of a given group
        """
        m: int = self.distributions_parameters["quantity_demanded_params"][group]
        if uncertain:
            return int(np.random.poisson(m))

        return m

    # static for now
    @staticmethod
    def compute_clairvoyant() -> float:
        """
        Finds the optimal super arm that maximises the utility gains
        :return: optimal super arm
        """
        clairvoyant = 100.0
        return clairvoyant

    def yield_first_secondaries(self) -> list[npt.NDArray[int]]:
        """
        Sends to the simulator the two best products to be the secondaries.

        :return: A list of n_products where for each product we have the two secondaries
        """
        if self.group_unknown:
            return [
                np.flip(np.argsort(self.distributions_parameters["product_graph"][0][i]))[:2] for i in range(self.n_products)
            ]

        return [
            np.flip(np.argsort(self.distributions_parameters["product_graph"][i][j]))[:2]
            for j in range(self.n_products)
            for i in range(self.n_groups)
        ]

    # TODO: fix this for context generation
    def yield_expected_alpha(self) -> list[float]:
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


class EnvironmentStep4(EnvironmentBase):
    def sample_alpha_ratios(self) -> None:
        """
        Override this method since alpha ratios become uncertain: we thus instead of just dint hte expected value to the
        simulator we sample from the dirichlet the distribution.

        :return: the realisations of the alpha ratios (dirichlet distribution9
        """
        # todo implement

    def sample_items_sold(self) -> None:
        """
        The quantity of items sold becomes uncertain so now we sample instead of just returning the
        expected value.
        :return: realisations of the
        """
        # todo implement


def random_product_matrix(size: tuple[int, int], fully_connected: bool = True) -> npt.NDArray[np.float64]:
    """
    Generate random product matrix.

    :param size: size of the matrix, i.e. (n_products, n_products)
    :param fully_connected: if True, the matrix is fully connected, i.e. all products are connected to all other products
    :return: a random product matrix
    """
    random_matrix = np.random.uniform(0.0, 1.0, size=size)
    np.fill_diagonal(random_matrix, val=0.0)

    if not fully_connected:
        flatten_matrix = random_matrix.flatten()
        indices_size = np.floor(len(flatten_matrix) * 0.2).astype(int)
        indices = np.random.choice(range(len(flatten_matrix)), size=indices_size, replace=False)

        flatten_matrix[indices] = 0.0
        random_matrix = flatten_matrix.reshape(size)

    return random_matrix

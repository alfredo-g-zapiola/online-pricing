import numpy as np

# from scipy.stats import wishart # for step 5: uncertain graph weights
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr


class EnvironmentBase:
    def __init__(self) -> None:

        self.n_products = 5
        self.n_groups = 3
        """
        Prices and margins: taken from the demand_curves.R we have the prices.
        We assume the cost to be the 40% of the standard price, so that when there is 40% discount,
        we break even (it would hardly make sense otherwise)
        Note the margin decreases linearly with the price
        """
        self.prices_and_margins = {
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
        self.distributions_parameters = {
            "n_people_params": [70, 50, 20],  # we have more poor people than rich people
            "dirichlet_params": [
                np.asarray([15, 10, 6, 5, 4, 6]),
                np.asarray([12, 9, 6, 4, 3, 4]),
                np.asarray([5, 5, 9, 7, 7, 8]),
            ],
            # for the quantity chosen daily we have a ... distribution
            "quantity_demanded_params": [1, 2, 3],
            # product graph probabilities
            "product_graph": [np.array(
                [
                    np.asarray([0.0] + list(np.random.uniform(0.5, 1, 4))),
                    np.asarray(
                        list(np.random.uniform(0.2, .8, size=1))
                        + [0.0]
                        + list(np.random.uniform(0.2, .8, size=3))
                    ),
                    np.asarray(
                        list(np.random.uniform(0.2, .8, size=2))
                        + [0.0]
                        + list(np.random.uniform(0.2, .8, size=2))
                    ),
                    np.asarray(
                        list(np.random.uniform(0.2, .8, size=3))
                        + [0.0]
                        + list(np.random.uniform(0.2, .8, size=1))
                    ),
                    np.asarray(list(np.random.uniform(0.2, .8, size=4)) + [0.0]),
                ]
            ), np.array(
                [
                    np.asarray([0.0] + list(np.random.uniform(0.3, .9, 4))),
                    np.asarray(
                        list(np.random.uniform(0.3, .9, size=1))
                        + [0.0]
                        + list(np.random.uniform(0.3, .9, size=3))
                    ),
                    np.asarray(
                        list(np.random.uniform(0.3, .9, size=2))
                        + [0.0]
                        + list(np.random.uniform(0.3, .9, size=2))
                    ),
                    np.asarray(
                        list(np.random.uniform(0.3, .9, size=3))
                        + [0.0]
                        + list(np.random.uniform(0.3, .9, size=1))
                    ),
                    np.asarray(list(np.random.uniform(0.3, .9, size=4)) + [0.0]),
                ]
            ), np.array(
                [
                    np.asarray([0.0] + list(np.random.uniform(0.4, .1, 4))),
                    np.asarray(
                        list(np.random.uniform(0.4, .1, size=1))
                        + [0.0]
                        + list(np.random.uniform(0.4, .1, size=3))
                    ),
                    np.asarray(
                        list(np.random.uniform(0.4, .1, size=2))
                        + [0.0]
                        + list(np.random.uniform(0.4, .1, size=2))
                    ),
                    np.asarray(
                        list(np.random.uniform(0.4, .1, size=3))
                        + [0.0]
                        + list(np.random.uniform(0.4, .1, size=1))
                    ),
                    np.asarray(list(np.random.uniform(0.4, .1, size=4)) + [0.0]),
                ]
            )
            ], # end of product_graph matrices list
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

    def __init_R(self):
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

    def sample_n_users(self) -> tuple:
        """
        Samples from the poisson distribution how many new potential clients of each group arrive on the current day

        :return: a list with the number of potential clients of each group
        """
        __n_users = tuple(
            np.random.poisson(self.distributions_parameters["n_people_params"][i], 1)
            for i in range(self.n_groups)
        )
        return __n_users

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
        if uncertain:
            d = robjects.r(
                """
                d <- sample.demand({}, {}, 0, 200 )
            """.format(
                    fname, price
                )
            )
            return robjects.r("d")[0]
        else:
            curve_f = robjects.r["{}".format(fname)]
            return curve_f(price)[0]

    def get_direct_clients(self, uncertain_alpha=False) -> dict[str, list[tuple[int, int]]]:
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
        if uncertain_alpha:
            dirichlet_sample = [np.random.dirichlet(self.distributions_parameters["dirichlet_params"][g])
                            for g in range(self.n_groups)]
        else:
            # we just take each element of the vector and divide it by the sum to get the expected value
            dirichlet_sample = [self.distributions_parameters["dirichlet_params"][g] / self.distributions_parameters["dirichlet_params"][g].sum()
                            for g in range(self.n_groups)]

        # take out clients going away (i.e. alpha_0)
        n_direct_clients = [int(n_user[g][0] * (1- dirichlet_sample[g][0]))  for g in range(self.n_groups)]

        direct_clients = {
            f"group_{idx}": list(
                zip(
                    np.random.choice(
                        range(cumsum_clients[idx], cumsum_clients[idx + 1]),
                        size=n_direct_clients[idx],
                        replace=False,
                    ),
                    np.random.choice(list(range(self.n_products)),
                                     n_direct_clients[idx],
                                     # take out element 0, re-normalise
                                     p=dirichlet_sample[idx][1:] / dirichlet_sample[idx][1:].sum()),
                )
            )
            for idx, n_group in enumerate(n_user)
        }
        return direct_clients

    def sample_quantity_bought(self, group, uncertain=False):
        """

        :param group:
        :return:
        """
        m = self.distributions_parameters["quantity_demanded_params"][group]
        if uncertain:
            return np.random.poisson(m)
        else:
            return m

    def compute_clairvoyant(self):
        """
        Finds the optimal superarm that maximises the utility gains
        :param self:
        :return: optimital superarm
        """
        clairvoyant = 100
        return clairvoyant

    def yield_first_secondaries(self):
        """
        Sends to the simulator the two best products to be the secondaries.

        :return: A list of n_products where for each product we have the two secondaries
        """
        return [
            np.flip(np.argsort(self.distributions_parameters["product_graph"][i]))[:2]
            for i in range(self.n_products)
        ]

    def yield_expected_alpha(self, context_generation=False):
        """
        It is assumed the simulator knows the expected values of the alpha ratios.

        :return: an array of the expected values of the five alpha ratios. If we assume a unique group,
        the weighted mean (according to the mean daily customers) is obtained
        """
        if not context_generation:
            # return the weighted mean (according to the number of people in the group) of the alpha ratios
            return sum(
                np.asarray(
                    [
                        self.distributions_parameters["n_people_params"][i]
                        * self.distributions_parameters["dirichlet_params"][i] / self.distributions_parameters["dirichlet_params"][i].sum()
                        for i in range(self.n_groups)
                    ]
                )
            ) / sum(
                [self.distributions_parameters["n_people_params"][i] for i in range(self.n_groups)]
            )
        else:
            return [alphae / alphae.sum() for alphae in self.distributions_parameters["dirichlet_params"]]


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

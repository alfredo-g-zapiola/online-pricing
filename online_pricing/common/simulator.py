import random
from collections import deque, namedtuple
from typing import Any, Deque, TypeVar, cast

from online_pricing.common.environment import EnvironmentBase
from online_pricing.common.influence_function import InfluenceFunctor
from online_pricing.common.social_influence import SocialInfluence
from online_pricing.helpers.tracer import Tracer
from online_pricing.models.learner import Learner, LearnerFactory, TSLearner
from online_pricing.models.user import User


class Simulator(object):
    def __init__(self, environment: EnvironmentBase, seed: int, tracer: Tracer, learner_factory: LearnerFactory):
        """
        Initialize the simulator.

        :param environment: environment of the simulator
        :param seed: seed of the simulator
        :param tracer: tracer of the simulator
        :param learner_factory: factory of the learners
        """
        # Base parameters #
        self.seed = seed  # TODO: use this
        self.groups = range(3)
        self.environment = environment
        self.secondaries = self.environment.yield_first_secondaries()
        self.expected_alpha_r = self.environment.yield_expected_alpha()  # set to True in step 7
        # lambda to go to second secondary products
        self._lambda = self.environment.get_lambda()

        # Price Parameters #
        self.prices = [
            [price_and_margin[0] for price_and_margin in prices_and_margins]
            for prices_and_margins in self.environment.prices_and_margins.values()
        ]
        self.margins = [
            [price_and_margin[1] for price_and_margin in prices_and_margins]
            for prices_and_margins in self.environment.prices_and_margins.values()
        ]
        # start with the lowest prices
        self.current_prices = [self.prices[idx][0] for idx in range(self.environment.n_products)]

        # Learners #
        self.learners = list[Learner]()
        for idx in range(self.environment.n_products):
            learner_factory.base_args = (self.environment.n_products, self.prices[idx])
            self.learners.append(learner_factory.get_learner())

        self.social_influence = SocialInfluence(
            self.environment.n_products, secondaries=self.secondaries, lambda_param=self._lambda
        )
        # estimate the matrix A (present in environment but not known)
        # this is later updated, initialisation not required
        self.estimated_edge_probas: list[list[float]]
        self.quantity_learners = [TSLearner(1, [0])]
        self.n_labeled_groups = 1

        self.influence_functor = InfluenceFunctor(secondaries=self.secondaries, _lambda=self._lambda)
        self.reward_tracer = tracer
        self.n_day = 0

    def sim_one_day(self) -> None:
        """
        Simulate what happens in one day.

        This function simulates what would happen in a real world scenario.
        Clients interact with a primary product. Each client belongs to a group which has
        different probability distributions - meaning behaviours - that determines the outcome of a
        visit. After each client interaction, the current learner (belonging to the current
        configuration of prices) is updated. Then, the cumulative sold products array is updated
        along with the empirical influence matrix that records the jumps to secondary products.
        """
        direct_clients = self.environment.get_direct_clients()
        products_sold: list[int] = [0] * self.environment.n_products
        n_users = 0
        for client in direct_clients:
            n_users += 1
            buys, influenced_episodes = self.sim_one_user(
                client=client,
                product_graph=self.environment.distributions_parameters["product_graph"][client.group].sample(),
                prices=self.current_prices,
            )
            products_sold = sum_by_element(products_sold, buys)

            self.update_learners(buys=buys, prices=self.current_prices)
            self.social_influence.add_episode(influenced_episodes)
        # Estimate probabilities
        # Regret calculator
        self.estimated_edge_probas = self.social_influence.estimate_probabilities()

        current_margins = [
            self.margins[product_id][self.prices[product_id].index(idx)]
            for product_id, idx in enumerate(self.current_prices)
        ]
        mean_reward_per_client = self.get_reward(n_user=n_users, products_sold=products_sold, margins=current_margins)
        learner_data = self.get_learner_data()

        self.reward_tracer.add_avg_reward(mean_reward_per_client)
        self.reward_tracer.add_arm_data(learner_data)

        next_day_configuration = self.greedy_algorithm()
        self.current_prices = [self.prices[idx][price_id] for idx, price_id in enumerate(next_day_configuration)]

        # print("\n =========== DAY OVER ===========")
        # print("Products sold:", products_sold)
        # print("Current prices:", self.current_prices)
        # print("Current margins:", current_margins)
        # print("Mean reward per client:", mean_reward_per_client)
        # print("Next configuration:", next_day_configuration)
        # print("Estimated edge probabilities:")
        # print_matrix(self.estimated_edge_probas)
        # print("Product Graph:")
        # print_matrix(self.environment.mean_product_graph)
        # print("Secondaries:")
        # print_matrix(self.secondaries, indexes=True)
        # print("\n")
        self.n_day += 1

    def sim_buy(self, group: int, product_id: int, price: float) -> int:
        """
        Simulate the buy of a product for a user belonging to a group.

        If the willing_price of a user is higher than the price of the product, the user will buy
        it. Next, the quantity of units bought will be decided, independently of the
        willing_price of a user, by sampling a probability distribution.

        :param group: group of the user
        :param product_id: product id
        :param price: price of the product
        :return: number of units bought
        """
        buy_probability = self.environment.sample_demand_curve(
            group=group, prod_id=product_id, price=price, n_day=self.n_day
        )
        n_units = 0
        if random.random() <= buy_probability:
            n_units = self.environment.sample_quantity_bought(group)
            self.quantity_learners[0].update(0, n_units)

        return n_units

    def sim_one_user(self, client: User, product_graph: Any, prices: list[float]) -> tuple[list[int], list[list[int]]]:
        """
        Function to simulate the behavior of a single user.

        A user may start from a primary product and then buy secondary products. This behaviour
        is implemented through a Breadth First Search algorithm. If a user buys a primary product,
        he will then choose to enter the secondary products. Each secondary product is associated with a probability
        to "click on it". If the user clicks on a secondary product, he will then choose to buy it if the price is
        interesting. When no more products are available, the iteration will end.

        :param client: user
        :param product_graph: secondary product probability graph
        :param prices: prices of the products
        :return: list of number of units bought per product and the influencing matrix
        """
        # Instantiate buys and influence episodes
        buys: list[int] = [0] * self.environment.n_products
        visited: list[int] = [0] * self.environment.n_products
        influence_episodes = []

        # Data structure that represent every node of the graph and a probability that the user will enter it
        VisitingNode = namedtuple("VisitingNode", ["product_id", "probability"])

        # Initialize the queue with the landing product, it has a probability of 1 to be seen
        visiting_que: Deque[VisitingNode] = deque()
        visiting_que.append(VisitingNode(product_id=client.landing_product, probability=1))
        while visiting_que:
            current_node = visiting_que.pop()
            product_id = current_node.product_id
            # If the user clicks on it
            if random.random() <= current_node.probability and not visited[product_id]:
                visited[product_id] = 1
                # Simulate the buy of the product and update records
                buys[product_id] = self.sim_buy(client.group, product_id, prices[product_id])
                influence_episodes.append(visited.copy())

                # If the user bought something, we unlock the secondary products
                if buys[product_id]:

                    # Add the first advised to the queue
                    first_advised = self.secondaries[product_id][0]
                    first_probability = product_graph[product_id][first_advised]
                    visiting_que.append(VisitingNode(product_id=first_advised, probability=first_probability))

                    # Add the second advised to the queue
                    second_advised = self.secondaries[product_id][1]
                    second_probability = product_graph[product_id][second_advised] * self._lambda
                    visiting_que.append(VisitingNode(product_id=second_advised, probability=second_probability))

        # We need to return the history of jumpes between pages, but just the information about
        # the landing page, meaning we need to know if the user viewed the first and/or second secondary products.
        history = influence_episodes[:3]
        prepare_episode = [history[0] if buys[client.landing_product] else [0] * self.environment.n_products]

        prepare_episode.append(
            sum_by_element(history[-1], history[0], difference=True) if buys[client.landing_product] else [-1]
        )

        # Return history records
        return buys, [episode for episode in prepare_episode if episode != [-1]]

    def update_learners(self, buys: list[int], prices: list[float]) -> None:
        """
        Update the learners with the buys of the user.

        The reward is 1 if the user bought a product, 0 otherwise.

        :param buys: list of number of units bought per product
        :param prices: prices of the products
        """
        arms_pulled = [self.learners[idx].get_arm(prices[idx]) for idx in range(self.environment.n_products)]
        did_buy = [int(buy > 0) for buy in buys]

        for idx, bought in enumerate(did_buy):
            self.learners[idx].update(arm_pulled=arms_pulled[idx], reward=bought)

    def conversion_rate(self, product_id: int, prices: list[float]) -> float:
        if not self.environment.unknown_demand_curve:
            return self.learners[product_id].sample_arm(self.learners[product_id].get_arm(prices[product_id]))
        else:
            # note we have to take the average conversion rate, so we weight the c_rate of each group
            # save old value
            save = self.environment.uncertain_demand_curve
            self.environment.uncertain_demand_curve = False  # since we need expected value
            rate = sum(
                [
                    self.environment.sample_demand_curve(
                        group=g, prod_id=product_id, price=prices[product_id], n_day=self.n_day
                    )
                    * self.environment.group_proportions[g]
                    for g in range(self.environment.n_groups)
                ]
            )
            self.environment.uncertain_demand_curve = save  # ripristinarlo
            return rate

    def influence_function(self, i: int, j: int, prices: list[float]) -> float:
        """
        Sums the probability of clicking product j given product i was bought (all possible paths, doing one to 4 jumps)
        :return:
        """

        def c_rate(prod_id: int) -> float:
            return self.conversion_rate(prod_id, prices=prices)

        if self.environment.unknown_product_weights:
            return self.influence_functor(i, j, c_rate, self.estimated_edge_probas)

        else:  # we do not use the estiamted edge probas
            return self.influence_functor(i, j, c_rate, self.environment.mean_product_graph)

    def mean_quantity_bought(self) -> float:
        if not self.environment.context_generation:
            if self.environment.unknown_quantity_bought:
                return self.quantity_learners[0].sample_arm(0)
            else:
                return sum(
                    [
                        self.environment.distributions_parameters["quantity_demanded_params"][g]
                        * self.environment.group_proportions[g]
                        for g in range(self.environment.n_groups)
                    ]
                )
        else:
            raise Exception("Need to develop context generation case")

    def greedy_algorithm(self) -> list[int]:
        """
        Greedy algorithm to select next day configuration of prices.

        It works as follows:
        Starting from the lowest configuration, which is the cheapest configuration with price indexes equal to 0,
        it iteratively increases one of the prices, checks if the new configuration has a better objective
        function value, and if so, it sets this new configuration. It does this until no better configurations exists.

        :return: the best configuration of prices for the next day
        """
        products = range(self.environment.n_products)
        n_prices = len(self.prices[0])

        best_configuration = [0] * self.environment.n_products
        current_target = self.objective_function([self.prices[product_id][0] for product_id in products])
        has_changed = True
        while has_changed:
            has_changed = False
            for price_to_increase in products:

                new_configuration = best_configuration.copy()
                # While checking that the index doesn't exceed the number of prices available
                if new_configuration[price_to_increase] >= n_prices - 1:
                    continue
                # Increase the price
                new_configuration[price_to_increase] += 1

                new_target = self.objective_function(
                    [self.prices[product_id][new_configuration[product_id]] for product_id in products]
                )
                # If objective value is higher, update the configuration
                if new_target > current_target:
                    best_configuration = new_configuration
                    current_target = new_target
                    has_changed = True

        return best_configuration

    # TODO(film): add group support and check this (coded too fast)

    def objective_function(self, prices: list[float]) -> float:

        current_margins = [
            self.margins[product_id][self.prices[product_id].index(idx)] for product_id, idx in enumerate(prices)
        ]
        alpha_ratios = cast(list[float], self.expected_alpha_r)

        if self.environment.unknown_demand_curve:  # use the estimates
            conversion_rates = [
                self.learners[product_id].sample_arm(self.learners[product_id].get_arm(prices[product_id]))
                for product_id in range(self.environment.n_products)
            ]
        else:

            def c_rate(prod_id: int) -> float:
                return self.conversion_rate(prod_id, prices=prices)

            conversion_rates = [c_rate(product_id) for product_id in range(self.environment.n_products)]
        return sum(
            [
                alpha_ratios[product_id + 1]
                * (
                    conversion_rates[product_id] * current_margins[product_id] * self.mean_quantity_bought()
                    + sum(
                        [
                            self.influence_function(product_id, secondary_product_id, prices)
                            * conversion_rates[secondary_product_id]
                            * current_margins[secondary_product_id]
                            * self.mean_quantity_bought()
                            for secondary_product_id in range(self.environment.n_products)
                            if secondary_product_id != product_id
                        ]
                    )
                )
                for product_id in range(self.environment.n_products)
            ]
        )

    def get_reward(self, n_user: int, products_sold: list[int], margins: list[float]) -> float:
        """
        Get the reward for a given user, given the products sold and the margins.

        :param n_user: number of the user.
        :param products_sold: list of products sold.
        :param margins: list of margins.
        :return: reward for the user.
        """
        return sum([margins[i] * products_sold[i] for i in range(self.environment.n_products)]) / n_user if n_user > 0 else 0

    def get_learner_data(self) -> list[list[float]]:
        """
        Get the data of the learners.

        :return: list of list of data.
        """
        return [[learner.sample_arm(price) for price in range(self.environment.n_prices)] for learner in self.learners]


MATRIX = TypeVar("MATRIX", list[int], list[list[int]])


def sum_by_element(array_1: MATRIX, array_2: MATRIX, difference: bool = False) -> MATRIX:
    """
    Sum lists - or matrices - by element.

    :param array_1: list or matrix to sum.
    :param array_2: list or matrix to sum.
    :param difference: if True, return the difference between the two arrays.
    :return: list or matrix with the sum of the two arrays.
    """
    if type(array_1) is not type(array_2):
        raise TypeError(f"Arrays must be of the same type, got {type(array_1)} and {type(array_2)}")

    if isinstance(array_1[0], list):
        return [sum_by_element(a1, a2) for a1, a2 in zip(array_1, array_2)]

    if difference:
        return [a1 - a2 for a1, a2 in zip(array_1, array_2)]

    return [sum(items) for items in zip(array_1, array_2)]


def print_matrix(matrix: list[list[float | int]], indexes: bool = False) -> None:
    if indexes:
        indices = ["-"] + [str(i) for i in range(1, len(matrix[0]) + 1)]
        new_matrix = [[str(matrix[j][i]) for i in range(len(matrix[0]))] for j in range(len(matrix))]
        new_matrix.insert(0, indices)
        for idx in range(1, len(new_matrix)):
            new_matrix[idx].insert(0, str(idx))

        print("\n".join(["".join([f"{item} " for item in row]) for row in new_matrix]))

    elif type(matrix[0][0]) == int:
        print("\n".join(["".join([f"{item} " for item in row]) for row in matrix]))
    else:
        print("\n".join(["".join([f"{item:.2f} " for item in row]) for row in matrix]))
import random
from typing import Any, Type
import itertools
from collections import deque, namedtuple
import numpy as np

from online_pricing.environment import EnvironmentBase
from online_pricing.learner import Learner, TSLearner
from online_pricing.social_influence import SocialInfluence


class Simulator(object):
    def __init__(self, environment: Type[EnvironmentBase] = EnvironmentBase, seed: int = 41703192):
        self.seed = seed
        self.groups = range(3)
        self.__SocialInfluence = None
        self.environment = environment()
        self.secondaries = self.environment.yield_first_secondaries()
        self.expected_alpha_r = self.environment.yield_expected_alpha(
            context_generation=False
        )  # set to True in step 7
        self.prices = [
            [*price_and_margins.keys()]
            for price_and_margins in self.environment.prices_and_margins.values()
        ]
        # start with lowest prices
        self.current_prices = [
            np.min(self.prices[idx]) for idx in range(self.environment.n_products)
        ]
        # lambda to go to second secondary product
        self._lambda = 0.5
        self.learners: list[Learner] = [
            TSLearner(n_arms=self.environment.n_products, prices=self.prices[idx])
            for idx in range(self.environment.n_products)
        ]
        self.social_influence = SocialInfluence()
        # estimate the matrix A (present in environment but not known)
        # TODO this should be updated, initialisation not required
        self.estimated_edge_probas = [
            np.random.uniform(size=5) * [1 if j in self.secondaries[i] else 0 for j in range(5)]
            for i in range(self.environment.n_products)
        ]

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
        # TODO aggiungere controllo id -1: altro sito web
        products_sold: list[int, int] = [0] * self.environment.n_products

        for group in self.groups:
            for client_id, primary_product in direct_clients[f"group_{group}"]:
                buys, influenced_episodes = self.sim_one_user(
                    group=group,
                    landing_product=primary_product,
                    product_graph=self.environment.distributions_parameters["product_graph"].copy(),
                    prices=self.current_prices,
                )
                products_sold = sum_by_element(products_sold, buys)

                self.update_learners(buys=buys, prices=self.current_prices)
                self.social_influence.add_episode(influenced_episodes)

        # TODO: Add here Social Influence and Greedy Algorithm
        self.estimated_edge_probas = [
            self.social_influence.estimate_probabilities(i, n_products=5) for i in range(5)
        ]

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
        willing_price = self.environment.sample_demand_curve(
            group=group, prod_id=product_id, price=price, uncertain=False
        )  # TODO filippo we could make one simulator per step
        n_units = 0
        if price < willing_price:
            n_units = self.environment.sample_quantity_bought(group)

        return n_units

    def sim_one_user(
        self, group: int, landing_product: int, product_graph: Any, prices: list[float]
    ) -> tuple[list[int], list[list[int]]]:
        """
        Function to simulate the behavior of a single user.

        A user may start from a primary product and then buy secondary products. This behaviour
        is implemented through a Breadth First Search algorithm. If a user buys a primary product,
        he will then choose to enter the secondary products. Each secondary product is associated with a probability
        to "click on it". If the user clicks on a secondary product, he will then choose to buy it if the price is
        interesting. When no more products are available, the iteration will end.

        :param group: group of the user
        :param landing_product: landing product id
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
        visiting_que = deque()
        visiting_que.append(VisitingNode(product_id=landing_product, probability=1))

        while visiting_que:
            current_node = visiting_que.pop()
            product_id = current_node.product_id
            visited[product_id] = 1

            # If the user clicks on it
            if random.random() <= current_node.probability and not visited[product_id]:
                # Simulate the buy of the product and update records
                buys[product_id] = self.sim_buy(group, product_id, prices[product_id])
                influence_episodes.append(
                    [0 if idx != product_id else 1 for idx in range(self.environment.n_products)]
                )

                # If the user bought something, we unlock the secondary products
                if buys[product_id]:
                    # Add the first advised to the queue
                    first_advised = (self.secondaries[product_id].primary,)
                    first_probability = product_graph[product_id][first_advised]
                    visiting_que.append(
                        VisitingNode(product_id=first_advised, probability=first_probability)
                    )

                    # Add the second advised to the queue
                    second_advised = self.secondaries[product_id].secondary
                    second_probability = product_graph[product_id][second_advised] * self._lambda
                    visiting_que.append(
                        VisitingNode(product_id=second_advised, probability=second_probability)
                    )

        # Return history records
        return buys, influence_episodes

    def update_learners(self, buys: list[int], prices: list[float]) -> None:
        """
        Update the learners with the buys of the user.

        The reward is 1 if the user bought a product, 0 otherwise.

        :param buys: list of number of units bought per product
        :param prices: prices of the products
        """
        arms_pulled = [
            self.learners[idx].get_arm(prices[idx]) for idx in range(self.environment.n_products)
        ]
        did_buy = [int(buy > 0) for buy in buys]

        for idx, bought in enumerate(did_buy):
            self.learners[idx].update(arm_pulled=arms_pulled[idx], reward=bought)

    def formula(self, alpha, conversion_rates, margins, influence_probability):
        s1 = 0
        s2 = 0
        sum = 0
        for i in range(0, 5):
            s1 += conversion_rates[i] * margins[i]
            for j in range(0, 5):
                if i != j:
                    s2 += influence_probability[i][j] * conversion_rates[j] * margins[j]
            sum = sum + alpha[i] * (s1 + s2)
            s2 = 0
            s1 = 0
        return sum

    def c_rate(self, j):
        return self.learners[j].sample_arm(np.argwhere(self.prices[j] == self.current_prices[j]))

    def influence_function(self, i, j):
        """
        Sums the probability of clicking product j given product i was bought (all possible paths, doing one to 4 jumps)
        :return:
        """

        def assign_sec(k, l):
            if self.secondaries[k][0] == l:
                return 1
            elif self.secondaries[k][1] == l:
                return 2
            else:
                return 0

        # all possible paths from index i to j (if j appears before other indices, we cut the path)
        paths = [path for path in itertools.permutations([k for k in range(5) if i != k])]
        influence = 0
        histories = []
        for path in paths:
            print("Current path: ", path)
            path_proba = 1
            last_edge = i
            history = [last_edge]
            for edge in path:
                # if it is secondary
                print("Path proba ", path_proba)
                status = assign_sec(last_edge, edge)
                if status > 0:  # if it appears as a secondary
                    cur_jump_proba = (
                        self.c_rate(last_edge)
                        * self.estimated_edge_probas[last_edge][edge]
                        * (self._lambda if status == 2 else 1)
                    )
                    if edge == j:

                        path_proba *= cur_jump_proba

                        if history in histories:  # avoid repetitions
                            path_proba *= 0
                            print("Already visited")
                        histories.append(history)
                        print("History: ", history, "\nprobability: ", path_proba, "\n")
                        break  # finish the path

                    else:  # the edge is not a destination
                        path_proba *= cur_jump_proba
                        last_edge = edge  # update the last edge
                        history.append(edge)
                else:
                    path_proba *= (
                        0  # infeasible: this product cannot be reached directly from last_edge
                    )
                    print("Infeasible. History: ", history, "\nprobability: ", path_proba, "\n")
                    break  # go to next path

            influence += path_proba
        return influence

    def greedy_algorithm(
        self, conversion_rates, margins, influence_probability
    ):  # greedy alg without considering groups. Alpha is a list of 5 elements,
        prices = [0, 0, 0, 0, 0]  # conversion_rates and margins are matrix 5x4 (products x prices)
        max = self.formula(
            self.alpha, conversion_rates[:, 0], margins[:, 0], influence_probability
        )  # influence_probability is a matrix 5x5 (products x products) where cell ij is the
        while True:  # probability to go from i to j
            changed = False
            best = prices  # best configuration
            for i in range(0, 5):
                temp = prices  # new configuration where it is incremented the price of a product
                cr = []  # list of 5 conversion rates, one for each product, to pass to the formula
                mr = []  # list of 5 margins, one for each product, to pass to the formula
                temp[i] += 1  # one price is incremented
                if temp[i] > 3:
                    temp[i] = 3  # there are max 4 prices
                for j in range(0, 5):
                    cr.append(
                        conversion_rates[j, temp[j]]
                    )  # for each product j, I obtain its conversion rate knowing its price in temp[j]
                    mr.append(
                        margins[j, temp[j]]
                    )  # for each product j, I obtain its margin knowing its price in temp[j]
                n = self.formula(self.alpha, cr, mr, influence_probability)
                if n > max:
                    max = n
                    best = temp  # save the best configuration
                    changed = True
            if not changed:
                return best
            if changed:
                prices = best

    def formula_with_groups(self, list_alpha, conversion_rates, margins, influence_probability):
        s1 = 0
        s2 = 0
        sum = 0
        for g in range(0, 3):
            for i in range(0, 5):
                s1 += conversion_rates[g][i] * margins[i]
                for j in range(0, 5):
                    if i != j:
                        s2 += influence_probability[i][j] * conversion_rates[g][j] * margins[j]
                sum = sum + list_alpha[g][i] * (s1 + s2)
                s2 = 0
                s1 = 0
        return sum

    def greedy_algorithm_with_groups(
        self, list_alpha, list_conversion_rates, margins, influence_probability
    ):  # greedy algorithm considering groups. Same as before except for:
        prices = [
            0,
            0,
            0,
            0,
            0,
        ]  # list_alpha is a list of 3 lists where each list contains 5 elements
        conversion_rates = (
            []
        )  # list_conversion_rates is a list of 3 matrices where each matrix is 5x4 (products x prices)
        for g in range(0, 3):
            conversion_rates.append(list_conversion_rates[g][:, 0])
        max = self.formula_with_groups(
            list_alpha,
            conversion_rates,
            margins[:, 0],
            influence_probability,  # compute the value for the configuration 0,0,0,0,0
        )
        while True:
            changed = False
            best = prices  # best configuration
            for i in range(0, 5):
                temp = prices  # new configuration where it is incremented the price of a product
                cr = [
                    [0 for v in range(5)] for w in range(3)
                ]  # 3 lists containining 5 conversion rates, one for each product in each group, to pass to the formula
                mr = (
                    []
                )  # 3 lists containining 5 margins, one for each product in each group, to pass to the formula
                temp[i] += 1  # one price is incremented
                if temp[i] > 3:
                    temp[i] = 3  # there are max 4 prices
                for g in range(0, 3):
                    for j in range(0, 5):
                        cr[g][j] = list_conversion_rates[g][
                            j, temp[j]
                        ]  # for each product j in group g, I obtain its conversion rate knowing its price in temp[j]
                for k in range(0, 5):
                    mr.append(
                        margins[k, temp[k]]
                    )  # for each product j in group g, I obtain its margin knowing its price in temp[j]
                n = self.formula_with_groups(list_alpha, cr, mr, influence_probability)
                if n > max:
                    max = n
                    best = temp  # save the best configuration
                    changed = True
            if not changed:
                return best
            if changed:
                prices = best


def sum_by_element(array_1: list[Any], array_2: list[Any]) -> list[Any]:
    """Sum lists - or matrices - by element."""
    if type(array_1) is not type(array_2):
        raise TypeError(f"Arrays must be of the same type, got {type(array_1)} and {type(array_2)}")

    if isinstance(array_1[0], list):
        return [sum_by_element(a1, a2) for a1, a2 in zip(array_1, array_2)]

    return [sum(items) for items in zip(array_1, array_2)]


def get_buys_reward(buys_dict: dict[str, dict[int, int]], margins: Any) -> int:
    """Calculate reward from margin detail and number of buys"""
    pass

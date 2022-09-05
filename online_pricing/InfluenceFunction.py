import itertools

class InfluenceFunctor:
    def __init__(self, secondaries, _lambda):
        self.secondaries = secondaries
        self._lambda = _lambda

    def __call__(self, i, j, c_rate, edge_probas):
        """
        :param i: starting product
        :param j: other product
        :param c_rate: function that yields the conversion rate of product of p \in P
        :param edge_probas: the edge probabilities (estimated in the case they are unknown)
        :return: sum of probabilities of different paths to click j given that i was bought
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
            path_proba = 1
            last_edge = i
            history = [last_edge]
            for edge in path:
                # if it is secondary
                status = assign_sec(last_edge, edge)
                if status > 0:  # if it appears as a secondary
                    cur_jump_proba = (
                        c_rate(last_edge)
                        * edge_probas[last_edge][edge]
                        * (self._lambda if status == 2 else 1)
                    )
                    if edge == j:

                        path_proba *= cur_jump_proba

                        if history in histories:  # avoid repetitions
                            path_proba *= 0
                        histories.append(history)
                        break  # finish the path

                    else:  # the edge is not a destination
                        path_proba *= cur_jump_proba
                        last_edge = edge  # update the last edge
                        history.append(edge)
                else:
                    path_proba *= (
                        0  # infeasible: this product cannot be reached directly from last_edge
                    )
                    break  # go to next path

            influence += path_proba
        return influence
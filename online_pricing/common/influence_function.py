import itertools
from typing import Callable, cast


class InfluenceFunctor:
    def __init__(self, secondaries: list[list[int]], _lambda: float) -> None:
        self.secondaries = secondaries
        self._lambda = _lambda
        self.feasible_paths = [[] for _ in range(len(self.secondaries))]
        self.compute_feasible_paths()

    def compute_feasible_paths(self) -> None:
        for i in range(5):
            paths_for_i = []
            paths = cast(list[tuple[int, ...]], [path for path in itertools.permutations([k for k in range(5) if i != k])])

            def assign_sec(primary: int, secondary: int) -> int:
                if self.secondaries[primary][0] == secondary:
                    return 1
                elif self.secondaries[primary][1] == secondary:
                    return 2
                else:
                    return 0

            for j in range(5):
                influence = 0.0
                histories = []
                infeasible_histories = []
                for path in paths:
                    path_proba = 1.0
                    last_edge = i
                    history = [last_edge]
                    for edge in path:
                        # if it is secondary
                        status = assign_sec(last_edge, edge)
                        if status > 0:  # if it appears as a secondary
                            history.append(edge)
                            cur_jump_proba = 1
                            if edge == j:

                                path_proba *= cur_jump_proba

                                if history in histories:  # avoid repetitions
                                    path_proba *= 0
                                    break
                                histories.append(history)
                                break  # finish the path

                            else:  # the edge is not a destination
                                path_proba *= cur_jump_proba
                                last_edge = edge  # update the last edge
                        else:
                            path_proba *= 0  # infeasible: this product cannot be reached directly from last_edge
                            infeasible_histories.append(history)
                            break  # go to next path
                    influence += path_proba
                paths_for_i.append(histories)
                self.feasible_paths[i] = paths_for_i

    def __call__(self, i: int, j: int, c_rate: Callable[[int], float], edge_probas: list[list[float]]) -> float:
        """
        :param i: starting product
        :param j: other product
        :param c_rate: function that yields the conversion rate of product of i in P
        :param edge_probas: the edge probabilities (estimated in the case they are unknown)
        :return: sum of probabilities of different paths to click j given that i was bought
        """

        def assign_sec(primary: int, secondary: int) -> int:
            if self.secondaries[primary][0] == secondary:
                return 1
            elif self.secondaries[primary][1] == secondary:
                return 2
            else:
                return 0

        # all possible paths from index i to j (if j appears before other indices, we cut the path)
        influence = 0.0
        histories = []
        for path in self.feasible_paths[i][j]:
            path_proba = 1.0
            last_edge = i
            history = [last_edge]
            for edge in path[1:]:
                # if it is secondary
                status = assign_sec(last_edge, edge)
                if status > 0:  # if it appears as a secondary
                    cur_jump_proba = (
                        (c_rate(last_edge) if last_edge != i else 1)
                        * edge_probas[last_edge][edge]
                        * (1 if status == 2 else 1)
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
                    path_proba *= 0  # infeasible: this product cannot be reached directly from last_edge
                    break  # go to next path
            influence += path_proba

        return influence

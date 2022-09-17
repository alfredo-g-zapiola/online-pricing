from typing import cast

import numpy as np
import numpy.typing as npt
from scipy.linalg import toeplitz
from scipy.stats import wishart


class WishartHandler:
    """
    A class to create product_graph.

    Note it assumes a wishart distribution, which is symmetric
    hence for the upper-right product_graph matrix we will have one wishart,
    for the lower-left another one; both initialised from the same toeplitz matrix which takes as input
    an ordered uniform sample
    """

    def __init__(
        self, size: int, df: int, unif_params: tuple[float, float], uncertain: bool, fully_connected: bool, seed: int
    ) -> None:
        np.random.seed(seed)
        self.size = size
        self.df = df
        first_rows = [  # to initialise the wishart distributions!
            np.flip(np.sort(np.random.uniform(size=size, low=unif_params[0], high=unif_params[1]))),
            np.flip(np.sort(np.random.uniform(size=size, low=unif_params[0], high=unif_params[1]))),
        ]
        first_rows[0][0] = 1  # set the highest values to 1 (this has to do with eigenvalues of the cov matrix)
        first_rows[1][0] = 1

        # to make the product graph, we have two wishart matrices (initialised by two toeplitz matrices
        # which are initialised by two different vectors which are samples of a same uniform distribution)
        # that uniform distribution depends on the group: richer people means higher values
        self.wisharts = [wishart(self.df, toeplitz(first_rows[0])), wishart(self.df, toeplitz(first_rows[1]))]
        self.uncertain = uncertain
        self.fully_connected = fully_connected
        self.zero_pct = 0.60  # percentage of the edges to set to 0

        # obtain mean:
        self.mean = self.generate_product_graph(
            [self.cov2corr(self.wisharts[0].scale * self.df), self.cov2corr(self.wisharts[1].scale)]
        )

        self.to_disconnect = [row for row in range(self.size) if np.random.random() <= self.zero_pct]
        if not self.fully_connected:
            self.mean = self.disconnect(self.mean)

        np.fill_diagonal(self.mean, val=-1.0)  # the diagonal has to be -1: a product does not influence itself!

    def sample(self) -> npt.NDArray[np.float32]:
        if self.uncertain:
            samples = [self.cov2corr(wish.rvs()) for wish in self.wisharts]
            prod_graph: npt.NDArray[np.float32] = self.generate_product_graph(samples)
            if not self.fully_connected:
                prod_graph = self.disconnect(prod_graph)
            np.fill_diagonal(prod_graph, val=-1)
            return prod_graph

        else:
            return self.mean  # if fully connected or not already addressed at the init

    @staticmethod
    def cov2corr(
        cov_matrix: npt.NDArray[
            np.float32,
        ]
    ) -> npt.NDArray[np.float32]:
        """
        From the wishart we sample a covariance matrix
        To have probabilities (i.e. values between 0 and 1) we transform it into a correlation matrix
        For this we take the variances diagonal matrix and invert it
        :param cov_matrix:
        :return:s
        """
        s_minus_1 = np.linalg.inv(np.sqrt(np.diag(np.diag(cov_matrix))))  # yes, we need np.diag twice
        corr_matrix: npt.NDArray[np.float32] = np.dot(np.dot(s_minus_1, cov_matrix), s_minus_1)
        return corr_matrix

    def generate_product_graph(self, wish_samples: list[npt.NDArray[np.float32]]) -> npt.NDArray[np.float32]:
        """


        https://stackoverflow.com/questions/8905501/extract-upper-or-lower-triangular-part-of-a-numpy-matrix

        :param wish_samples:
        :return:
        """

        sample = np.zeros((self.size, self.size))
        sample += np.triu(wish_samples[0])  # add the first wishart to the upper triangle
        sample += np.tril(wish_samples[1])
        return cast(npt.NDArray[np.float32], sample)

    def disconnect(self, random_matrix: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Set some edges to 0
        :param random_matrix: the matrix of which some rows are set to 0
        :return:
        """
        for row_index in range(self.size):
            if row_index in self.to_disconnect:
                random_matrix[row_index] *= 0
        return random_matrix

        # flatten_matrix = random_matrix.flatten()
        # indices_size = int(np.floor(len(flatten_matrix) * 0.60))  # 60% of the matrix is zeroed
        # indices = np.random.choice(range(len(flatten_matrix)), size=indices_size, replace=False)
        #
        # flatten_matrix[indices] = 0.0
        # random_matrix = flatten_matrix.reshape(size)
        # return random_matrix

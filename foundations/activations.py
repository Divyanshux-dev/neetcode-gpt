import numpy as np
from numpy.typing import NDArray

class Solution:

    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        transsig = 1 / (1 + np.exp(-z))
        return np.round(transsig, 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        transrelu = np.maximum(0, z)
        return np.round(transrelu, 5)

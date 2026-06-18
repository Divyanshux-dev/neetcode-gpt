import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        l = -1/(np.size(y_true))
        s = np.sum(y_true*np.log(y_pred + 1e-7) + (1 - y_true)*np.log(1-y_pred + 1e-7))
        return np.round(l*s, 4)
       

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        ## y_true is sample and shape to get len we should get the number of sample
        l = -1/y_true.shape[0]
        s = np.sum(np.sum(y_true*np.log(y_pred + 1e-7)))
        return np.round(l*s, 4)

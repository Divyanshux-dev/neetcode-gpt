import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        
       z = z - np.max(z) 
       
       y = np.exp(z) / np.sum(np.exp(z))
        # Hint: subtract max(z) for numerical stability before computing exp
       return np.round(y, 4)
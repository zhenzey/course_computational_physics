import numpy as np
def energy_gradient(energyf, h, x):
    """

    :return:
    """
    derivative = []
    for i in range(x.size):
        w1 = np.array([x_j + (h if j == i else 0) for j, x_j in enumerate(x)])
        w2 = np.array([x_j - (h if j == i else 0) for j, x_j in enumerate(x)])
        derivative.append((f(w1)-f(w2))/ 2 / h)
    deriative = np.array(deriative).reshape(-1,3)
    return derivative
    pass

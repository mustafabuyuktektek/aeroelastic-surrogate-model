import numpy as np


def solve_flutter(params):

    mach, aoa, q, k = params.T
    flutter_load = (q * mach ** 2) / k + np.sin(aoa * np.pi) * np.exp(mach)

    noise = np.random.normal(0, 0.05, size=flutter_load.shape)
    return flutter_load + noise
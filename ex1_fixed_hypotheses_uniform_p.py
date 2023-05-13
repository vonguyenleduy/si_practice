import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def run():
    n = 3
    x = np.random.normal(loc=0, scale=1, size=(n, 1))

    # vector of direction of interest (always test x_1)
    eta = np.zeros((n, 1))
    eta[0] = 1

    # test-statistic
    etaTx = np.dot(eta.T, x)[0][0]

    # compute p_value
    p_value = 1 - norm.cdf(etaTx)

    return p_value


if __name__ == '__main__':
    # np.random.seed(5)
    # run()

    max_iteration = 12000
    list_p_value = []

    for each_iter in range(max_iteration):
        if each_iter % 10 == 0:
            print(each_iter)

        p_value = run()

        if p_value is not None:
            list_p_value.append(p_value)

    plt.rcParams.update({'font.size': 17})
    plt.hist(list_p_value)
    plt.tight_layout()
    plt.show()
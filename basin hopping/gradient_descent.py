import numpy as np
from numpy.linalg import norm


def line_search(line_f, alpha0, m, c=1.0, t=1.0):
    while line_f(alpha0) > line_f(0) + alpha0*c*m:
        alpha0 *= t
    return alpha0


def steepest_descent(x0, f, df, iterations, tolerance=1e-8, c=1.0, t=1.0):
    """


    :param x0:
    :param f:
    :param df:
    :param iterations:
    :param tolerance:
    :param c:
    :param t:
    :return:
    """

    distance = 1e10
    alpha = 1.0
    x1 = x0.copy()
    iteration = 0

    while distance > tolerance:
        if iteration > iterations:
            return x1, False
        iteration += 1
        x0[:] = x1
        gradient = df(x0)
        m = norm(gradient)
        p = -gradient / m

        def line(alpha):
            return f(x0 + alpha*p)

        alpha = line_search(line, alpha, m, c, t)
        x1[:] = x0 + alpha * p
        distance = norm(x1 - x0)

    return x1, True

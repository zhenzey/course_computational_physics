import numpy as np
import energy_gradient
import gradient_descent

def basin_hopping(energyf, newr, iterations, x0, minimiserf, vtol):
    """
     >>>basin_hopping(ljpotential, 0.1, 100, x= np.array([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0],[1.0,1.0,1.0]]), steepest_descend, 1e-5)
     1

    :param energyf:
    :param newr:
    :param iterations:
    :param x0:
    :param minimiserf:
    :param vtol:
    :return:
    """


    step_size = 1.0
    temperature = 1.0
    v = energyf(x0)
    x_g_min = x0.copy()
    v_g_min = v
    new = 0
    accept = 0
    new_flag = False

    for iteration in range(iterations):
        x = x0 + step_size * (2 - np.random.rand(*x0.shape))
        x_min = minimiserf(x)
        v_new = energyf(x_min)
        if abs(v_new - v) > vtol:
            new += 1
            new_flag = True
        else:
            new_flag = False
        if v_new < v or np.exp(-(v_new - v)/temperature) > np.random.rand():
            x0[:] = x_min
            v = v_new
            if new_flag:
                accept += 1
            if v < v_g_min:
                v_g_min = v
                x_g_min[:] = x0[:]
        print("iteration, v, v_g_min:{:10n}{:20.10f}{:20.10f}".format(iteration, v, v_g_min))
        if iteration % 50 == 0:
            if new / iteration > newr:
                step_size /= 1.01
            else:
                step_size *= 1.01
            if accept / new > acceptr:
                temperature /= 1.01
            else:
                temperature *= 1.01

    return v_g_min, x_g_min

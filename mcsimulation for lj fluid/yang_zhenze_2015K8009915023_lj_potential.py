import numpy as np
import math
def pair_energy(x, energy=None, args=()):
    """

    Calculates the potential energy of a configuration of particles.

    :param x: positions of the particles
    :type x: numpy.ndarray
    :param energy: the pairwise energy function.
                        must be of the form f(x, *args).
    :type energy: callable
    :param args: arguments to pass to the function

    :return: gradient of the configuration
    :rtype: numpy.ndarray, same shape as x
    """

    particles = x.shape[0]

    left_indices, right_indices = np.triu_indices(particles, k=1)  # get indices of upper triangle minus diagonal
    r_1_2 = x[right_indices] - x[left_indices]  # vectors between all unique particle pairs
    r_squared = (r_1_2 * r_1_2).sum(1, keepdims=True)
    r = np.sqrt(r_squared)

    return energy(r, *args).sum()

def pair_energy_period(x, square_size=1.0, energy=None,  args=()):
    """

    Calculates the potential energy of a configuration of particles.

    :param x: positions of the particles
    :type x: numpy.ndarray
    :param energy: the pairwise energy function.
                        must be of the form f(x, *args).
    :type energy: callable
    :param args: arguments to pass to the function

    :return: gradient of the configuration
    :rtype: numpy.ndarray, same shape as x
    """

    particles = x.shape[0]
    list = [[0,0],[square_size,0],[-square_size,0],[0,square_size],[0,-square_size]]
    r = []
    left_indices, right_indices = np.triu_indices(particles, k=1)  # get indices of upper triangle minus diagonal
    r_1_2 = x[right_indices] - x[left_indices]
    r_1_2[:] = r_1_2[:] % square_size
    r_1_2[:] = np.where(r_1_2 > square_size / 2, r_1_2 - square_size, r_1_2)#rcut < square_size / 2, calculate the interaction where r < square_size / 2
    r_squared = (r_1_2 * r_1_2).sum(1, keepdims=True)
    r = np.sqrt(r_squared)



    return energy(r, *args).sum()

def lennard_jones_energy(r, sigma=1.0, epsilon=1.0, cut_off_distance=2.5):
    """

    Calculates the Lennard-Jones energy for particles with diameter sigma
    at a separation r with a well-depth epsilon.

    >>> lennard_jones_energy(1.0, 1.0, 1.0)
    0.0

    >>> lennard_jones_energy(2**(1/6), 1.0, 1.0)
    -1.0

    >>> lennard_jones_energy(0.0, 1.0, 1.0)
    Traceback (most recent call last):
    AssertionError: separation between particles is <= 0

    >>> lennard_jones_energy(1.0, -1.0, 1.0)
    Traceback (most recent call last):
    AssertionError: particle diameter is not strictly positive

    :param r: the distance between two particles
    :type r: float or numpy.ndarray
    :param sigma: the diameter of a particle
    :type sigma: float or numpy.ndarray
    :param epsilon: the well depth of the potential
    :type epsilon: float or numpy.ndarray
    :param cut_off_distance: distance to cut off the potentoa;
    :type cut_off_distance: float

    :return: the Lennard-Jones energy of the particle pair(s)
    :rtype: float or numpy.ndarray
    """

    assert np.all(r > 0.0), "separation between particles is <= 0"
    assert sigma > 0.0, "particle diameter is not strictly positive"

    r6 = (sigma / r)
    r_cutoff_6 = (sigma / cut_off_distance)
    r6 *= r6
    r6 *= r6 * r6
    r_cutoff_6 *= r_cutoff_6
    r_cutoff_6 *= r_cutoff_6 * r_cutoff_6

    return np.where(r >= cut_off_distance, 0.0, 4 * epsilon * r6 * (r6 - 1) - 4 * epsilon * r_cutoff_6 * (r_cutoff_6 - 1)).sum()

def pair_lennard_jones_energy(x,square_size, rcut, density):
    return pair_energy(x, energy=lennard_jones_energy, args=(1.0, 1.0, rcut))

def pair_lennard_jones_energy_period(x,square_size,rcut, density):
    etail =  math.pi * density * (0.4 * rcut ** -10 -rcut ** -4)
    return pair_energy_period(x, square_size, energy=lennard_jones_energy, args=(1.0, 1.0, rcut)) - etail

def lennard_jones_gradient(r, sigma=1.0, epsilon=1.0,cut_off_distance=2.5):
    """

    Calculates the Lennard-Jones gradient for particles with diameter sigma
    at a separation r with a well-depth epsilon.

    >>> lennard_jones_gradient(1.0, 1.0, 1.0)
    -24.0

    >>> abs(lennard_jones_gradient(2**(1/6), 1.0, 1.0)) < 1e-14
    True

    >>> lennard_jones_gradient(0.0, 1.0, 1.0)
    Traceback (most recent call last):
    AssertionError: separation between particles is <= 0

    >>> lennard_jones_gradient(1.0, -1.0, 1.0)
    Traceback (most recent call last):
    AssertionError: particle diameter is not strictly positive

    :param r: the distance between two particles
    :type r: float or numpy.ndarray
    :param sigma: the diameter of a particle
    :type sigma: float or numpy.ndarray
    :param epsilon: the well depth of the potential
    :type epsilon: float or numpy.ndarray

    :return: the Lennard-Jones gradient of the particle pair(s)
    :rtype: float or numpy.ndarray
    """

    #assert np.all(r > 0.0), "separation between particles is <= 0"
    #assert sigma > 0.0, "particle diameter is not strictly positive"

    r6 = (sigma / r)
    r6 *= r6
    r6 *= r6 * r6

    return np.where(r > cut_off_distance, 0.0, 24 * epsilon * r6 * (1 - 2 * r6) / r)

def virialcoff_period(x,square_size, rcut, gradient=lennard_jones_gradient,args=()):

    particles = x.shape[0]
    dimension = x.shape[1]

    g = np.zeros([particles, particles, dimension])

    if gradient is None:
        return g.sum(1)

    left_indices, right_indices = np.triu_indices(particles, k=1)  # get indices of upper triangle minus diagonal
    r_1_2 = x[right_indices] - x[left_indices]  # vectors between all unique particle pairs
    r_1_2[:] = r_1_2[:] % square_size
    r_1_2[:] = np.where(r_1_2 > square_size / 2, r_1_2 - square_size, r_1_2)#rcut < square_size / 2, calculate the interaction where r < square_size / 2
    r_squared = (r_1_2 * r_1_2).sum(1, keepdims=True)
    r = np.sqrt(r_squared)

    grad = gradient(r,  *args) * r_1_2 / r
    virial = -grad * r_1_2
    return virial.sum() / 2

def radial_distribution_period(x, square_size, rcut, parts=20):
    particles = x.shape[0]
    dimension = x.shape[1]

    g = np.zeros(parts + 1)
    left_indices, right_indices = np.triu_indices(particles, k=1)  # get indices of upper triangle minus diagonal
    r_1_2 = x[right_indices] - x[left_indices]  # vectors between all unique particle pairs
    r_1_2[:] = r_1_2[:] % square_size
    r_1_2[:] = np.where(r_1_2 > square_size / 2, r_1_2 - square_size, r_1_2)
    r_squared = (r_1_2 * r_1_2).sum(1, keepdims=True)

    r_squared.tolist()
    for i in range(len(r_squared)):
        if r_squared[i] < square_size / 2:

            ig = int(r_squared[i])
            g[ig] += 2
    return g








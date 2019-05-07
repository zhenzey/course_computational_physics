import numpy as np
import random
import math
import time
from lj_potential import pair_lennard_jones_energy
from lj_potential import  pair_lennard_jones_energy_period
from lj_potential import virialcoff_period
from lj_potential import radial_distribution_period
import matplotlib.pyplot as plt


temp = 1.0
def posi_particles(square_size, sqrt_particles):
    list = []
    particles = sqrt_particles ** 2
    period = square_size / (sqrt_particles - 1)/1.1
    for j in range(particles):
        list.append(((j % sqrt_particles) + 1) * period)
        list.append(((j // sqrt_particles) + 1) * period)
    return np.array(list).reshape(-1,2)


def random_move(x,step_size = 0.1):
    return x + step_size * (0.5 - np.random.rand(*x.shape))

def boundry_condition(type,square_size, x0=np.array([]), x=np.array([])):
    """

    :param type: boundary condition type
    :param x0: last configuration
    :param x: current configuration
    :param square_size: the length of the box
    :return:
    """
    if type == "HW":
        if np.any(abs(x - square_size / 2) > square_size / 2 ):
            return x0
        else:

            return x
    elif type == "PBC":
        if np.any(abs(x - square_size / 2) > square_size / 2):
            return x % square_size
        else:
            return x


def monte_carlo(x, move_f, energy_f, sample_f, iterations, sample_frequency, type, square_size, rcut, density):
    assert sample_frequency > 0, "sample frequency is negative"

    sum_statistics = 0.0
    number = 0.0
    list = []
    for iteration in range(iterations):
        y = boundry_condition(type, square_size, x, move_f(x))
        last_energy = energy_f(x,square_size,rcut,density)
        current_energy = energy_f(y,square_size,rcut,density)
        if current_energy < last_energy or random.random() < math.exp((last_energy - current_energy) / temp):
            x = y
        if iteration % sample_frequency == 0 and iteration > 0:
            sum_statistics += sample_f(x,square_size,rcut,density)
            number += 1

    return sum_statistics / number

def sample_pressure(x, square_size, rcut, density):
    p= density * temp + virialcoff_period(x,square_size,rcut,args=(1.0,1.0,rcut))/(square_size**2)
    rcut3 = rcut**(-3)
    p_tail = 16 * np.pi * (density**2) * (2 * rcut3**3 /3 - rcut3)
    return p + p_tail

def sample_gr(x, square_size, rcut, density):
    g = radial_distribution_period(x,square_size,rcut,100)
    parts = g.shape[0]
    d_volume = np.pi * ((np.arange(parts) + 1) ** 2 - np.arange(parts) ** 2) * (square_size / 2 /parts) ** 2
    return g / d_volume / density




def main():
    # Q 1(a),(b)

    efile = open("energy-N.dat", "w")
    efile.write("# average energy\n")
    efile.write("energy particles HW PBC\n")
    for i in range(4, 34):
        density = 0.2
        particles = i ** 2
        square_size = math.sqrt(particles / density)
        x = posi_particles(square_size, i)
        total_energy = monte_carlo(x, random_move, pair_lennard_jones_energy, pair_lennard_jones_energy, 10000, 50,
                                   "HW", square_size, 2.5, 0.2)
        efile.write('%10.2f' % (i ** 2))
        efile.write('%10.2f ' % (total_energy / i / i))
        total_energy = monte_carlo(x, random_move, pair_lennard_jones_energy_period, pair_lennard_jones_energy_period,
                                   10000, 50, "PBC", square_size, 2.5, 0.2)
        efile.write('%10.2f\n' % (total_energy / i / i))
        print(i ** 2, total_energy / i / i)
    efile.close()

    # Q 2
    efile = open("energy-rcut.dat", "w")
    efile.write("# average energy\n")
    efile.write("energy rcut\n")
    density = 0.2
    particles = 100
    square_size = math.sqrt(particles / density)
    x = posi_particles(square_size, int(math.sqrt(particles)))
    for i in np.logspace(-4, -1, 2):
        rcut = i * square_size
        total_energy = monte_carlo(x, random_move, pair_lennard_jones_energy_period, pair_lennard_jones_energy_period,
                                   10000, 50, "PBC", square_size, rcut, 0.2)
        efile.write('%10.2f ' % rcut)
        efile.write('%10.2f\n' % (total_energy / particles))
        print(rcut, total_energy / particles)
    efile.close()

    # Q 3
    pfile = open("pressure-density.dat", "w")
    pfile.write("# average pressure\n")
    pfile.write("density pressure\n")
    for n in np.linspace(22, 23, 1):
        t0 = time.time()
        for density in np.linspace(0.1, 0.9, 9):
            n = int(n)
            square_size = math.sqrt(n ** 2 / density)
            x = posi_particles(square_size, n)
            pressure = monte_carlo(x, random_move, pair_lennard_jones_energy_period, sample_pressure, 1000, 50, "PBC",
                                   square_size, 2.5, 0.2)
            pfile.write('%10.2f ' % density)
            pfile.write('%10.2f\n' % pressure)
            print(n ** 2, density, pressure)
        t1 = time.time()
        print(n ** 2, t1 - t0)
        print()
    # Q 4
    gfile = open("g(r)-r.dat", "w")
    gfile.write("# density distribution\n")
    gfile.write("density g(r) r\n")
    for n in np.linspace(20, 21, 1):
        t0 = time.time()
        for density in np.logspace(-2, 2, 5):
            n = int(n)
            square_size = math.sqrt(n ** 2 / density)
            x = posi_particles(square_size, n)
            g = monte_carlo(x, random_move, pair_lennard_jones_energy_period, sample_gr, 1000, 50, "PBC", square_size,
                            2.5, 0.2)
            gfile.write('%10.2f ' % density)
            gfile.write('%10.2str\n' % g)
            print(n ** 2, density, g)

        t1 = time.time()
        print(n ** 2, t1 - t0)
        print()

    gfile.close()

    t1 = time.time()
    print(n ** 2, t1 - t0)
    print()


if __name__ == "__main__":
    main()



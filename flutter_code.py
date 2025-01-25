#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:22:50 2024

@author: awang104-2, Nithin Kishore Nallathambi, jtindall
"""

import numpy as np
from scipy.special import hankel2
import matplotlib.pyplot as plt


# Input parameters
b = 1
omega_h = 0.4
omega_alpha = 1
omega_h_normalized = omega_h / omega_alpha  # Natural plunge frequency
x_alpha = 0.1  # Center of mass location
r_alpha = 1 / 2  # Radius of gyration
mu = 3  # Mass ratio
a = -0.4  # Elastic axis location


def broydens_method(function, x0, max_iterations=1000, show_text=False):
    error = 1
    iterations = 0
    determinant = lambda x: np.linalg.det(function(x))

    xvec = x0
    Kmat = np.identity(2)
    tmpval = determinant(xvec)
    Fvec = np.array([tmpval.real, tmpval.imag])

    while (error > 10**-5) and (iterations < max_iterations):
        if show_text:
            print('Iterations:', iterations)
            print('Values:', xvec)
            print('Determinant:', tmpval)
        svec = np.linalg.solve(Kmat, -Fvec)
        xvec = xvec + 0.02 * svec
        Fvecold = Fvec
        tmpval = determinant((xvec[0], xvec[1]))
        Fvec = np.array([tmpval.real, tmpval.imag])
        yvec = Fvec - Fvecold
        Kmat = Kmat + 0.02 * np.outer(yvec - Kmat @ svec, svec) / np.linalg.norm(svec)**2
        error = np.linalg.norm(svec)
        iterations += 1

    if iterations >= max_iterations:
        print("Did not converge.")
        return np.array([np.nan, np.nan])

    return xvec


def F(omega, U):
    V = U / b
    M = np.array([
        [mu * (1 - omega_h_normalized ** 2 / omega ** 2) + Lh(omega / V), x_alpha * mu + La(omega / V)],
        [x_alpha * mu + Mh(omega / V) - Lh(omega / V) * (1 / 2 + a), r_alpha**2 * mu * (1 - 1/omega**2) + Ma(omega / V) - (La(omega / V) + 1 / 2) * (1 / 2 + a) + Lh(omega / V) * (1 / 2 + a)**2]
    ])
    return M


# Theodorsen's function (C(k))
def theodorsen_function(k):
    H1 = hankel2(1, k)
    H0 = hankel2(0, k)
    return H1 / (H1 + 1j * H0)


# Aerodynamic coefficients
def Lh(k):
    return 1 - 2j * theodorsen_function(k) / k
    # return 2 * np.pi * (1 + 1j * k * Ck)


def La(k):
    return 1 / 2 - 1j / k * (1 + 2 * theodorsen_function(k)) - 2 * theodorsen_function(k) / k**2
    # return 2 * np.pi * (1 + 1j * k / 2 * Ck)


def Ma(k):
    return 3 / 8 - 1j / k


def Mh(k):
    return 1 / 2


x_alpha = 0.1
omega_hs = np.linspace(0.1, 2, 50)
list_of_speeds = []
for i in range(50):
    omega_h_normalized = omega_hs[i]
    initial_guess = np.array([1, 1])
    function = lambda x: F(x[0], x[1])
    solutions = broydens_method(function, initial_guess)
    omega, U = solutions
    list_of_speeds.append(U)
plt.plot(omega_hs, list_of_speeds)

x_alpha = 0.2
omega_hs = np.linspace(0.1, 2, 50)
list_of_speeds = []
for i in range(50):
    omega_h_normalized = omega_hs[i]
    initial_guess = np.array([1, 1])
    function = lambda x: F(x[0], x[1])
    solutions = broydens_method(function, initial_guess)
    omega, U = solutions
    list_of_speeds.append(U)
plt.plot(omega_hs, list_of_speeds)
plt.show()


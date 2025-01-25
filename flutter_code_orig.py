#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:22:50 2024

@author: jtindall
"""

import numpy as np
import scipy

# Define the F-matrix function
def aeroelastic_F(U, omega, omega_h, omega_alpha, x_alpha, r_alpha, mu, a):
    """
    Compute the aeroelastic determinant F(U, omega).
    """
    # Reduced frequency
    k = omega * b / U

    # Aerodynamic coefficients (Lh, L_alpha, M_alpha)
    L_h = 2 * np.pi + 2 * np.pi * k
    L_alpha = 2 * np.pi + 2 * np.pi * k
    M_alpha = 2 * np.pi + 2 * np.pi * k

    # Frequency terms
    omega_ratio_h = omega_h / omega
    omega_ratio_alpha = omega_alpha / omega

    # Construct the aeroelastic F matrix
    F11 = (mu * (1 - (omega_h**2)/(omega**2))) + L_h
    F12 = x_alpha * mu + (L_alpha - L_h * (0.5+ a))
    F21 = x_alpha * mu + (0.5 - L_h * (0.5 + a))
    F22 = (r_alpha**2* mu * (1 - (omega_h**2)/(omega**2))) + M_alpha - (L_alpha + 0.5) * (0.5 + a) + L_h * ((0.5 + a)**2)

    # Assemble the matrix
    F = np.array([[F11, F12], [F21, F22]])
    

    # Return determinant of F
    return np.linalg.det(F)


# Custom Broyden's method implementation
def broydens_method(F_func, x0, tol=1e-2, max_iter=500, regularization=1e-3, verbose=True):
    """
    Custom implementation of Broyden's method with diagnostics and regularization.
    """
    x = np.array(x0, dtype=float)
    B = np.eye(len(x))  # Initial Jacobian approximation
    F_x = np.array(F_func(x))

    for iteration in range(max_iter):
        if np.linalg.norm(F_x) < tol:
            if verbose:
                print(f"Converged in {iteration} iterations")
            return x

        # Regularize the Jacobian to avoid singularity
        B_reg = B + np.eye(B.shape[0]) * regularization

        try:
            dx = np.linalg.solve(B_reg, -F_x)
        except np.linalg.LinAlgError:
            raise ValueError("Regularized Jacobian is still singular.")

        x_new = x + dx
        F_x_new = np.array(F_func(x_new))
        
        delta_x = x_new - x
        delta_F = F_x_new - F_x

        if np.linalg.norm(delta_x) < 1e-10:
            raise ValueError("Iterations are stagnating. Check the initial guess or scaling.")

        if verbose:
            print(f"Iter {iteration}: x = {x}, F_x = {F_x}, det(F_x) = {np.linalg.norm(F_x)}")
            print(f"delta_x = {delta_x}, delta_F = {delta_F}")

        if np.linalg.norm(delta_x) > 1e-12:
            B += np.outer((delta_F - B @ delta_x), delta_x) / (np.dot(delta_x, delta_x) + 1e-8)

        x = x_new
        F_x = F_x_new

    raise ValueError("Broyden's method did not converge within the maximum number of iterations")





# Wrapper to solve for flutter speed and frequency
def flutter_solver(omega_h, omega_alpha, x_alpha, r_alpha, mu, a, initial_guess):
    """
    Solve for flutter speed (U_F) and flutter frequency (omega_F) using Broyden's method.
    """
    def F_root(x):
        U, omega = x
        det_F = aeroelastic_F(U, omega, omega_h, omega_alpha, x_alpha, r_alpha, mu, a)
        return [det_F.real, det_F.imag]  # Ensure consistent 2-element output

    solution = broydens_method(F_root, initial_guess)
    return solution

def F_root(x):
    U, omega = x
    det_F = aeroelastic_F(U, omega, omega_h, omega_alpha, x_alpha, r_alpha, mu, a)
    return det_F


# Constants for the airfoil
b = 1.0  # Semi-chord length (normalized)
rho = 1.0  # Air density (normalized)

# Input parameters
omega_h = 1  # Natural plunge frequency
omega_alpha = 1  # Natural pitch frequency
x_alpha = 0.2  # Center of mass location
r_alpha = 0.3  # Radius of gyration
mu = 0.333  # Mass ratio
a = -0.4  # Elastic axis location
initial_guess = [0.8, 1.0]  # Initial guess for [U, omega]


#print(scipy.optimize.newton(F_root, initial_guess))


'''
for U_test in np.linspace(1, 50, 10):
    for omega_test in np.linspace(0.1, 10, 10):
        det = aeroelastic_F(U_test, omega_test, omega_h, omega_alpha, x_alpha, r_alpha, mu, a)
        print(det)
'''


print(np.linalg.norm([1, 2])**2)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 10:44:05 2026

@author: gracetait
"""

import numpy as np
from matplotlib import pyplot as plt
import os
import argparse
import scipy
from numba import njit

@njit
def gauss_seidel_sweep_numba(phi, rho, l):
    """
    Performs one Gauss-Seidel sweep in 3D.
    Numba handles the triple nested loops at C-speed.
    """
    old_phi = phi.copy()
    for i in range(1, l - 1):
        for j in range(1, l - 1):
            for k in range(1, l - 1):
                phi[i, j, k] = (phi[i - 1, j, k] + phi[i + 1, j, k] +
                                phi[i, j - 1, k] + phi[i, j + 1, k] +
                                phi[i, j, k - 1] + phi[i, j, k + 1] +
                                rho[i, j, k]) / 6.0
    return np.abs(phi - old_phi)

@njit
def sor_sweep_numba(phi, rho, l, omega):
    """
    Performs one SOR sweep in 3D.
    Directly modifies phi in-place for memory efficiency.
    """
    old_phi = phi.copy()
    for i in range(1, l - 1):
        for j in range(1, l - 1):
            for k in range(1, l - 1):
                # Standard GS update
                gs = (phi[i - 1, j, k] + phi[i + 1, j, k] +
                      phi[i, j - 1, k] + phi[i, j + 1, k] +
                      phi[i, j, k - 1] + phi[i, j, k + 1] +
                      rho[i, j, k]) / 6.0
                
                # Over-relaxation step
                phi[i, j, k] = (1.0 - omega) * phi[i, j, k] + omega * gs
                
    return np.abs(phi - old_phi)

@njit
def sor_sweep_2d_numba(phi, rho, l, omega):
    """
    Performs one SOR sweep on a 2D lattice.
    Much more efficient for infinite wire geometries.
    """
    old_phi = phi.copy()
    for i in range(1, l - 1):
        for j in range(1, l - 1):
            # 2D update logic (4 neighbors instead of 6)
            gs = (phi[i - 1, j] + phi[i + 1, j] +
                  phi[i, j - 1] + phi[i, j + 1] +
                  rho[i, j]) / 4.0
            
            phi[i, j] = (1.0 - omega) * phi[i, j] + omega * gs
                
    return np.abs(phi - old_phi)

class Poisson(object):
    """
    A class that implements the non-dimensionised 3D Poisson equation on a cubic 
    lattice.
    """

    def __init__(self, l, tolerance, omega, init):
        """
        Initialise the Poisson solver.

        Parameters
        ----------
        l : int
            Side length of the cubic lattice.
        tolerance : float
            Convergence threshold.
        omega : float
            SOR relaxation parameter.
        init : {"Monopole", "Wire"}
            Charge distribution to use:
            - "Monopole" — single unit charge at the lattice centre.
            - "Wire" — line of unit charges along the central z-axis.

        Returns
        -------
        None.

        """

        # Define parameters
        self.l = l
        self.omega = omega
        self.tolerance = tolerance
        self.init_type = init
        
        if init == "Monopole":
            
            self.phi = self.init_phi_3d()
            self.rho = self.init_rho_monopole()
            
        if init == "Wire":
            
            self.phi = self.init_phi_2d()
            self.rho = self.init_rho_wire()

    def boundary_conditions(self, phi):
        """
        Enforce Dirichlet (zero) boundary conditions on all six faces.

        Parameters
        ----------
        phi : np.ndarray of shape (l, l, l)
            Potential field to modify in-place.
 
        Returns
        -------
        np.ndarray of shape (l, l, l)
            The same array with boundary values set to zero.

        """

        # 3D case
        if phi.ndim == 3:
            # Set boundaries to be zero
            phi[0, :, :], phi[-1, :, :] = 0, 0
            phi[:, 0, :], phi[:, -1, :] = 0, 0
            phi[:, :, 0], phi[:, :, -1] = 0, 0
            
        # 2D case
        else:
            phi[0, :], phi[-1, :] = 0, 0
            phi[:, 0], phi[:, -1] = 0, 0

        return phi

    def init_phi_3d(self):
        """
        Initialise the potential field with uniform random noise and set the 
        boundaries to be zero.
        3D case

        Returns
        -------
        np.ndarray of shape (l, l, l)
            Randomly initialised potential field with zero boundaries.

        """

        # Initialise lattice to have some random noise between 0 and 1
        phi = np.random.rand(self.l, self.l, self.l)

        # Assign to self.phi and set boundaries to be zero
        return self.boundary_conditions(phi)
    
    def init_phi_2d(self):
        """
        Initialise the potential field with uniform random noise and set the 
        boundaries to be zero.
        2D case

        Returns
        -------
        np.ndarray of shape (l, l)
            Randomly initialised potential field with zero boundaries.

        """

        # Initialise lattice to have some random noise between 0 and 1
        phi = np.random.rand(self.l, self.l)

        # Assign to self.phi and set boundaries to be zero
        return self.boundary_conditions(phi)

    def init_rho_monopole(self):
        """
        Initialise the charge density as a point monopole at the lattice centre.

        Returns
        -------
        np.ndarray of shape (l, l, l)
            Charge density array with a single non-zero entry at the centre.

        """

        # Initiate rho as a monopole
        rho = np.zeros(shape=(self.l, self.l, self.l))
        rho[self.l // 2, self.l // 2, self.l // 2] = 1

        return rho
    
    def init_rho_wire(self):
        """
        Initialise the charge density as an infinite wire along the z-axis.
        In 2D due to symmetry 

        Returns
        -------
        rnp.ndarray of shape (l, l, l)
            Charge density array with a line of unit values along the z-axis.

        """
        
        # Initiate rho as a monopole
        # In 2D since due to symmetry 
        rho = np.zeros(shape=(self.l, self.l))
        rho[self.l // 2, self.l // 2] = 1

        return rho

    def jacobi(self):
        """
        Performs one Jacobi iteration over the entire lattice.

        Returns
        -------
        np.ndarray of shape (l, l, l)
            Absolute change |φ_new − φ_old| at every grid point.

        """

        # Calculate new phi, taking into consideration periodic boundaries
        if self.phi.ndim == 3:
            # 3D update (6 neighbors)
            new_phi = (np.roll(self.phi, 1, axis=0) + np.roll(self.phi, -1, axis=0) +
                       np.roll(self.phi, 1, axis=1) + np.roll(self.phi, -1, axis=1) +
                       np.roll(self.phi, 1, axis=2) + np.roll(self.phi, -1, axis=2) +
                       self.rho) / 6.0
        else:
            # 2D update (4 neighbors)
            new_phi = (np.roll(self.phi, 1, axis=0) + np.roll(self.phi, -1, axis=0) +
                       np.roll(self.phi, 1, axis=1) + np.roll(self.phi, -1, axis=1) +
                       self.rho) / 4.0

        # Set boundaries to be zero
        new_phi = self.boundary_conditions(new_phi)

        # Calculate distance between old and new phi
        distance = np.abs(self.phi - new_phi)

        # Update phi 
        self.phi = new_phi

        return distance

    def gauss_seidel(self):
        """
        Perform one Gauss-Seidel sweep over the entire lattice using Numba logic.

        Returns
        -------
        np.ndarray of shape (l, l, l)
            Absolute change |φ_new − φ_old| at every grid point.

        """

        # Ensure array is float64 for Numba
        distance = gauss_seidel_sweep_numba(self.phi.astype(np.float64), 
                                            self.rho.astype(np.float64), 
                                            self.l)
        
        return distance

    def sor(self):
        """
        Perform one SOR sweep in 2D or 3D using Numba logic.

        Returns
        -------
        np.ndarray of shape (l, l, l)
            Absolute change |φ_new − φ_old| at every grid point.

        """

        # Convert to float64 to ensure Numba compatibility
        phi_f64 = self.phi.astype(np.float64)
        rho_f64 = self.rho.astype(np.float64)
        
        if self.phi.ndim == 3:
            distance = sor_sweep_numba(phi_f64, rho_f64, self.l, self.omega)
        else:
            distance = sor_sweep_2d_numba(phi_f64, rho_f64, self.l, self.omega)
            
        self.phi = phi_f64 # Update class field
        return distance


    def get_electric_field(self):
        """
        Compute the electric field from the converged potential.

        Returns
        -------
        E_x : np.ndarray of shape (l, l, l)
            x-cartesian component of the electric field.
        E_y : np.ndarray of shape (l, l, l)
            y-cartesian component of the electric field.
        E_z : np.ndarray of shape (l, l, l)
            z-cartesian component of the electric field.

        """
        
        E_x = -(np.roll(self.phi, -1, axis=0) - np.roll(self.phi, 1, axis=0)) / 2
        E_y = -(np.roll(self.phi, -1, axis=1) - np.roll(self.phi, 1, axis=1)) / 2
        E_z = -(np.roll(self.phi, -1, axis=2) - np.roll(self.phi, 1, axis=2)) / 2
        
        return E_x, E_y, E_z
    
    def get_magnetic_field(self):
        """
        Compute the magnetic field from the converged vector potential.

        Returns
        -------
        B_x : np.ndarray of shape (l, l, l)
            x-cartesian component of the magnetic field.
        B_y : np.ndarray of shape (l, l, l)
            y-cartesian component of the magnetic field.
        B_z : np.ndarray of shape (l, l, l)
            z-cartesian component of the magnetic field.

        """
        
        grad_x = (np.roll(self.phi, -1, axis=0) - np.roll(self.phi, 1, axis=0)) / 2
        grad_y = (np.roll(self.phi, -1, axis=1) - np.roll(self.phi, 1, axis=1)) / 2
        
        # B = curl(A), with A = (0, 0, phi)
        # B_z = 0 since A_z has no variation along z for an infinite wire
        return grad_y, -grad_x, np.zeros_like(self.phi)


class Simulation(object):
    """
    A class to handle the execution, measurement, and visualisation 
    of the Poisson simulation.
    """
    
    def __init__(self, l, tolerance, omega):
        """
        Initialise the simulation parameters.

        Parameters
        ----------
        l : int
            Side length of the cubic lattice (l × l × l grid points).
        tolerance : float
            Convergence threshold.
        omega : float
            SOR relaxation parameter.

        Returns
        -------
        None.

        """

        # Define parameters
        self.l = l
        self.omega = omega
        self.tolerance = tolerance

    def electric_measurements(self, alg, filename):
        """
        Solve the Poisson equation for a monopole and save the midplane field data.

        Parameters
        ----------
        alg : {"j", "gs", "sor"}
            Iterative algorithm to use: Jacobi, Gauss-Seidel, or SOR.
        filename : str
            Output data filename (written under ``outputs/datafiles/``).

        Returns
        -------
        None.

        """
        
        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        datafiles_folder = os.path.join(outputs_directory, "datafiles")
        file_path = os.path.join(datafiles_folder, filename)
        
        # If the folders don't exist, create them
        if not os.path.exists(datafiles_folder):
            os.makedirs(datafiles_folder)
            
        # Initialise Poisson class
        poisson = Poisson(self.l, self.tolerance, self.omega, init = "Monopole")

        # Define algorithm
        if alg == "j":
            update = poisson.jacobi

        elif alg == "gs":
            update = poisson.gauss_seidel

        else:
            update = poisson.sor

        # Update phi and obtain the distance between old and new phi
        distance = update()

        # Let the error be the largest distance
        error = np.max(distance)

        # Set time to zero
        t = 0

        # Continue to update phi until the error is smaller than the tolerance
        while error > self.tolerance:
            print(f"Simulating step = {t}", end = '\r')

            # Update the animation and obtain the error
            distance = update()
            error = np.max(distance)
            t += 1
        
        # Obtain the electric field
        E_x, E_y, E_z = poisson.get_electric_field()
        E = np.sqrt(E_x**2 + E_y**2 + E_z**2)
        
        # Extract the midplanes
        phi_midplane = poisson.phi[:, :, self.l // 2]
        E_x_midplane = E_x[:, :, self.l // 2]
        E_y_midplane = E_y[:, :, self.l // 2]
        E_z_midplane = E_z[:, :, self.l // 2]
        E_midplane = E[:, :, self.l // 2]
        
        # Open in "a" (append) or "w" (overwrite) mode
        # Write the values into the file
        with open(file_path, "w") as f:
            
            f.write("x,y,r,E_x,E_y,E_z,E,phi\n")

            # Iterate through x and y coordiantes
            for x in range(self.l):
                for y in range(self.l):
                    
                    # Calculate the distance from the central charge
                    r = np.sqrt((x - self.l // 2)**2 + (y - self.l // 2)**2)
                    
                    # Obtain the electric field value
                    E_x_val = E_x_midplane[x,y]
                    E_y_val = E_y_midplane[x,y]
                    E_z_val = E_z_midplane[x,y]
                    
                    # Obtain the potential and total electric field magnitude
                    phi_val = phi_midplane[x, y]
                    E_val = E_midplane[x, y]
                       
                    # Write to the file
                    f.write(f"{x},{y},{r},{E_x_val},{E_y_val},{E_z_val},{E_val},{phi_val}\n")
                    
    def magnetic_measurements(self, alg, filename):
        """
        Solve the Poisson equation for a wire and save midplane field data.

        Parameters
        ----------
        alg : {"j", "gs", "sor"}
            Iterative algorithm to use: Jacobi, Gauss-Seidel, or SOR.
        filename : str
            Output data filename (written under ``outputs/datafiles/``).

        Returns
        -------
        None.

        """
        
        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        datafiles_folder = os.path.join(outputs_directory, "datafiles")
        file_path = os.path.join(datafiles_folder, filename)
        
        # If the folders don't exist, create them
        if not os.path.exists(datafiles_folder):
            os.makedirs(datafiles_folder)
            
        # Initialise Poisson class
        poisson = Poisson(self.l, self.tolerance, self.omega, init = "Wire")

        # Define algorithm
        if alg == "j":
            update = poisson.jacobi

        elif alg == "gs":
            update = poisson.gauss_seidel

        else:
            update = poisson.sor

        # Update phi and obtain the distance between old and new phi
        distance = update()

        # Let the error be the largest distance
        error = np.max(distance)

        # Set time to zero
        t = 0

        # Continue to update phi until the error is smaller than the tolerance
        while error > self.tolerance:
            print(f"Simulating step = {t}", end = '\r')

            # Update the animation and obtain the error
            distance = update()
            error = np.max(distance)
            t += 1
        
        # Obtain the electric field
        M_x, M_y, M_z = poisson.get_magnetic_field()
        M = np.sqrt(M_x**2 + M_y**2 + M_z**2)
        
        # Extract the midplanes but we don't need to slice since it is 2D
        phi_2d = poisson.phi
        M_x_2d = M_x
        M_y_2d = M_y
        M_z_2d = M_z
        M_2d   = M
        
        # Open in "a" (append) or "w" (overwrite) mode
        # Write the values into the file
        with open(file_path, "w") as f:
            
            f.write("x,y,r,M_x,M_y,M_z,M,phi\n")

            # Iterate through x and y coordiantes
            for x in range(self.l):
                for y in range(self.l):
                    
                    # Calculate the distance from the wire
                    r = np.sqrt((x - self.l // 2)**2 + (y - self.l // 2)**2)
                    
                    # Obtain the magnetic field values
                    M_x_val = M_x_2d[x,y]
                    M_y_val = M_y_2d[x,y]
                    M_z_val = M_z_2d[x,y]
                    
                    # Obtain the potential and total magnetic field magnitude
                    phi_val = phi_2d[x, y]
                    M_val = M_2d[x, y]
                        
                    # Write to the files
                    f.write(f"{x},{y},{r},{M_x_val},{M_y_val},{M_z_val},{M_val},{phi_val}\n")
                    
    def f(self, x, A, B):
        """
        Linear model used for curve fitting.

        Parameters
        ----------
        x : array-like
            Independent variable.
        A : float
            Slope.
        B : float
            Intercept.
 
        Returns
        -------
        np.ndarray
            Evaluated model values "A * x + B".

        """
        
        return A * x + B
                    
    def plot_field_vs_distance_measurements(self, filename, field_type):
        """
        Generate and save a plot of the field magnitude vs radial distance on 
        a log-log scale with a power-law fit.

        Parameters
        ----------
        filename : str
            Name of the data file under ``outputs/datafiles/``.
        field_type : {"Electric", "Magnetic"}
            Used to label plot axes and titles.

        Returns
        -------
        None.

        """
        
        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        filename_path = os.path.join(outputs_directory, "datafiles", filename)
        plots_folder = os.path.join(outputs_directory, "plots")
        
        # If the folders dont exist, create them
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)

        # Create an empty list to store input data
        input_data = []        

        # Read in the data from the specified text file
        try:
            with open(filename_path, "r") as filein:
                for line in filein:
                    input_data.append(line.strip("\n").split(","))
                    
        # If text file cannot be found, print error
        except FileNotFoundError:
            print(f"Error: Could not find {filename_path}")
            return 
        
        # Convert input data into a np array
        input_data = np.array(input_data[1:], dtype = float)
        
        # Collect the input data
        field = input_data[:, 6]
        r = input_data[:, 2]
        
        # Filter out zeros to allow log plotting
        mask1 = (r > 0) & (field > 0)
        r_masked = r[mask1]
        field_masked = field[mask1]
        
        # Transform into log space for linear fit
        ln_r = np.log(r_masked)
        ln_field = np.log(field_masked)
        
        # Filter out the data near the center and boundaries
        mask2 = (r_masked > 3) & (r_masked < self.l // 6)
        
        # Create empty plots
        fig, ax = plt.subplots(1, 1, figsize=(8, 10))
        
        # Plot the data
        popt, pcov = scipy.optimize.curve_fit(self.f, ln_r[mask2], ln_field[mask2])
        ax.plot(ln_r, ln_field, 'yo', markersize=2, label='Data')
        ax.plot(ln_r, self.f(ln_r, *popt), 
                  'k-', label=f'Fit: {popt[0]:.3f}ln(r) {popt[1]:+.3f}')
        ax.legend(loc="upper right")
        
        # Set titles
        ax.set_title(
    rf"{field_type} Field of the z-axis midplane vs Distance"
    "\n"
    rf"{self.l} x {self.l} lattice with {self.tolerance} accuracy", 
    fontsize=16
)
        ax.set_xlabel("ln(Distance)", fontsize = 14)
        ax.set_ylabel(rf"ln($|{field_type} field|$)", fontsize = 14)
        
        # Save the plots to the plots folder
        save_filename = filename.replace(".txt", "field_plot.png")
        save_path = os.path.join(plots_folder, save_filename)
        plt.savefig(save_path, dpi = 300)
        
        # Print message
        print(f"Plots successfully saved to: {save_path}")
        
        # Show final plot
        plt.show()
        
    def plot_potential_vs_distance_measurements(self, filename, field_type):
        """
        Generate and save a plot of the field potential vs radial distance.

        Parameters
        ----------
        filename : str
            Name of the data file under ``outputs/datafiles/``.
        field_type : {"Electric", "Magnetic"}
            Selects the axis labels, plot style, and fit transformation.

        Returns
        -------
        None.

        """
        
        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        filename_path = os.path.join(outputs_directory, "datafiles", filename)
        plots_folder = os.path.join(outputs_directory, "plots")
        
        # If the folders dont exist, create them
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)

        # Create an empty list to store input data
        input_data = []        

        # Read in the data from the specified text file
        try:
            with open(filename_path, "r") as filein:
                for line in filein:
                    input_data.append(line.strip("\n").split(","))
                    
        # If text file cannot be found, print error
        except FileNotFoundError:
            print(f"Error: Could not find {filename_path}")
            return 

        # Convert input data into a np array
        input_data = np.array(input_data[1:], dtype = float)
        
        # Collect the input data
        phi = input_data[:, 7]
        r = input_data[:, 2]
        
        # Filter out zeros to allow log plotting
        mask1 = (r > 0) & (phi > 0)
        r_masked = r[mask1]
        phi_masked = phi[mask1]
        
        # Transform into log space for linear fit
        ln_r = np.log(r_masked)
        ln_phi = np.log(phi_masked)
        
        # Filter out data near the center and the boundaries
        mask2 = (r_masked > 3) & (r_masked < self.l // 6)
        
        if field_type == "Electric":
            
            # Create empty plots
            fig, ax = plt.subplots(1, 1, figsize=(8, 10))
            
            # Plot the data
            popt, pcov = scipy.optimize.curve_fit(self.f, ln_r[mask2], ln_phi[mask2])
            ax.plot(ln_r, ln_phi, 'yo', markersize=2, label='Data')
            ax.plot(ln_r, self.f(ln_r, *popt), 
                      'k-', label=f'Fit: {popt[0]:.3f}ln(r) {popt[1]:+.3f}')
            ax.legend(loc="upper right")
            
            # Set titles
            ax.set_title(
    rf"{field_type} Potential of the z-axis midplane vs Distance"
    "\n" 
    rf"{self.l} x {self.l} lattice with {self.tolerance} accuracy", 
    fontsize=16
)
            ax.set_xlabel("ln(r)", fontsize = 14)
            ax.set_ylabel(r"ln($\phi$)", fontsize = 14)
            
        if field_type == "Magnetic":
            
            # Create empty plots
            fig, ax = plt.subplots(1, 1, figsize=(8, 10))
            
            # Plot the contour
            popt, pcov = scipy.optimize.curve_fit(self.f, ln_r[mask2], phi_masked[mask2])
            ax.plot(ln_r, phi_masked, 'yo', markersize=2, label='Data')
            ax.plot(ln_r, self.f(ln_r, *popt), 
                      'k-', label=f'Fit: {popt[0]:.3f}ln(r) {popt[1]:+.3f}')
            ax.legend(loc="upper right")
            
            # Set titles
            ax.set_title(
    rf"{field_type} Potential of the z-axis midplane vs Distance"
    "\n" 
    rf"{self.l} x {self.l} lattice with {self.tolerance} accuracy", 
    fontsize=16
)
            ax.set_xlabel("ln(r)", fontsize = 14)
            ax.set_ylabel(r"$A_z$", fontsize = 14)
        
        # Save the plots to the plots folder
        save_filename = filename.replace(".txt", "potential_plot.png")
        save_path = os.path.join(plots_folder, save_filename)
        plt.savefig(save_path, dpi = 300)
        
        # Print message
        print(f"Plots successfully saved to: {save_path}")
        
        # Show final plot
        plt.show()
        
    def plot_potential_measurements(self, filename, field_type):
        """
        Generate and save a contour map and a heatmap of the potential on the
        z-axis midplane.

        Parameters
        ----------
        filename : str
            Name of the data file under ``outputs/datafiles/``.
        field_type : {"Electric", "Magnetic"}
            Used to label plot titles.

        Returns
        -------
        None.

        """
        
        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        filename_path = os.path.join(outputs_directory, "datafiles", filename)
        plots_folder = os.path.join(outputs_directory, "plots")
        
        # If the folders dont exist, create them
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)

        # Create an empty list to store input data
        input_data = []        

        # Read in the data from the specified text file
        try:
            with open(filename_path, "r") as filein:
                for line in filein:
                    input_data.append(line.strip("\n").split(","))
                    
        # If text file cannot be found, print error
        except FileNotFoundError:
            print(f"Error: Could not find {filename_path}")
            return 
            
        # Create an empty lattice
        phi = np.zeros(shape = (self.l, self.l, self.l))
        
        # Convert input data into a np array
        input_data = np.array(input_data[1:], dtype = float)
        
        # Collect the input data
        x = input_data[:, 0].astype(int)
        y = input_data[:, 1].astype(int)
        phi[x, y, self.l // 2] = input_data[:, 7]
        
        # Create empty plots
        fig, ax = plt.subplots(1, 1, figsize=(8, 10))
        
        # Plot the data
        plot = plt.contour(phi[:, :, self.l // 2].T, levels = 10)
        plt.colorbar(plot, shrink = 0.65)
        ax.set_title(
    rf"{field_type} Potential of the z-axis midplane"
    "\n"
    rf"{self.l} x {self.l} lattice with {self.tolerance} accuracy", 
    fontsize=16
)
        ax.set_xlabel("x", fontsize = 14)
        ax.set_ylabel("y", fontsize = 14)
        
        # Save the plots to the plots folder
        save_filename = filename.replace(".txt", "contour_plot.png")
        save_path = os.path.join(plots_folder, save_filename)
        plt.savefig(save_path, dpi = 300)
        
        # Create empty plots
        fig, ax = plt.subplots(1, 1, figsize=(8, 10))
        
        # Plot the contour
        plot = plt.imshow(phi[:, :, self.l // 2].T, origin = "lower")
        plt.colorbar(plot, shrink = 0.65)
        ax.set_title(
    rf"{field_type} Potential of the z-axis midplane"
    "\n"
    rf"{self.l} x {self.l} lattice with {self.tolerance} accuracy", 
    fontsize=16
)
        ax.set_xlabel("x", fontsize = 14)
        ax.set_ylabel("y", fontsize = 14)
        
        # Save the plots to the plots folder
        save_filename = filename.replace(".txt", "heatmap_plot.png")
        save_path = os.path.join(plots_folder, save_filename)
        plt.savefig(save_path, dpi = 300)
        
        # Print message
        print(f"Plots successfully saved to: {save_path}")
        
        # Show final plot
        plt.show()
        
    def plot_field_measurements(self, filename, field_type):
        """
        Generate and save a quiver plot of the field vectors on the z-axis
        midplane.

        Parameters
        ----------
        filename : str
            Name of the data file under ``outputs/datafiles/``.
        field_type : {"Electric", "Magnetic"}
            Used to label the plot title.

        Returns
        -------
        None.

        """
        
        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        filename_path = os.path.join(outputs_directory, "datafiles", filename)
        plots_folder = os.path.join(outputs_directory, "plots")
        
        # If the folders dont exist, create them
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)

        # Create an empty list to store input data
        input_data = []        

        # Read in the data from the specified text file
        try:
            with open(filename_path, "r") as filein:
                for line in filein:
                    input_data.append(line.strip("\n").split(","))
                    
        # If text file cannot be found, print error
        except FileNotFoundError:
            print(f"Error: Could not find {filename_path}")
            return 
            
        # Create an empty lattice
        field_x = np.zeros(shape = (self.l, self.l))
        field_y = np.zeros(shape = (self.l, self.l))
        field_z = np.zeros(shape = (self.l, self.l))
        
        # Convert input data into a np array
        input_data = np.array(input_data[1:], dtype = float)
        
        # Collect the input data
        x = input_data[:, 0].astype(int)
        y = input_data[:, 1].astype(int)
        field_x[x, y] = input_data[:, 3]
        field_y[x, y] = input_data[:, 4]
        field_z[x, y] = input_data[:, 5]
        
        # Create a meshgrid
        X, Y = np.meshgrid(np.arange(self.l), np.arange(self.l))
        
        # Skip every 3 pixels so the arrows aren't too crowded
        skip = (slice(None, None, 3), slice(None, None, 3))
        
        # Normalise and take into consideration divide by zero error
        field_tot = np.sqrt(field_x**2 + field_y**2 + field_z**2)
        field_tot_new = np.where(field_tot == 0, 1, field_tot)
        field_x_norm = field_x / field_tot_new
        field_y_norm = field_y / field_tot_new
            
        # Create empty plots
        fig, ax = plt.subplots(1, 1, figsize=(8, 10))
        
        # Plot the data
        ax.quiver(X[skip], Y[skip], field_x_norm.T[skip], field_y_norm.T[skip])
        ax.set_title(
    rf"{field_type} Field Vectors $E_x, E_y$"
    "\n"
    rf"{self.l} x {self.l} lattice with {self.tolerance} accuracy", 
    fontsize=16
)
        ax.set_xlabel("x", fontsize = 14)
        ax.set_ylabel("y", fontsize = 14)
        
        # Save the plots to the plots folder
        save_filename = filename.replace(".txt", "_plot.png")
        save_path = os.path.join(plots_folder, save_filename)
        plt.savefig(save_path, dpi = 300)
        
        # Print message
        print(f"Plots successfully saved to: {save_path}")
        
        # Show final plot
        plt.show()
        
    def plot_sors(self, filename):
        """
        Generate and plot the number of SOR sweeps to convergence vs the 
        relaxation parameter ω.

        Parameters
        ----------
        filename : str
            Name of the data file under ``outputs/datafiles/``.

        Returns
        -------
        None.

        """
        
        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        filename_path = os.path.join(outputs_directory, "datafiles", filename)
        plots_folder = os.path.join(outputs_directory, "plots")
        
        # If the folders dont exist, create them
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)

        # Create an empty list to store input data
        input_data = []        

        # Read in the data from the specified text file
        try:
            with open(filename_path, "r") as filein:
                for line in filein:
                    input_data.append(line.strip("\n").split(","))
                    
        # If text file cannot be found, print error
        except FileNotFoundError:
            print(f"Error: Could not find {filename_path}")
            return 
        
        # Convert input data into a np array
        input_data = np.array(input_data[1:], dtype = float)
        
        # Collect the input data
        omegas = input_data[:, 0]
        sweeps = input_data[:, 1]
        
        # Find the optimal omega
        min_idx = np.argmin(sweeps)
        optimal_omega = omegas[min_idx]
        optimal_sweeps = sweeps[min_idx]
        
        # Create empty plots
        fig, ax = plt.subplots(1, 1, figsize=(8, 10))
        
        # Plot the data
        ax.plot(omegas, sweeps, 'yo', markersize=2)
        ax.axvline(x=optimal_omega, color='r', linestyle='--', linewidth=1,
           label=rf"Optimal $\omega$ = {optimal_omega:.2f} ({int(optimal_sweeps)} sweeps)")
        
        ax.legend(loc="upper right")
        
        # Set titles
        ax.set_title(
    rf"Total sweeps vs the relaxation parameter $\omega$"
    "\n" 
    rf"{self.l} x {self.l} lattice with {self.tolerance} accuracy", 
    fontsize=16
)
        ax.set_xlabel(r"Relaxation parameter $\omega$", fontsize = 14)
        ax.set_ylabel("Total sweeps", fontsize = 14)
        
        # Save the plots to the plots folder
        save_filename = filename.replace(".txt", "field_plot.png")
        save_path = os.path.join(plots_folder, save_filename)
        plt.savefig(save_path, dpi = 300)
        
        # Print message
        print(f"Plots successfully saved to: {save_path}")
        
        # Show final plot
        plt.show()
        
    def sors_measurements(self, filename):
        """
        Measure the total number of sweeps until convergence for different
        relaxation parameters ω.

        Parameters
        ----------
        filename : str
            Output data filename (written under ``outputs/datafiles/``).

        Returns
        -------
        None.

        """
        
        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        datafiles_folder = os.path.join(outputs_directory, "datafiles")
        file_path = os.path.join(datafiles_folder, filename)
        
        # If the folders don't exist, create them
        if not os.path.exists(datafiles_folder):
            os.makedirs(datafiles_folder)
        
        # Make a range of omegas to test
        omegas = np.round(np.arange(1, 2, step = 0.01), 2)
        sweeps = []
        max_omega = np.max(omegas)
        
        # Iterate through all omegas
        for omega in omegas:
            print(f"\rSimulating omega = {omega}/{max_omega}", end='', flush=True)
            
            # Initialise Poisson class
            poisson = Poisson(self.l, self.tolerance, omega, init = "Monopole")
            
            # Initialise time and error
            t = 0
            error = 100
            
            # Continue to update phi until the error is smaller than the tolerance
            # Break if simulation goes on too long
            while error > self.tolerance and t < 10000:
    
                # Update the animation and obtain the error using Numbda
                distance = poisson.sor()
                error = np.max(distance)
                t += 1
                
            sweeps.append(t)
            
        # Open in "a" (append) or "w" (overwrite) mode
        # Write the values into the file
        with open(file_path, "w") as f:
            
            f.write("omega,t\n")
    
            for i in range(len(omegas)):
                            
                f.write(f"{omegas[i]},{sweeps[i]}\n")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Poisson")

    # User input parameters
    parser.add_argument("--l", type=int, default=100,
                        help="Lattice size (l x l)")
    # parser.add_argument("--dx", type = float, default = 1, help = "Spatial step")
    # parser.add_argument("--dt", type = float, default = 0.01, help = "Time step")
    parser.add_argument("--mode", type=str, default="ani", choices=["ani", "mea"],
                        help="Animation or measurements")
    parser.add_argument("--steps", type=int, default=100000,
                        help="Number of simulation steps")
    parser.add_argument("--omega", type=float, default=1.87, help="Omega")
    parser.add_argument("--tol", type=float, default=1e-6, help="Tolerance")
    parser.add_argument("--type", type=str, default="e", choices=["e", "m", "s"],
                        help="Electricm magnetic, or sors experiment")
    parser.add_argument("--alg", type=str, default="j", choices=["j", "gs", "sor"],
                        help="Algorithms: Jacobi (j), Gauss-Seidel (gs), Over-relaxation (sor)")

    args = parser.parse_args()

    # Pass in parameters to the classes
    sim = Simulation(l=args.l, tolerance=args.tol, omega=args.omega)

    if args.type == "e":

        filename = f"electric_{args.alg}alg_{args.l}l_{args.tol}tol_2.txt"
        sim.electric_measurements(args.alg, filename)
        sim.plot_potential_measurements(filename, field_type = "Electric")
        sim.plot_field_measurements(filename, field_type = "Electric")
        sim.plot_field_vs_distance_measurements(filename, field_type = "Electric")
        sim.plot_potential_vs_distance_measurements(filename, field_type = "Electric")
        
    if args.type == "m":
        
        filename = f"magnetic_{args.alg}alg_{args.l}l_{args.tol}tol_2.txt"
        sim.magnetic_measurements(args.alg, filename)
        sim.plot_potential_measurements(filename, field_type = "Magnetic")
        sim.plot_field_measurements(filename, field_type = "Magnetic")
        sim.plot_field_vs_distance_measurements(filename, field_type = "Magnetic")
        sim.plot_potential_vs_distance_measurements(filename, field_type = "Magnetic")
        
    if args.type == "s":
        
        filename = f"sors_experiment_{args.l}l_{args.tol}tol_2.txt"
        sim.sors_measurements(filename)
        sim.plot_sors(filename)
      
import numpy as np
import matplotlib.pyplot as plt
import os

save_path = os.getcwd()

# Problem parameters
nu = 0.01 / np.pi  # Viscosity coefficient
L = 1  # Domain length in x
T = 1  # Total simulation time
Nx = 500  # Number of spatial grid points
dx = 2 * L / Nx  # Spatial step size
dt = 0.0001  # Time step
Nt = round(T / dt)  # Total number of time steps

# Spatial and temporal grids
x = np.linspace(-L, L, Nx + 1)
t = np.linspace(0, T, Nt + 1)

# Solution matrix (each column is u at a different time)
U = np.zeros((Nx + 1, Nt + 1))

# Initial conditions
U[:, 0] = -np.sin(np.pi * x)

# Tridiagonal matrix for Crank-Nicolson (Implicit diffusion)
sigma = nu * dt / dx**2
A = np.diag((1 + sigma) * np.ones(Nx - 1)) + np.diag(-0.5 * sigma * np.ones(Nx - 2), 1) + np.diag(-0.5 * sigma * np.ones(Nx - 2), -1)

# Time loop (Crank-Nicolson Method)
for n in range(Nt):
    u_old = U[:, n].copy()
    u_new = u_old.copy()
    
    # Right-hand side vector of the implicit equation
    b = (1 - sigma) * u_old[1:Nx] + 0.5 * sigma * (u_old[:Nx-1] + u_old[2:Nx+1])
    
    # Explicit treatment of the convective term (-u * du/dx)
    for i in range(1, Nx):
        if u_old[i] > 0:
            du_dx = (u_old[i] - u_old[i-1]) / dx  # Classic upwind
        else:
            du_dx = (u_old[i+1] - u_old[i]) / dx  # Reverse upwind
        conv = -dt * u_old[i] * du_dx
        b[i-1] += conv  # Add convection with upwind
    
    # Solve linear system A*u = b
    u_new[1:Nx] = np.linalg.solve(A, b)
    
    # Apply boundary conditions (Dirichlet)
    u_new[0] = 0
    u_new[-1] = 0
    
    # Store result
    U[:, n + 1] = u_new

# Selected time instants for plotting
t_selected = [0, 0.1, 0.2, 0.25, 0.3, 0.5, 0.75, 1]
t_indices = [round(ti / dt) for ti in t_selected]

# Plot the solution at selected time instants
plt.figure(figsize=(5, 5*6/8))
for i, ti in enumerate(t_selected):
    plt.plot(x, U[:, t_indices[i]], label=f't = {ti}', alpha=0.75)
plt.xlabel('x', fontsize=12)
plt.ylabel('u(x,t)', fontsize=12)
plt.legend()
plt.savefig(os.path.join(save_path, "Burgers.pdf"), bbox_inches="tight")

for i, ti in enumerate(t_selected):
    filename = f'burgers_solution_t{ti}.npy'
    data = {'x': x, 'u': U[:, t_indices[i]]}  
    np.save(filename, data)


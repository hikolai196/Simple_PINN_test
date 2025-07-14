import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------------------------------------
# Solves the 2D steady-state heat conduction equation in a square plate using the Finite Volume Method (FVM).

# ===Parameters===
# nx : int
#     Number of control volumes (nodes) in the x-direction.
# ny : int
#     Number of control volumes (nodes) in the y-direction.
# L : float
#     Length of one side of the square plate (meters).
# T_top : float
#     Temperature at the top boundary (degrees Celsius).
# T_other : float
#     Temperature at the left, right, and bottom boundaries (degrees Celsius).
# k : float
#     Thermal conductivity of the material (W/m·K).
# TOLERANCE : float
#     Convergence criterion for the maximum change in temperature.
# MAX_ITER : int
#     Maximum number of iterations allowed.

# ===Returns===
# T : ndarray
#     2D array of temperatures at each node (degrees Celsius).
# Q_total : float
#     Total heat transfer rate through the top boundary (Watts).
# L : float
#     Length of the plate (meters).
# num_iterations : int
#    Number of iterations performed before convergence.
# ---------------------------------------------------------


def FVM(nx=100, ny=100, L=1e-3, T_top=60.0, T_other=20.0, k=205.0, TOLERANCE=1e-4, MAX_ITER=10000):
    dx = L / nx
    dy = L / ny

    T = np.ones((ny, nx)) * T_other

    # Apply boundary conditions (Dirichlet)
    T[0, :] = T_top         # Top boundary
    T[:, 0] = T_other       # Left boundary
    T[:, -1] = T_other      # Right boundary
    T[-1, :] = T_other      # Bottom boundary

    # FVM coefficients
    aW = k * dy / dx
    aE = k * dy / dx
    aS = k * dx / dy
    aN = k * dx / dy
    aP = aW + aE + aS + aN

    converged = False

    for iteration in tqdm(range(MAX_ITER), desc="FVM Solving"):
        T_old = T.copy()

        # Vectorized update for all internal nodes
        T[1:-1, 1:-1] = (
            aW * T[1:-1, 0:-2] +
            aE * T[1:-1, 2:] +
            aS * T[2:, 1:-1] +
            aN * T[0:-2, 1:-1]
        ) / aP

        # Re-apply Dirichlet boundary conditions
        T[0, :] = T_top         # Top boundary
        T[:, 0] = T_other       # Left boundary
        T[:, -1] = T_other      # Right boundary
        T[-1, :] = T_other      # Bottom boundary

        if np.max(np.abs(T - T_old)) < TOLERANCE:
            converged = True
            break

    if converged:
        print(f'Converged in {iteration+1} iterations.')
    else:
        print("Did not converge within the maximum number of iterations.")

    # --- Heat transfer calculation ---
    # dTdy: Temperature gradient normal to the top boundary (degrees C / m)
    # Q_per_width: Heat flux per unit width (W/m)
    # Q_total: Total heat transfer across the top edge (Watts)
    # Since the plate is L x L (default 1 mm x 1 mm), Q_total is for the full width (L meters)
    dTdy = (T[1, 1:-1] - T[0, 1:-1]) / dy    # (degrees C / m)
    Q_per_width = -k * dTdy                  # (W/m)
    Q_total = np.mean(Q_per_width) * L       # (W) for the length L (default 1e-3 m)

    return T, Q_total, L, iteration+1

def plot_temperature(T, L):
    plt.imshow(T, cmap='hot', origin='lower', extent=[0, L*1e3, 0, L*1e3], aspect='equal')
    plt.colorbar(label='Temperature (°C)')
    plt.title('2D Steady-State Temperature (FVM, 100x100, Aluminum)')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.grid(False)
    plt.show()

def main():
    T, Q_total, L, num_iterations = FVM()
    print(f"Approximate total heat transfer rate through top boundary: {Q_total:.6e} W")
    print(f"Number of iterations to converge: {num_iterations}")
    plot_temperature(T, L)

if __name__ == "__main__":
    main()
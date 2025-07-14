import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def FDM(nx=100, ny=100, L=1e-3, T_top=60.0, T_other=20.0, k=205.0, TOLERANCE=1e-4, MAX_ITER=10000):
    dx = L / nx
    dy = L / ny

    T = np.ones((ny, nx)) * T_other

    # Apply boundary conditions (Dirichlet)
    T[0, :] = T_top         # Top boundary
    T[:, 0] = T_other       # Left boundary
    T[:, -1] = T_other      # Right boundary
    T[-1, :] = T_other      # Bottom boundary

    alpha_x = k / dx**2
    alpha_y = k / dy**2
    aP = 2 * (alpha_x + alpha_y)

    converged = False

    for iteration in tqdm(range(MAX_ITER), desc="FDM Solving"):
        T_old = T.copy()
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                T[i, j] = (alpha_x * (T[i, j-1] + T[i, j+1]) +
                           alpha_y * (T[i-1, j] + T[i+1, j])) / aP
        if np.max(np.abs(T - T_old)) < TOLERANCE:
            converged = True
            break

    if converged:
        print(f'FDM converged in {iteration+1} iterations.')
    else:
        print("FDM did not converge within the maximum number of iterations.")

    # Calculate heat transfer rate at the top boundary (Fourier's Law, FDM style)
    dTdy = (T[1, 1:-1] - T[0, 1:-1]) / dy    # temperature gradient at top (excluding corners)
    Q_per_width = -k * dTdy                  # W/m (since sheet is 1 mm wide)
    Q_total = np.mean(Q_per_width) * L       # total heat transfer (W)

    return T, Q_total, L

def plot_temperature(T, L):
    plt.imshow(T, cmap='hot', origin='lower', extent=[0, L*1e3, 0, L*1e3], aspect='equal')
    plt.colorbar(label='Temperature (°C)')
    plt.title(f'2D Steady-State Temperature (FDM, 100x100, Aluminum)')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.grid(False)
    plt.show()

def main():
    T, Q_total, L = FDM()
    print(f"Total heat transfer rate through top boundary: {Q_total:.6e} W")
    plot_temperature(T, L)

if __name__ == "__main__":
    main()
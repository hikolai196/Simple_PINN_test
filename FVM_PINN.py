import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --- Neural network definition ---
class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            layer = nn.Linear(layers[i], layers[i+1])
            nn.init.xavier_normal_(layer.weight)  # Xavier initialization
            self.layers.append(layer)

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        for layer in self.layers[:-1]:
            xy = self.activation(layer(xy))
        return self.layers[-1](xy)

# --- PDE residual using autograd ---
def pde_residual(model, x, y):
    x.requires_grad_(True)
    y.requires_grad_(True)
    T = model(x, y)
    T_x = torch.autograd.grad(T, x, torch.ones_like(T), create_graph=True)[0]
    T_xx = torch.autograd.grad(T_x, x, torch.ones_like(T_x), create_graph=True)[0]
    T_y = torch.autograd.grad(T, y, torch.ones_like(T), create_graph=True)[0]
    T_yy = torch.autograd.grad(T_y, y, torch.ones_like(T_y), create_graph=True)[0]
    return T_xx + T_yy

# --- Loss function with normalization and weighting ---
def loss_fn(model, xy_f, xy_top, xy_bottom, xy_left, xy_right, T_top, T_other, weights):
    mse = nn.MSELoss()
    # Interior/collocation points
    x_f = torch.tensor(xy_f[:,0:1], dtype=torch.float32, requires_grad=True)
    y_f = torch.tensor(xy_f[:,1:2], dtype=torch.float32, requires_grad=True)
    res = pde_residual(model, x_f, y_f)
    loss_pde = mse(res, torch.zeros_like(res))

    # Boundary points
    # Top
    x_top = torch.tensor(xy_top[:,0:1], dtype=torch.float32)
    y_top = torch.tensor(xy_top[:,1:2], dtype=torch.float32)
    T_pred_top = model(x_top, y_top)
    loss_top = mse(T_pred_top, T_top.expand_as(T_pred_top))
    # Bottom
    x_bot = torch.tensor(xy_bottom[:,0:1], dtype=torch.float32)
    y_bot = torch.tensor(xy_bottom[:,1:2], dtype=torch.float32)
    T_pred_bot = model(x_bot, y_bot)
    loss_bot = mse(T_pred_bot, T_other.expand_as(T_pred_bot))
    # Left
    x_left = torch.tensor(xy_left[:,0:1], dtype=torch.float32)
    y_left = torch.tensor(xy_left[:,1:2], dtype=torch.float32)
    T_pred_left = model(x_left, y_left)
    loss_left = mse(T_pred_left, T_other.expand_as(T_pred_left))
    # Right
    x_right = torch.tensor(xy_right[:,0:1], dtype=torch.float32)
    y_right = torch.tensor(xy_right[:,1:2], dtype=torch.float32)
    T_pred_right = model(x_right, y_right)
    loss_right = mse(T_pred_right, T_other.expand_as(T_pred_right))

    # Weighted sum
    total_loss = (weights['pde'] * loss_pde +
                  weights['top'] * loss_top +
                  weights['bottom'] * loss_bot +
                  weights['left'] * loss_left +
                  weights['right'] * loss_right)
    return total_loss

# --- Main routine ---
def main():
    # Problem parameters (normalized domain [0,1] for better training)
    L = 1e-3
    T_top = 60.0
    T_other = 20.0
    N_f = 20000   # Collocation points
    N_b = 2000    # Boundary points per edge

    # Collocation (interior) points in [0,1]x[0,1]
    xy_f = np.random.rand(N_f, 2)

    # Boundary points
    x_b = np.linspace(0, 1, N_b).reshape(-1, 1)
    y_b = np.linspace(0, 1, N_b).reshape(-1, 1)
    xy_top = np.hstack([x_b, np.ones_like(x_b)])
    xy_bottom = np.hstack([x_b, np.zeros_like(x_b)])
    xy_left = np.hstack([np.zeros_like(y_b), y_b])
    xy_right = np.hstack([np.ones_like(y_b), y_b])

    # Model and optimizer
    model = PINN([2, 64, 64, 64, 1])
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.7)

    T_top_tensor = torch.tensor([[T_top]], dtype=torch.float32)
    T_other_tensor = torch.tensor([[T_other]], dtype=torch.float32)

    # Weighting for loss terms (tune as needed)
    weights = {'pde': 1.0, 'top': 10.0, 'bottom': 10.0, 'left': 10.0, 'right': 10.0}

    # Training loop
    epochs = 5000
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model, xy_f, xy_top, xy_bottom, xy_left, xy_right,
                       T_top_tensor, T_other_tensor, weights)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 500 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    # Optionally: Fine-tune with LBFGS optimizer for better convergence
    def closure():
        optimizer_lbfgs.zero_grad()
        loss = loss_fn(model, xy_f, xy_top, xy_bottom, xy_left, xy_right,
                       T_top_tensor, T_other_tensor, weights)
        loss.backward()
        return loss
    optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=500)
    optimizer_lbfgs.step(closure)

    # Prediction on a grid for visualization (map back to [0, L])
    nx, ny = 100, 100
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    x_flat = torch.tensor(X.flatten()[:,None], dtype=torch.float32)
    y_flat = torch.tensor(Y.flatten()[:,None], dtype=torch.float32)
    with torch.no_grad():
        T_pred = model(x_flat, y_flat).cpu().numpy().reshape(ny, nx)

    # Plot
    plt.imshow(T_pred, cmap='hot', origin='lower', extent=[0, L*1e3, 0, L*1e3], aspect='equal')
    plt.colorbar(label='Temperature (°C)')
    plt.title('2D Steady-State Temperature (PINN, Optimized)')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.show()

if __name__ == "__main__":
    main()
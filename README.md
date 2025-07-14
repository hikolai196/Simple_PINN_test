# 2D Steady-State Heat Conduction Solver using PINN
#### This app demonstrates a Physics-Informed Neural Network (PINN) for solving the 2D steady-state heat conduction (Laplace) equation on a square plate with Dirichlet boundary conditions.


---

## Problem Description

We solve the Laplace equation:

$$
  \\frac{\\partial^2 T}{\\partial x^2} + \\frac{\\partial^2 T}{\\partial y^2} = 0
$$

on a square domain \\([0, L] \\times [0, L]\\), subject to:

- **Top boundary:** \\(T(x, L) = T_{\\text{top}}\\)
- **Bottom, left, right boundaries:** \\(T = T_{\\text{other}}\\)

---

## Features

- PINN implementation using PyTorch
- Flexible neural network architecture
- Enforces PDE and boundary conditions via loss function
- Visualization of the temperature field
- Easily customizable parameters

---

## Requirements

- Python 3.7+
- [PyTorch](https://pytorch.org/)
- numpy
- matplotlib

Install requirements with:
```bash
pip install torch numpy matplotlib

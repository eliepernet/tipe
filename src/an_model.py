
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import json

import torch
import torch.nn as nn

from src.data import AneurysmData

sys.setrecursionlimit(10000)

w, h = 150, 90
data = AneurysmData("../input/datasets/eliepernet/solid-77-vtk/Solid_77.vtk", w, h)

rho = 1055.0
mu = 0.0035
nu = mu / rho

V = 0.3

L_ref = 0.02
U_ref = V
P_ref = rho * U_ref **2  # Pa

Re = U_ref * L_ref / nu

inlet_points = []
i = np.min(data.inside[:, 0])
while data.grid[0][i] == 2:
    inlet_points.append([0, i * (0.015 / h)])
    i += 1

inlet_radius = 0.002

def distance_to_inlet_center(x, y):
    return np.sqrt((x - 0.00375) ** 2 + y ** 2)

def speed(x, y):
    return V * (1 - (distance_to_inlet_center(x, y) ** 2 / inlet_radius ** 2))

inlet_points = []
inlet_speed_values = []

for cell in data.d_inside:
    if distance_to_inlet_center(cell[0], cell[1]) <= inlet_radius:
        inlet_points.append([float(cell[0]), float(cell[1])])
        inlet_speed_values.append([0.0, float(speed(cell[0], cell[1]))])

inlet_speed_values = np.array(inlet_speed_values)
inlet_points = np.array(inlet_points)


mse = nn.MSELoss()

inside_points = torch.tensor(np.float64(np.column_stack((data.d_inside[:, 0] / L_ref, data.d_inside[:, 1] / L_ref))), dtype=torch.float64, requires_grad=True)
inlet_points = torch.tensor(np.float64(np.column_stack((inlet_points[:, 0] / L_ref, inlet_points[:, 1] / L_ref))), dtype=torch.float64, requires_grad=True)
outside_points = torch.tensor(np.float64(np.column_stack((data.d_walls[:, 0] / L_ref, data.d_walls[:, 1] / L_ref))), dtype=torch.float64, requires_grad=True)
outlet_points = torch.tensor(np.float64(np.array([[x * (0.02 / w) / L_ref, y * (0.015 / h) / L_ref] for x, y, _ in data.inside[(data.inside[:, 1] == 0) & (data.inside[:, 0] > 50)]])), dtype=torch.float64, requires_grad=True)

inlet_labels = torch.tensor(np.column_stack((inlet_speed_values[:, 0] / V, inlet_speed_values[:, 1] / V)), dtype=torch.float64)

PINNModel = nn.Sequential(
    nn.Linear(2, 128),
    nn.Tanh(),
    nn.Linear(128, 128),
    nn.Tanh(),
    nn.Linear(128, 128),
    nn.Tanh(),
    nn.Linear(128, 3)
).double()

def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)

PINNModel.apply(initialize_weights)

mse = nn.MSELoss()
losses = []

def loss_function(coords, u, v, p):
    grad_u = torch.autograd.grad(u, coords, torch.ones_like(u), create_graph=True)[0]
    du_x = grad_u[:, 0]
    du_y = grad_u[:, 1]

    grad_v = torch.autograd.grad(v, coords, torch.ones_like(v), create_graph=True)[0]
    dv_x = grad_v[:, 0]
    dv_y = grad_v[:, 1]

    grad_p = torch.autograd.grad(p, coords, torch.ones_like(p), create_graph=True)[0]
    dp_x = grad_p[:, 0]
    dp_y = grad_p[:, 1]

    du_xx = torch.autograd.grad(du_x.sum(), coords, create_graph=True)[0][:, 0]
    du_yy = torch.autograd.grad(du_y.sum(), coords, create_graph=True)[0][:, 1]

    dv_xx = torch.autograd.grad(dv_x.sum(), coords, create_graph=True)[0][:, 0]
    dv_yy = torch.autograd.grad(dv_y.sum(), coords, create_graph=True)[0][:, 1]

    ns_x = u * du_x + v * du_y + dp_x - ( 1 /Re) * (du_xx + du_yy)
    ns_y = u * dv_x + v * dv_y + dp_y - ( 1 /Re) * (dv_xx + dv_yy)

    continuity = du_x + dv_y

    return ns_x, ns_y, continuity


def compute_total_loss():
    # PDE — inside_points DOIT avoir requires_grad=True
    inside_values = PINNModel(inside_points)
    u_in = inside_values[:, 0]
    v_in = inside_values[:, 1]
    p_in = inside_values[:, 2]
    ns_x, ns_y, continuity = loss_function(inside_points, u_in, v_in, p_in)

    ns_loss   = mse(ns_x, torch.zeros_like(ns_x)) + mse(ns_y, torch.zeros_like(ns_y))
    cont_loss = mse(continuity, torch.zeros_like(continuity))

    # Wall (no-slip)
    wall_values = PINNModel(outside_points)
    wall_loss = mse(wall_values[:, 0], torch.zeros_like(wall_values[:, 0])) \
                + mse(wall_values[:, 1], torch.zeros_like(wall_values[:, 1]))

    # Inlet
    inlet_values = PINNModel(inlet_points)
    inlet_loss = mse(inlet_values[:, 0], inlet_labels[:, 0]) \
                 + mse(inlet_values[:, 1], inlet_labels[:, 1])

    # Outlet — pression nulle en sortie (Neumann ou p=0)
    outlet_values = PINNModel(outlet_points)
    outlet_loss = mse(outlet_values[:, 2], torch.zeros_like(outlet_values[:, 2]))

    total = (
            10 * ns_loss
            + 10 * cont_loss
            + 5  * wall_loss
            + 20 * inlet_loss
            + 5  * outlet_loss
    )

    losses.append([ns_loss.item(), cont_loss.item(), wall_loss.item(),
                   inlet_loss.item(), outlet_loss.item(), total.item()])
    return total

def train_pinn(adam_epochs=40000, lbfgs_epochs=5000, adam_lr=1e-3):
    lossses = []
    mse = nn.MSELoss()

    optimizer_adam = torch.optim.Adam(PINNModel.parameters(), lr=adam_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_adam, T_max=adam_epochs, eta_min=1e-5
    )

    for epoch in range(adam_epochs):
        optimizer_adam.zero_grad()

        total_loss = compute_total_loss()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(PINNModel.parameters(), max_norm=1.0)
        optimizer_adam.step()
        scheduler.step()

    optimizer_lbfgs = torch.optim.LBFGS(
        PINNModel.parameters(),
        lr=1.0,
        max_iter=20,
        max_eval=25,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        history_size=100,
        line_search_fn="strong_wolfe"
    )

    lbfgs_iter = 0

    def closure():
        nonlocal lbfgs_iter

        optimizer_lbfgs.zero_grad()
        total_loss = compute_total_loss()
        total_loss.backward()

        lbfgs_iter += 1
        return total_loss

    for _ in range(lbfgs_epochs // 20):
        optimizer_lbfgs.step(closure)

        if losses[-1][5] < 1e-6:
            break

train_pinn()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

model_filename = f"./pinn_model.pt"
torch.save({
    'model_state_dict': PINNModel.state_dict(),
    'model_architecture': str(PINNModel),
    'timestamp': timestamp
}, model_filename)

losses_filename = f"./training_losses.json"

losses_json = {
    'timestamp': timestamp,
    'total_epochs': len(losses),
    'losses_per_epoch': losses
}

with open(losses_filename, 'w') as f:
    json.dump(losses_json, f, indent=2)


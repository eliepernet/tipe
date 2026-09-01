
import torch
import torch.nn as nn
import numpy as np

import json

from src.utils import ns_residuals, kova_data, u_func, v_func, p_func

Re = 40
nu = 1 / Re
l = 1 / (2 * nu) - np.sqrt(1 / (4 * nu ** 2) + 4 * np.pi ** 2)

PINNModel = nn.Sequential(
    nn.Linear(2, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 3)
).double()

def compute_total_loss():
    inside_values = PINNModel(inside_points)
    ns_x, ns_y, continuity = ns_residuals(inside_points, inside_values[:, 0], inside_values[:, 1], inside_values[:, 2], 1/Re)

    ns_loss_x_inside, ns_loss_y_inside = mse(ns_x, torch.zeros_like(ns_x)), mse(ns_y, torch.zeros_like(ns_y))
    continuity_loss_inside = mse(continuity, torch.zeros_like(continuity))

    outside_values = PINNModel(outside_points)
    outside_loss = mse(outside_values, outside_true_results)

    losses.append([ns_loss_x_inside.item(), ns_loss_y_inside.item(), continuity_loss_inside.item(), outside_loss.item(), ns_loss_x_inside.item() + ns_loss_y_inside.item() + continuity_loss_inside.item() + outside_loss.item()])

    return ns_loss_x_inside + ns_loss_y_inside + continuity_loss_inside + 100 * outside_loss

adam_optimizer = torch.optim.Adam(PINNModel.parameters(), lr=0.001)
optimizer_lbfgs = torch.optim.LBFGS(
    PINNModel.parameters(),
    lr=1.0,
    max_iter=20,
    history_size=50,
    tolerance_grad=1e-7,
    tolerance_change=1e-9,
    line_search_fn="strong_wolfe"
)

mse = nn.MSELoss()

outside_points, inside_points = kova_data()

outside_true_results = torch.stack([
    u_func(outside_points[:, 0], outside_points[:, 1]),
    v_func(outside_points[:, 0], outside_points[:, 1]),
    p_func(outside_points[:, 0])
], dim=1)

adam_epochs = 1000
lbgfs_epochs = 1000

losses = []

for _ in range(adam_epochs):
    adam_optimizer.zero_grad()
    loss = compute_total_loss()
    loss.backward()
    losses.append(loss.item())
    adam_optimizer.step()

for _ in range(lbgfs_epochs):
    def closure():
        optimizer_lbfgs.zero_grad()
        loss = compute_total_loss()
        loss.backward()
        return loss

    loss = optimizer_lbfgs.step(closure)

torch.save(PINNModel.state_dict(), "./PINN_Model_09.pt")

with open("losses_09.json", "w") as f:
    json.dump(losses, f)

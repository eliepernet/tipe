import torch
import torch.nn as nn
import numpy as np
import json

from src.utils import ns_residuals

PINNModel = nn.Sequential(
    nn.Linear(2, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 3)
).double()
PINNModel.load_state_dict(torch.load("../models/kova/kova_9/PINN_Model_09.pt"))

outside_points = torch.Tensor([[-0.5, i] for i in np.linspace(-0.5, 1.5, 25)] + [[1, i] for i in np.linspace(-0.5, 1.5, 25)] + [[i, -0.5] for i in np.linspace(-0.5, 1, 25)] + [[i, 1.5] for i in np.linspace(-0.5, 1, 25)]).double()
outside_true_results = PINNModel(outside_points).detach()

inside_points = torch.tensor(
    np.column_stack((np.random.uniform(-0.5, 1.0, 2000), np.random.uniform(-0.5, 1.5, 2000)))
).double().requires_grad_(True)

Re = 10
nu = 1 / Re
l = 1 / (2 * nu) - np.sqrt(1 / (4 * nu ** 2) + 4 * np.pi ** 2)

def compute_total_loss():
    inside_values = PINNModel(inside_points)
    ns_x, ns_y, continuity = ns_residuals(inside_points,
                                           inside_values[:, 0],
                                           inside_values[:, 1],
                                           inside_values[:, 2],
                                          1/Re
                                           )

    ns_loss_x_inside, ns_loss_y_inside = mse(ns_x, torch.zeros_like(ns_x)), mse(ns_y, torch.zeros_like(ns_y))
    continuity_loss_inside = mse(continuity, torch.zeros_like(continuity))

    outside_values = PINNModel(outside_points)
    outside_loss = mse(outside_values, outside_true_results)

    losses.append([ns_loss_x_inside.item(), ns_loss_y_inside.item(), continuity_loss_inside.item(), outside_loss.item()])

    return ns_loss_x_inside + ns_loss_y_inside + continuity_loss_inside + outside_loss


mse = nn.MSELoss()

lbgfs_epochs = 1000

optimizer_lbfgs = torch.optim.LBFGS(
    PINNModel.parameters(),
    lr=1.0,
    max_iter=20,
    history_size=50,
    tolerance_grad=1e-7,
    tolerance_change=1e-9,
    line_search_fn="strong_wolfe"
)

losses = []

for _ in range(lbgfs_epochs):
    def closure():
        optimizer_lbfgs.zero_grad()
        loss = compute_total_loss()
        loss.backward()
        return loss

    loss = optimizer_lbfgs.step(closure)

torch.save(PINNModel.state_dict(), "./PINN_Model_09_ft.pt")

with open("losses_09.json", "w") as f:
    json.dump(losses, f)

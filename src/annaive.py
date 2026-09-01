import sys
import numpy as np

import json

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from src.data import AneurysmData
from src.utils import increase_points_number, parabolic_inlet, ns_residuals

sys.setrecursionlimit(10000)

w, h = 150, 90
data = AneurysmData("../input/datasets/eliepernet/solid-77-vtk/Solid_77.vtk", w, h)

rho = 1055.0
mu = 0.0035
nu = mu / rho

V_max = 1.114
L_ref = 0.02
P_ref = rho * V_max ** 2  # Pa
Re = V_max * 0.004 / nu
nu = 1 / Re
inlet_radius = 0.09

def to_tensor(xy_np):
    return torch.tensor(np.float64(xy_np), dtype=torch.float64)

inside_points = list(np.float64(np.column_stack((data.d_inside[:, 0] / L_ref, data.d_inside[:, 1] / L_ref))))
inside_points = torch.tensor(inside_points).double()

inside_points = increase_points_number(inside_points, 4, 0.001, 0.001)
inside_points = inside_points.detach().requires_grad_(True)

outside_points = torch.tensor(
    np.float64(np.column_stack((data.d_walls[:, 0] / L_ref, data.d_walls[:, 1] / L_ref)))).double()

inlet_points = torch.column_stack((
    torch.linspace(0.1, 0.285, 150),
    torch.zeros(150)
)).double()
inlet_points = inlet_points.detach().requires_grad_(True)

inlet_labels = torch.column_stack((
    torch.zeros(len(inlet_points)),
    parabolic_inlet(inlet_points[:, 0].detach())
)).double().detach()

outlet_mask = (data.inside[:, 1] == 0) & (data.inside[:, 0] > 50)
outlet_xy = data.inside[outlet_mask, :2] / L_ref  # ← cohérent
outlet_points = to_tensor(outlet_xy)

outside_points = increase_points_number(outside_points, 8, 0.001, 0.001).detach()
outlet_points = increase_points_number(outlet_points, 16, 0.001, 0.001).detach()

PINNModel = nn.Sequential(
    nn.Linear(2, 128), nn.Tanh(),
    nn.Linear(128, 128), nn.Tanh(),
    nn.Linear(128, 128), nn.Tanh(),
    nn.Linear(128, 128), nn.Tanh(),
    nn.Linear(128, 3)
).double()


def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1.0)
        nn.init.zeros_(module.bias)


PINNModel.apply(initialize_weights)

mse = nn.MSELoss()
losses = []


def loss_function():
    inside_values = PINNModel(inside_points)
    ns_x, ns_y, continuity = ns_residuals(inside_points, inside_values[:, 0], inside_values[:, 1], inside_values[:, 2],
                                          nu)

    ns_loss = mse(ns_x, torch.zeros_like(ns_x)) + mse(ns_y, torch.zeros_like(ns_y))
    cont_loss = mse(continuity, torch.zeros_like(continuity))

    wall_values = PINNModel(outside_points)
    wall_loss = mse(wall_values[:, 0], torch.zeros_like(wall_values[:, 0])) \
                + mse(wall_values[:, 1], torch.zeros_like(wall_values[:, 1]))

    # Inlet
    inlet_values = PINNModel(inlet_points)
    inlet_loss = mse(inlet_values[:, 0], inlet_labels[:, 0]) \
                 + mse(inlet_values[:, 1], inlet_labels[:, 1])

    outlet_values = PINNModel(outlet_points)
    outlet_loss = mse(outlet_values[:, 2], torch.zeros_like(outlet_values[:, 2]))

    total = (
            50 * (ns_loss
                  + cont_loss)
            + 10 * wall_loss
            + 50 * inlet_loss
            + 5 * outlet_loss
    )

    losses.append([ns_loss.item(), cont_loss.item(), wall_loss.item(),
                   inlet_loss.item(), outlet_loss.item(), total.item()])
    return total


lossses = []

adam_epochs = 10000

adam_optimizer = torch.optim.Adam(PINNModel.parameters(), lr=0.001)
scheduler = StepLR(adam_optimizer, step_size=15000, gamma=0.1)

for epoch in range(adam_epochs):
    adam_optimizer.zero_grad()

    total_loss = loss_function()
    total_loss.backward()

    adam_optimizer.step()
    scheduler.step()

optimizer_lbfgs = torch.optim.LBFGS(
    PINNModel.parameters(),
    lr=1.0,
    max_iter=1000,
    history_size=50,
    tolerance_grad=1e-7,
    tolerance_change=1e-9,
    line_search_fn="strong_wolfe"
)

def closure():
    optimizer_lbfgs.zero_grad()
    loss = loss_function()
    loss.backward()
    return loss

torch.save(PINNModel.state_dict(), "./AN_model_01.pt")

with open("AN_model_01.json", "w") as f:
    json.dump(losses, f)
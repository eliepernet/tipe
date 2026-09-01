import json

import numpy as np
import torch
import torch.nn as nn


from src.utils import extract_ds_values, ns_residuals

H5_FILE = "../coarse_dataset/AllFields_Resultats_MESH_210.h5"
T = 8  # pas de temps 20

RE = 520
nu = 1 / RE

Q = 7

P  = -3700 + 1.32 * Q
P_ref = abs(P)
P_norm = P / P_ref

U_ref = 1000

model = nn.Sequential(
    nn.Linear(2, 256),
    nn.Tanh(),
    nn.Linear(256, 256),
    nn.Tanh(),
    nn.Linear(256, 256),
    nn.Tanh(),
    nn.Linear(256, 3)
).double()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

mse = nn.MSELoss()
losses = []
adam_epochs = 10000

x, y, uv = extract_ds_values(H5_FILE, T)

inside_points = torch.tensor(
    np.column_stack((x / 18, y / 14)),
).double().requires_grad_(True)

inside_labels = torch.tensor(uv / U_ref).double()
outlet_points = torch.column_stack((torch.linspace(14.5 / 18, 17.85 / 18, 50), torch.zeros(50))).double()

def compute_total_loss():
    inside_values = model(inside_points)
    inside_values_loss = mse(inside_values[:, :2], inside_labels)

    ns_x, ns_y, continuity = ns_residuals(inside_points, inside_values[:, 0], inside_values[:, 1], inside_values[:, 2], nu)

    ns_loss_x_inside, ns_loss_y_inside = mse(ns_x, torch.zeros_like(ns_x)), mse(ns_y, torch.zeros_like(ns_y))
    continuity_loss_inside = mse(continuity, torch.zeros_like(continuity))

    outlet_values = model(outlet_points)
    outlet_values_loss = mse(outlet_values[:, 2], torch.full_like(outlet_values[:, 2], P_norm).double())

    losses.append([
        inside_values_loss.item(),
        ns_loss_x_inside.item(),
        ns_loss_y_inside.item(),
        continuity_loss_inside.item(),
        outlet_values_loss.item(),
        ns_loss_x_inside.item() + ns_loss_y_inside.item() + continuity_loss_inside.item() + outlet_values_loss.item() + inside_values_loss.item()
    ])

    return ns_loss_x_inside + ns_loss_y_inside + continuity_loss_inside + 20 * outlet_values_loss + 100 * inside_values_loss


for _ in range(adam_epochs):
    optimizer.zero_grad()
    loss = compute_total_loss()
    loss.backward()
    optimizer.step()

optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(), lr=1.0, max_iter=50000,
    history_size=100, line_search_fn="strong_wolfe"
)

def closure():
    optimizer_lbfgs.zero_grad()
    loss = compute_total_loss()
    loss.backward()
    return loss

optimizer_lbfgs.step(closure)

torch.save(model.state_dict(), "./DS_Model_03.pt")

with open("DS_Model_03.json", "w") as f:
    json.dump(losses, f)

import json

import numpy as np
import torch
import torch.nn as nn

from src.utils import extract_ds_values, increase_points_number, parabolic_inlet, extract_ds_outside_values, \
    ns_residuals

H5_FILE = "../input/datasets/eliepernet/mesh-210/AllFields_Resultats_MESH_210.h5"
MODEL_FILE = "../input/models/eliepernet/modelds/pytorch/default/1/DS_Model_03.pt"

T = 20

V_max = 0.777
P_fact = 0.521

Re = 377
nu = 1 / Re

PINNModel = nn.Sequential(
    nn.Linear(2, 256),
    nn.Tanh(),
    nn.Linear(256, 256),
    nn.Tanh(),
    nn.Linear(256, 256),
    nn.Tanh(),
    nn.Linear(256, 3)
).float()
PINNModel.load_state_dict(torch.load(MODEL_FILE))
frozen_params = {n: p.clone().detach() for n, p in PINNModel.named_parameters()}

optimizer_transfer = torch.optim.Adam(
    PINNModel.parameters(),
    lr=1e-5
)

mse = nn.MSELoss()
losses = []

x, y, _ = extract_ds_values(H5_FILE, T)

inside_points = torch.tensor(
    np.column_stack((x / 20, y / 14)),
).float()

inside_points = increase_points_number(inside_points, 4, 0.002, 0.002)
inside_points = inside_points.detach().requires_grad_(True)

inlet_points = torch.column_stack((
    torch.linspace(0.1, 0.285, 150),
    torch.zeros(150)
)).float()
inlet_points = inlet_points.detach().requires_grad_(True)

inlet_labels = torch.column_stack((
    torch.zeros(len(inlet_points)),
    parabolic_inlet(inlet_points[:, 0].detach(), V_max)
)).float().detach()

x_out, y_out = extract_ds_outside_values(H5_FILE)
outside_points = torch.tensor(
    np.column_stack((x_out / 20, y_out / 14))
).float()
outside_points = outside_points[outside_points[:, 1] > 0.001].detach()

outlet_points = torch.column_stack((
    torch.linspace(14.5 / 20, 17.85 / 20, 150),
    torch.zeros(150)
)).float().detach()


def compute_total_loss(epoch, n_epochs, lambda_anchor=5e3):
    inside_values = PINNModel(inside_points)

    # ── BC ──────────────────────────────────────────────────────────
    inlet_values = PINNModel(inlet_points)
    outside_values = PINNModel(outside_points)
    outlet_values = PINNModel(outlet_points)

    inlet_loss = mse(inlet_values[:, :2], inlet_labels)
    outside_loss = mse(outside_values[:, :2], torch.zeros_like(outside_values[:, :2]))
    outlet_loss = mse(outlet_values[:, 2], torch.full_like(outlet_values[:, 2], P_fact))

    bc_loss = 50 * outside_loss + 50 * inlet_loss + 5 * outlet_loss

    ns_x, ns_y, continuity = ns_residuals(
        inside_points,
        inside_values[:, 0], inside_values[:, 1], inside_values[:, 2],
        nu
    )
    ns_loss = mse(ns_x, torch.zeros_like(ns_x)) \
              + mse(ns_y, torch.zeros_like(ns_y))
    cont_loss = mse(continuity, torch.zeros_like(continuity))

    w_pde = min(1.0, max(0.0, (epoch - 0.2 * n_epochs) / (0.4 * n_epochs)))
    pde_loss = w_pde * (ns_loss + cont_loss)

    pinn_loss = bc_loss + pde_loss

    anchor_loss = sum(
        (p - frozen_params[n]).pow(2).sum()
        for n, p in PINNModel.named_parameters()
    )

    total = pinn_loss + lambda_anchor * anchor_loss

    losses.append([
        total.item(),
        ns_loss.item(),
        cont_loss.item(),
        inlet_loss.item(),
        outside_loss.item(),
        outlet_loss.item(),
        anchor_loss.item(),
        w_pde,
        lambda_anchor
    ])

    return total, w_pde


def get_lambda(epoch, n_epochs, lambda_start=1e5, lambda_end=1e2):
    t = epoch / n_epochs
    return lambda_start * (lambda_end / lambda_start) ** t


transfer_epochs = 8000

for epoch in range(transfer_epochs):
    optimizer_transfer.zero_grad()
    lam = get_lambda(epoch, transfer_epochs)
    loss, w_pde = compute_total_loss(epoch, transfer_epochs, lambda_anchor=lam)
    loss.backward()
    optimizer_transfer.step()

last_losses = [l[0] for l in losses[-200:]]

variation = max(last_losses) - min(last_losses)
if variation < 0.05 * np.mean(last_losses):
    print("Adam stable → lancement L-BFGS")

    for p in PINNModel.parameters():
        p.requires_grad = True

    optimizer_lbfgs = torch.optim.LBFGS(
        PINNModel.parameters(), lr=1.0,
        max_iter=500, history_size=50,
        line_search_fn="strong_wolfe"
    )

    def closure_transfer():
        optimizer_lbfgs.zero_grad()
        loss, _ = compute_total_loss(transfer_epochs, transfer_epochs, lambda_anchor=0.0)
        loss.backward()
        return loss

    for step in range(20):
        optimizer_lbfgs.step(closure_transfer)

torch.save(PINNModel.state_dict(), "./DS_Model_03.pt")

with open("DS_Model_03.json", "w") as f:
    json.dump(losses, f)

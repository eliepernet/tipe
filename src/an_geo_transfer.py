import sys

import json

import h5py
import numpy as np
import pyvista as pv
import torch
import torch.nn as nn

sys.setrecursionlimit(10000)


def ns_residuals(coords, u, v, p, nu):
    grad_u = torch.autograd.grad(u, coords, torch.ones_like(u), create_graph=True)[0]
    du_x = grad_u[:, 0]
    du_y = grad_u[:, 1]

    grad_v = torch.autograd.grad(v, coords, torch.ones_like(v), create_graph=True)[0]
    dv_x = grad_v[:, 0]
    dv_y = grad_v[:, 1]

    grad_p = torch.autograd.grad(p, coords, torch.ones_like(p), create_graph=True)[0]
    dp_x = grad_p[:, 0]
    dp_y = grad_p[:, 1]

    du_xx = torch.autograd.grad(du_x, coords, torch.ones_like(du_x), create_graph=True)[0][:, 0]
    du_yy = torch.autograd.grad(du_y, coords, torch.ones_like(du_y), create_graph=True)[0][:, 1]

    dv_xx = torch.autograd.grad(dv_x, coords, torch.ones_like(dv_x), create_graph=True)[0][:, 0]
    dv_yy = torch.autograd.grad(dv_y, coords, torch.ones_like(dv_y), create_graph=True)[0][:, 1]

    ns_x = u * du_x + v * du_y + dp_x - nu * (du_xx + du_yy)
    ns_y = u * dv_x + v * dv_y + dp_y - nu * (dv_xx + dv_yy)
    continuity = du_x + dv_y

    return ns_x, ns_y, continuity


def extract_ds_outside_values(file_name):
    with h5py.File(file_name, "r") as f:
        coords = f["data0"][:]
        connectivity = f["data1"][:]

    cells = np.hstack([np.full((len(connectivity), 1), 4), connectivity]).ravel()
    cell_types = np.full(len(connectivity), pv.CellType.TETRA)
    mesh = pv.UnstructuredGrid(cells, cell_types, coords)

    slice_plane = mesh.slice(normal='z')
    edges = slice_plane.extract_feature_edges(boundary_edges=True,
                                              non_manifold_edges=False,
                                              feature_edges=False,
                                              manifold_edges=False)
    contour_points = edges.points
    x = np.array(list(map(lambda x_c: x_c + 10, contour_points[:, 0])))
    return x, contour_points[:, 1]


def extract_ds_values(file_name, T):
    with h5py.File(file_name, "r") as f:
        coords = f["data0"][:]
        connectivity = f["data1"][:]
        vitesses = f[f"data{2 + 2 * T}"][:]

    cells = np.hstack([np.full((len(connectivity), 1), 4), connectivity]).ravel()
    cell_types = np.full(len(connectivity), pv.CellType.TETRA)
    mesh = pv.UnstructuredGrid(cells, cell_types, coords)
    mesh.point_data["Vitesse"] = vitesses

    slice_plane = mesh.slice(normal='z')

    points = slice_plane.points[:, :2]
    U = slice_plane.point_data["Vitesse"]
    u = U[:, 0]
    v = U[:, 1]

    uv_list = np.stack([u, v], axis=1)
    x = points[:, 0]
    y = points[:, 1]

    x = np.array(list(map(lambda x_c: x_c + 10, x)))
    return x, y, uv_list

rho = 1055.0
mu = 0.0035
nu = mu / rho

V_max = 1.08

U_ref = V_max
P_ref = rho * U_ref ** 2  # Pa

Re = 522

def increase_points_number(points, n_variations=8, step_size=0.05, noise_std=0.02):
    augmented = [points]

    for _ in range(n_variations):
        # Pas aléatoire dans [-step_size, step_size]
        random_step = (torch.rand_like(points) * 2 - 1) * step_size
        # Ajouter un peu de bruit
        noise = torch.randn_like(points) * noise_std
        # Nouveaux points
        new_points = points + random_step + noise
        augmented.append(new_points)

    return torch.cat(augmented, dim=0)


def parabolic_inlet(x):
    x_center = (x.min() + x.max()) / 2
    R = (x.max() - x.min()) / 2

    return V_max * (1 - ((x - x_center) / R) ** 2)


PINNModel = nn.Sequential(
    nn.Linear(2, 256),
    nn.Tanh(),
    nn.Linear(256, 256),
    nn.Tanh(),
    nn.Linear(256, 256),
    nn.Tanh(),
    nn.Linear(256, 3)
).double()
PINNModel.load_state_dict(torch.load("../models/ds_model/model_05/DS_Model_03.pt"))

mse = nn.MSELoss()

H5_FILE = "../coarse_dataset/AllFields_Resultats_MESH_205.h5"
T = 8  # pas de temps 20

x, y, _ = extract_ds_values(H5_FILE, T)

inside_points = torch.tensor(
    np.column_stack((x / 20, y / 14)),
).double()

inside_points = increase_points_number(inside_points, 2, 0.001, 0.001)
inside_points = inside_points.detach().requires_grad_(True)

inlet_points = torch.column_stack((
    torch.linspace(0.1, 0.285, 150),
    torch.zeros(150)
)).double()
inlet_points = inlet_points.detach().requires_grad_(True)

inlet_labels = torch.column_stack((
    torch.zeros(len(inlet_points)),
    parabolic_inlet(inlet_points[:, 0].detach())
)).double().detach()

x_out, y_out = extract_ds_outside_values(H5_FILE)
outside_points = torch.tensor(
    np.column_stack((x_out / 20, y_out / 14))
).double()
outside_points = outside_points[outside_points[:, 1] > 0.001].detach()

outlet_points = torch.column_stack((
    torch.linspace(14.5 / 20, 17.85 / 20, 150),
    torch.zeros(150)
)).double().detach()

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
    outlet_loss = mse(outlet_values[:, 2], torch.ones_like(outlet_values[:, 2]))

    total = (
            ns_loss
            + cont_loss
            + 50 * wall_loss
            + 50 * inlet_loss
            + 5 * outlet_loss
    )

    losses.append([ns_loss.item(), cont_loss.item(), wall_loss.item(),
                   inlet_loss.item(), outlet_loss.item(), total.item()])
    return total


optimizer_lbfgs = torch.optim.LBFGS(
    PINNModel.parameters(),
    lr=0.1,
    max_iter=50,
    history_size=50,
    tolerance_grad=1e-7,
    tolerance_change=1e-9,
    line_search_fn="strong_wolfe"
)

for epoch in range(200):
    def closure():
        optimizer_lbfgs.zero_grad()
        loss = loss_function()
        loss.backward()
        return loss


    optimizer_lbfgs.step(closure)

    if epoch % 20 == 0:
        print(f"[{epoch}] loss = {losses[-1][-1]:.4e}")

torch.save(PINNModel.state_dict(), "./FT_model_01.pt")

with open("FT_model_01.json", "w") as f:
    json.dump(losses, f)

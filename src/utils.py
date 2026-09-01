import json

import h5py
import torch
import pyvista as pv
import numpy as np
from matplotlib import pyplot as plt, tri
import matplotlib.colors as mcolors


def kova_solution_np(x, y, nu):
    l = 1 / (2 * nu) - np.sqrt(1 / (4 * nu**2) + 4 * np.pi**2)

    u = 1 - np.exp(l * x) * np.cos(2 * np.pi * y)
    v = (l / (2 * np.pi)) * np.exp(l * x) * np.sin(2 * np.pi * y)
    p = 0.5 * (1 - np.exp(2 * l * x))

    return np.stack([u, v, p], axis=1)


def kova_solution(x, y, nu) -> torch.Tensor:
    result_np = kova_solution_np(x.detach().cpu().numpy(), y.detach().cpu().numpy(), nu)

    return torch.tensor(result_np, dtype=x.dtype, device=x.device)


def load_losses(filename):
    losses = []

    with open(filename, "r") as f:
        losses = [i for i in json.load(f) if np.shape(i)]

    return np.array(losses)


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


def plot_field(x, y, z, plot_title, z_label, show_norm=True):
    triang = tri.Triangulation(x, y)
    triangles = triang.triangles

    x_tri = x[triangles]
    y_tri = y[triangles]

    a = np.sqrt((x_tri[:, 1] - x_tri[:, 0]) ** 2 + (y_tri[:, 1] - y_tri[:, 0]) ** 2)
    b = np.sqrt((x_tri[:, 2] - x_tri[:, 1]) ** 2 + (y_tri[:, 2] - y_tri[:, 1]) ** 2)
    c = np.sqrt((x_tri[:, 0] - x_tri[:, 2]) ** 2 + (y_tri[:, 0] - y_tri[:, 2]) ** 2)

    max_edge = np.max(np.stack([a, b, c], axis=1), axis=1)

    threshold = 0.05
    triang.set_mask(max_edge > threshold)

    # --- Plot ---
    fig, ax = plt.subplots()
    if show_norm:
        norm = mcolors.TwoSlopeNorm(vmin=z.min(), vcenter=0, vmax=z.max())
        tcf = ax.tripcolor(triang, z, cmap='RdBu_r', shading='gouraud', norm=norm)
    else:
        tcf = ax.tripcolor(triang, z, cmap='viridis', shading='gouraud')
    ax.set_aspect('auto')
    ax.set_xlim(x.min() - 0.05, x.max() + 0.05)
    ax.set_ylim(y.min() - 0.05, y.max() + 0.05)
    plt.colorbar(tcf, ax=ax, label=z_label)
    plt.title(plot_title)
    plt.tight_layout()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def kova_data():
    outside_points = torch.Tensor(
        [[-0.5, i] for i in np.linspace(-0.5, 1.5, 25)] +
        [[1, i] for i in np.linspace(-0.5, 1.5, 25)] +
        [[i, -0.5] for i in np.linspace(-0.5, 1,25)] +
        [[i, 1.5] for i in np.linspace(-0.5, 1, 25)]).double()
    inside_points = torch.tensor(
        np.column_stack((np.random.uniform(-0.5, 1.0, 2000), np.random.uniform(-0.5, 1.5, 2000)))
    ).double().requires_grad_(True)

    return outside_points, inside_points

def u_func(x, y):
    return 1 - torch.exp(l * x) * torch.cos(2 * torch.pi * y)

def v_func(x, y):
    return (l / (2 * torch.pi)) * torch.exp(l * x) * torch.sin(2 * np.pi * y)

def p_func(x):
    return 1 / 2 * (1 - torch.exp(2 * l * x))

def parabolic_inlet(x, V_max):
    x_center = (x.min() + x.max()) / 2
    R = (x.max() - x.min()) / 2

    return V_max * (1 - ((x - x_center) / R) ** 2)

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
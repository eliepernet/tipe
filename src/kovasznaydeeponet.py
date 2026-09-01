import torch
import torch.nn as nn
import numpy as np

import json

from models import KovaDeepONet
from utils import kova_solution, ns_residuals


def loss_function(re_values):
    ns_total_loss = torch.tensor(0.0).double()
    bc_total_loss = torch.tensor(0.0).double()

    for re in re_values:
        re_t = torch.tensor([re]).double()
        nu = 1.0 / re

        inside_points = torch.tensor(
            np.column_stack((np.random.uniform(-0.5, 1.0, 2000), np.random.uniform(-0.5, 1.5, 2000)))
        ).double().requires_grad_(True)

        inside_results = model(re_t, inside_points).squeeze(0)

        ns_x, ns_y, continuity = ns_residuals(inside_points, inside_results[:, 0], inside_results[:, 1], inside_results[:, 2], nu)
        ns_total_loss += (mse(ns_x, torch.zeros_like(ns_x)) + mse(ns_y, torch.zeros_like(ns_y)) + mse(continuity, torch.zeros_like(continuity)))

        outside_results = model(re, outside_points).squeeze(0)
        labels = kova_solution(outside_points[:, 0], outside_points[:, 1], nu)
        bc_total_loss += mse(outside_results, labels)

    n = len(re_values)
    return LAMBDA_NS * ns_total_loss / n, LAMBDA_BC * bc_total_loss / n

mse = nn.MSELoss()

P          = 128    # Dimension de l'espace latent branch/trunk
N_COLLOC   = 2000   # Points intérieurs par Re
N_BC       = 25     # Points par côté frontière
RE_RANGE   = (20, 100)

LAMBDA_NS = 1.0
LAMBDA_BC = 50.0

RE_BATCH_SIZE = 8

outside_points = torch.Tensor([[-0.5, i] for i in np.linspace(-0.5, 1.5, 25)] + [[1, i] for i in np.linspace(-0.5, 1.5, 25)] + [[i, -0.5] for i in np.linspace(-0.5, 1, 25)] + [[i, 1.5] for i in np.linspace(-0.5, 1, 25)]).double()

model = KovaDeepONet().double()

losses = []

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)

for epoch in range(5000):
    # Échantillonnage aléatoire de Re à chaque époque
    re_batch = np.random.uniform(*RE_RANGE, RE_BATCH_SIZE).tolist()

    optimizer.zero_grad()
    loss_ns, loss_bc = loss_function(re_batch)
    loss = loss_ns + loss_bc
    losses.append([loss_ns.item(), loss_bc.item(), loss.item()])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if epoch % 500 == 0:
        print(f"[{epoch:5d}] NS={loss_ns:.2e} | BC={loss_bc:.2e}")


torch.save(model.state_dict(), "./DON_kova.pt")

with open("losses_09.json", "w") as f:
    json.dump(losses, f)

import torch.nn as nn
import torch


class KovaDeepONet(nn.Module):
    def __init__(self, p=128, re_min=20, re_max=100):
        super(KovaDeepONet, self).__init__()
        self.p = p
        self.re_min = re_min
        self.re_max = re_max

        self.trunk_net = MLP(4, 128, 2, p, final_tanh=False)
        self.branch_net = MLP(4, 128, 1, p * 3)

        self.bias = nn.Parameter(torch.zeros(3).double())
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, re, points):
        re_norm = 2 * (re - self.re_min) / (self.re_max - self.re_min) - 1

        b = self.branch_net(re_norm.unsqueeze(-1)).reshape(-1, 3, self.p)
        t = self.trunk_net(points)

        out = torch.einsum('bcp,np->bcn', b, t)
        return out.permute(0, 2, 1) + self.bias

class MLP(nn.Module):
    def __init__(self, hidden_n, layer_depth, input_n, output_n, final_tanh=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.final_tanh = final_tanh

        self.layers.append(nn.Linear(input_n, layer_depth))

        for i in range(hidden_n):
            self.layers.append(nn.Linear(layer_depth, layer_depth))

        self.layers.append(nn.Linear(layer_depth, output_n))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = torch.tanh(self.layers[i](x))

        x = self.layers[-1](x)
        if self.final_tanh:
            x = torch.tanh(x)  # Tanh final optionnel
        return x

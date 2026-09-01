class FourierEmbedding(nn.Module):
    def __init__(self, in_dim=2, m=64, sigma=1.0):
        super().__init__()
        # B est fixe (non entraînable) — tiré une fois
        B = torch.randn(in_dim, m) * sigma
        self.register_buffer('B', B)

    def forward(self, x):
        # x : (N, 2) → projection : (N, m)
        proj = x @ self.B          # (N, m)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        # sortie : (N, 2m)
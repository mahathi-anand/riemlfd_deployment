"""Inference-only RiemannianLfD models in plain PyTorch, loaded from the checkpoints in bundles/.

The modules mirror the training code's stochman.nnj layers name for name, so the checkpoints'
state dicts load unchanged:
    VAE  encoder_loc  [BatchNorm, Residual(in->out)] per layer, then Linear(d, d)
         decoder_loc  the reverse without the first BatchNorm, then Linear(D, D)
         dec_std_pos  RBF -> PosLinear(K, 1) -> Reciprocal -> PosLinear(1, pos_dof)
         dec_std_qua  RBF -> PosLinear(K, qua_dof)
    NODE ode_func.net Linear(2d, H) -> ELU -> Linear(H, H) -> ELU -> Linear(H, H) -> ELU -> Linear(H, d)
All sizes are read from the checkpoint shapes. Only what deployment needs is implemented: the
encoder mean, the decoder mean (position, unit quaternion) and the metric G = J^T J of the
embedding [mu_pos, mu_qua, std_pos, 1 / std_qua] (encoder_scale is training-only and skipped).

The full model, including the training pipeline exists at https://github.com/DianaCuervo/RiemannianLfD
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# stochman's Reciprocal(inv_max_std) in VAE.init_std, inv_max_std = sqrt(1e-12)
RECIPROCAL_B = 1e-6


class Flat(nn.Sequential):
    """nnj.Sequential: run the layers on [-1, d_in] and restore the leading dimensions."""

    def forward(self, x):
        lead = x.shape[:-1]
        return super().forward(x.reshape(-1, x.shape[-1])).reshape(*lead, -1)


class Residual(nn.Module):
    """nnj.ResidualBlock(Linear(i, o), Softplus()): projection(x) + softplus(F(x))."""

    def __init__(self, i, o):
        super().__init__()
        self._F = nn.Sequential(nn.Linear(i, o), nn.Softplus())
        self._projection = nn.Linear(i, o)

    def forward(self, x):
        return self._projection(x) + self._F(x)


class RBF(nn.Module):
    """nnj.RBF: exp(-beta * |x - c_k|^2); beta is not in the state dict (set from rbf_beta)."""

    def __init__(self, dim, k, beta):
        super().__init__()
        self.points = nn.Parameter(torch.zeros(k, dim), requires_grad=False)
        self.beta = beta

    def forward(self, x):
        d2 = (x ** 2).sum(1, keepdim=True) + (self.points ** 2).sum(1)[None] - 2.0 * x @ self.points.T
        return torch.exp(-self.beta * d2.clamp(min=0.0))


class PosLinear(nn.Linear):
    """nnj.PosLinear: a linear layer with softplus-positive weights and bias."""

    def forward(self, x):
        return F.linear(x, F.softplus(self.weight), None if self.bias is None else F.softplus(self.bias))


class Reciprocal(nn.Module):
    def forward(self, x):
        return 1.0 / (x + RECIPROCAL_B)


class VAE(nn.Module):
    def __init__(self, layers, pos_dof, n_clusters, rbf_beta):
        super().__init__()
        self.d, self.pos_dof = layers[-1], pos_dof
        qua_dof = layers[0] - pos_dof
        enc = []
        for i, o in zip(layers[:-1], layers[1:]):
            enc += [nn.BatchNorm1d(i), Residual(i, o)]
        self.encoder_loc = Flat(*enc, nn.Linear(layers[-1], layers[-1]))
        dec = []
        for i, o in zip(layers[:0:-1], layers[-2::-1]):
            dec += [nn.BatchNorm1d(i), Residual(i, o)]
        self.decoder_loc = Flat(*dec[1:], nn.Linear(layers[0], layers[0]))
        self.dec_std_pos = Flat(RBF(self.d, n_clusters, rbf_beta), PosLinear(n_clusters, 1, bias=False),
                                Reciprocal(), PosLinear(1, pos_dof))
        self.dec_std_qua = Flat(RBF(self.d, n_clusters, rbf_beta), PosLinear(n_clusters, qua_dof))

    def encode(self, x):
        """Latent mean of input points [..., D]."""
        return self.encoder_loc(x)

    def decode(self, z):
        """Decoded position mean [..., pos_dof] and unit quaternion [..., qua_dof] of latents [..., d]."""
        x = self.decoder_loc(z)
        q = x[..., self.pos_dof:]
        return x[..., :self.pos_dof], q / q.norm(dim=-1, keepdim=True)

    def embed(self, z):
        pos, q = self.decode(z)
        return torch.cat([pos, q, self.dec_std_pos(z), 1.0 / self.dec_std_qua(z)], dim=-1)

    def metric(self, z):
        """G = J^T J of the embedding at latents [N, d], as [N, d, d]."""
        eye = torch.eye(self.d, dtype=z.dtype)
        J = torch.stack([torch.func.jvp(self.embed, (z,), (eye[k].expand_as(z),))[1] for k in range(self.d)], -1)
        return J.transpose(-1, -2) @ J


class ODEFunc(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * latent_dim, hidden_dim), nn.ELU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
                                 nn.Linear(hidden_dim, latent_dim))

    def forward(self, t, z, z_goal):
        """Latent velocity dz/dt at z, conditioned on the goal latent (t is unused)."""
        return self.net(torch.cat([z, z_goal.expand_as(z)], dim=-1))


class NODE(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.ode_func = ODEFunc(latent_dim, hidden_dim)


def load_vae(path):
    """The VAE of a training checkpoint (needs model_state_dict and rbf_beta), in eval mode."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "rbf_beta" not in ckpt:
        raise ValueError(f"{path} has no 'rbf_beta'; re-export it with scripts/export_bundles.py")
    sd = ckpt["model_state_dict"]
    layers = [sd["encoder_loc.0.weight"].shape[0]]  # D, then the out size of each residual block
    k = 1
    while f"encoder_loc.{k}._projection.weight" in sd:
        layers.append(sd[f"encoder_loc.{k}._projection.weight"].shape[0])
        k += 2
    pos_dof = sd["dec_std_pos.3.weight"].shape[0]
    n_clusters = sd["dec_std_pos.0.points"].shape[0]
    vae = VAE(layers, pos_dof, n_clusters, float(ckpt["rbf_beta"]))
    used = vae.state_dict().keys()
    vae.load_state_dict({key: v for key, v in sd.items() if key in used})
    skipped = {key.split(".")[0] for key in sd.keys() - used}
    if skipped - {"encoder_scale", "encoder_scale_fixed", "decoder_scale_pos", "decoder_scale_qua",
                  "prior_loc", "prior_scale"}:
        raise ValueError(f"{path}: unexpected parameters {sorted(skipped)}")
    return vae.eval()


def load_node(path):
    """The NODE of a training checkpoint, latent and hidden sizes read from its weights, in eval mode."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    hidden_dim, latent_dim = sd["ode_func.net.0.weight"].shape[0], sd["ode_func.net.6.weight"].shape[0]
    node = NODE(latent_dim, hidden_dim)
    node.load_state_dict(sd)
    return node.eval()

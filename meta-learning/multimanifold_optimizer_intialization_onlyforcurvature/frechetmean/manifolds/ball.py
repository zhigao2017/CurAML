import torch

from .manifold import Manifold
from frechetmean.utils import EPS, cosh, sinh, tanh, arcosh, arsinh, artanh, sinhdiv, divsinh


class Poincare(Manifold):
    def __init__(self, K=-1.0, edge_eps=1e-3):
        super(Poincare, self).__init__()
        self.edge_eps = 1e-3
        assert K < 0
        if torch.is_tensor(K):
            self.K = K
        else:
            self.K = torch.tensor(K)

    def sh_to_dim(self, sh):
        if hasattr(sh, '__iter__'):
            return sh[-1]
        else:
            return sh

    def dim_to_sh(self, dim):
        if hasattr(dim, '__iter__'):
            return dim[-1]
        else:
            return dim

    def zero(self, *shape):
        return torch.zeros(*shape)

    def zero_tan(self, *shape):
        return torch.zeros(*shape)

    def zero_like(self, x):
        return torch.zeros_like(x)

    def zero_tan_like(self, x):
        return torch.zeros_like(x)

    def lambda_x(self, x, k, keepdim=False):
        return 2 / (1 + k * x.pow(2).sum(dim=-1, keepdim=keepdim)).clamp_min(min=EPS[x.dtype])

    def inner(self, x, u, v, k, keepdim=False):
        return self.lambda_x(x, k, keepdim=True).pow(2) * (u * v).sum(dim=-1, keepdim=keepdim)

    def proju(self, x, u, k):
        return u

    def projx(self, x, k):
        norm = x.norm(dim=-1, keepdim=True).clamp(min=EPS[x.dtype])
        maxnorm = (1 - self.edge_eps) / (-k).sqrt()
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def egrad2rgrad(self, x, u, k):
        return u / self.lambda_x(x, k, keepdim=True).pow(2)

    def mobius_addition(self, x, y, k):
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        y2 = y.pow(2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 - 2 * k * xy - k * y2) * x + (1 + k * x2) * y
        denom = 1 - 2 * k * xy + (k.pow(2)) * x2 * y2
        return num / denom.clamp_min(EPS[x.dtype])

    def exp(self, x, u, k):
        '''
        u_norm = u.norm(dim=-1, keepdim=True).clamp_min(min=EPS[x.dtype])
        second_term = (
            tanh((-k).sqrt() / 2 * self.lambda_x(x, k, keepdim=True) * u_norm) * u / ((-k).sqrt() * u_norm)
        )
        gamma_1 = self.mobius_addition(x, second_term, k)
        '''  
        gamma_1 = self.mobius_addition(x, (
            tanh((-k).sqrt() / 2 * self.lambda_x(x, k, keepdim=True) * u.norm(dim=-1, keepdim=True).clamp_min(min=EPS[x.dtype])) * u / ((-k).sqrt() * u.norm(dim=-1, keepdim=True).clamp_min(min=EPS[x.dtype]))
        ), k)
        return gamma_1

    def log(self, x, y, k):
        '''
        sub = self.mobius_addition(-x, y, k)
        sub_norm = sub.norm(dim=-1, keepdim=True).clamp_min(EPS[x.dtype])
        lam = self.lambda_x(x, k, keepdim=True)
        return 2 / ((-k).sqrt() * lam) * artanh((-k).sqrt() * sub_norm) * sub / sub_norm
        '''
        return 2 / ((-k).sqrt() * self.lambda_x(x, k, keepdim=True)) * artanh((-k).sqrt() * self.mobius_addition(-x, y, k).norm(dim=-1, keepdim=True).clamp_min(EPS[x.dtype])) * self.mobius_addition(-x, y, k) / self.mobius_addition(-x, y, k).norm(dim=-1, keepdim=True).clamp_min(EPS[x.dtype])

    def dist(self, x, y, k, squared=False, keepdim=False):
        dist = 2 * artanh((-k).sqrt() * self.mobius_addition(-x, y, k).norm(dim=-1)) / (-k).sqrt()
        return dist.pow(2) if squared else dist

    def _gyration(self, u, v, w, k):
        u2 = u.pow(2).sum(dim=-1, keepdim=True)
        v2 = v.pow(2).sum(dim=-1, keepdim=True)
        uv = (u * v).sum(dim=-1, keepdim=True)
        uw = (u * w).sum(dim=-1, keepdim=True)
        vw = (v * w).sum(dim=-1, keepdim=True)
        a = - k.pow(2) * uw * v2 - k * vw + 2 * k.pow(2) * uv * vw
        b = - k.pow(2) * vw * u2 + k * uw
        d = 1 - 2 * k * uv + k.pow(2) * u2 * v2
        return w + 2 * (a * u + b * v) / d.clamp_min(EPS[u.dtype])

    def transp(self, x, y, u, k):
        return (
            self._gyration(y, -x, u, k)
            * self.lambda_x(x, k, keepdim=True)
            / self.lambda_x(y, k, keepdim=True)
        )


    def __str__(self):
        return 'Poincare Ball'

    def squeeze_tangent(self, x):
        return x

    def unsqueeze_tangent(self, x):
        return x

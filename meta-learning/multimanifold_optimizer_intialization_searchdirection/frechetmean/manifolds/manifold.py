import abc

import numpy as np
import torch

from frechetmean.utils import EPS


class Manifold(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def zero(self, *shape):
        pass

    @abc.abstractmethod
    def zero_like(self, x):
        pass

    @abc.abstractmethod
    def zero_tan(self, *shape):
        pass

    @abc.abstractmethod
    def zero_tan_like(self, x):
        pass

    @abc.abstractmethod
    def inner(self, x, u, v, k,keepdim=False):
        pass

    def norm(self, x, u, k, squared=False, keepdim=False):
        norm_sq = self.inner(x, u, u, k,keepdim)
        norm_sq.data.clamp_(EPS[u.dtype])
        return norm_sq if squared else norm_sq.sqrt()

    @abc.abstractmethod
    def proju(self, x, u,k):
        pass

    def proju0(self, u,k):
        return self.proju(self.zero_like(u), u, k)

    @abc.abstractmethod
    def projx(self, x,k):
        pass

    def egrad2rgrad(self, x, u, k):
        return self.proju(x, u, k)

    @abc.abstractmethod
    def exp(self, x, u, k):
        pass

    def exp0(self, u,k):
        return self.exp(self.zero_like(u), u, k)

    @abc.abstractmethod
    def log(self, x, y,k):
        pass

    def log0(self, y,k):
        return self.log(self.zero_like(y), y,k)
        
    def dist(self, x, y, k, squared=False, keepdim=False):
        return self.norm(x, self.log(x, y, k), k, squared, keepdim)

    def pdist(self, x, k,squared=False):
        assert x.ndim == 2
        n = x.shape[0]
        m = torch.triu_indices(n, n, 1, device=x.device)
        return self.dist(x[m[0]], x[m[1]], k, squared=squared, keepdim=False)

    def transp(self, x, y, u, k):
        return self.proju(y, u, k)

    def transpfrom0(self, x, u, k):
        return self.transp(self.zero_like(x), x, u, k)
    
    def transpto0(self, x, u, k):
        return self.transp(x, self.zero_like(x), u, k)

    def mobius_addition(self, x, y, k):
        return self.exp(x, self.transp(self.zero_like(x), x, self.log0(y)), k )

    @abc.abstractmethod
    def sh_to_dim(self, shape):
        pass

    @abc.abstractmethod
    def dim_to_sh(self, dim):
        pass

    @abc.abstractmethod
    def squeeze_tangent(self, x):
        pass

    @abc.abstractmethod
    def unsqueeze_tangent(self, x):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass

    def frechet_variance(self, x, mu, k, w=None):
        """
        Args
        ----
            x (tensor): points of shape [..., points, dim]
            mu (tensor): mean of shape [..., dim]
            w (tensor): weights of shape [..., points]

            where the ... of the three variables line up
        
        Returns
        -------
            tensor of shape [...]
        """
        distance = self.dist(x, mu.unsqueeze(-2), k, squared=True)
        if w is None:
            return distance.mean(dim=-1)
        else:
            return (distance * w).sum(dim=-1)

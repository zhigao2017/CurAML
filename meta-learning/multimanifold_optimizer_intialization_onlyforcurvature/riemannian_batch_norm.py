import torch
import torch.nn as nn
import math

from frechetmean.manifolds import Poincare, Lorentz
from frechetmean.frechet import frechet_mean
from hyptorch.pmath import poincare_mean
from hyptorch.pmath import dist_matrix, expmap, expmap0, logmap, logmap0, frechet_variance, transp

class RiemannianBatchNorm(nn.Module):
    def __init__(self, manifold):
        super(RiemannianBatchNorm, self).__init__()

    def forward(self, x, updates,running_mean, running_var, bn_weight, bn_biased, k, training=True, momentum=0.9):
        #on_manifold = self.man.exp0(bn_weight,k)
        on_manifold = expmap0(bn_weight,c=-k)
        #print('in BN after on_manifold',torch.sum(torch.isnan(on_manifold))) 
        if training:
            # frechet mean, use iterative and don't batch (only need to compute one mean)
            #input_mean = frechet_mean(x, self.man, k)
            #print('in BN before x',torch.sum(torch.isnan(x)),x) 
            input_mean = poincare_mean(x,dim=0,c=-k)
            #print('in BN after input_mean',torch.sum(torch.isnan(input_mean)),input_mean) 
            input_var = frechet_variance(x, input_mean, c=-k)
            #print('in BN after input_var',torch.sum(torch.isnan(input_var)),input_var)  
            # transport input from current mean to learned mean
            #print('in BN after input_logm1',torch.sum(torch.isnan(input_logm)),input_logm)
            # re-scaling
            #print('in BN after input_logm1 bn_biased',bn_biased)
            #print('in BN after input_logm2',torch.sum(torch.isnan(input_logm)),input_logm)
            # project back
            trans=(bn_biased / (input_var + 1e-6)).sqrt() * transp( input_mean, on_manifold, logmap(input_mean, x, c=-k), k)
            output = expmap(on_manifold, trans, c=-k)
            #print('in BN after output',torch.sum(torch.isnan(output)),output)
            updates += 1

            running_mean = expmap(
                running_mean,
                (1 - momentum) * logmap(running_mean, input_mean, c=-k), c=-k
            )
            
            running_var = (
                1 - 1 / updates
            ) * running_var + input_var / updates

        else:
            if updates == 0:
                raise ValueError("must run training at least once")

            #input_mean = frechet_mean(x, self.man, k)
            input_mean = poincare_mean(x,dim=0,c=-k)
            input_var = frechet_variance(x, input_mean, c=-k)

            input_logm = transp(
                input_mean,
                running_mean,
                logmap(input_mean, x, c=-k), k
            )

            assert not torch.any(torch.isnan(input_logm))

            # re-scaling
            input_logm = (
                running_var / (x.shape[0] / (x.shape[0] - 1) * input_var + 1e-6)
            ).sqrt() * input_logm

            # project back
            output = expmap(on_manifold, input_logm, c=-k)

        return output

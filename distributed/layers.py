from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import comm

from torch.cuda import amp

from networks.helpers import trunc_normal_

# matmul parallel
from distributed.mappings import copy_to_parallel_region
from distributed.mappings import gather_from_parallel_region, reduce_from_parallel_region
from typing import Tuple

class DistributedMatmul(nn.Module):
    """Distributed Matrix Multiply"""

    def __init__(
        self,
        inp_dim,
        out_dim,
        comm_inp_name,
        comm_out_name,
        bias=True,
    ):
        super(DistributedMatmul, self).__init__()

        # get sizes
        self.comm_inp_name = comm_inp_name
        self.comm_out_name = comm_out_name
        comm_inp_size = comm.get_size(self.comm_inp_name)
        comm_out_size = comm.get_size(self.comm_out_name)

        assert (
            inp_dim % comm_inp_size == 0
        ), f"Error, the size of input feature dim ({inp_dim}) has to be evenly divisible by the input feature comm dim ({comm_inp_size})"
        assert (
            out_dim % comm_out_size == 0
        ), f"Error, the size of output feature dim ({out_dim}) has to be evenly divisible by the output feature comm dim ({comm_out_size})"

        # compute reduced dims
        inp_dim_local = inp_dim // comm_inp_size
        out_dim_local = out_dim // comm_out_size

        # parameters
        comm_names_shared = [c for c in comm.get_names(meta=False) if c not in [comm_inp_name, comm_out_name]]
        self.weight = nn.Parameter(torch.ones(out_dim_local, inp_dim_local))
        self.weight.is_shared_mp = comm_names_shared
        self.weight.sharded_dims_mp = [
            self.comm_out_name,
            self.comm_inp_name,
            None,
            None,
        ]
        if bias:
            self.bias = nn.Parameter(torch.ones(1, 1, out_dim_local))
            self.bias.is_shared_mp = comm_names_shared
            self.bias.sharded_dims_mp = [None, self.comm_out_name, None, None]

        # init weights
        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.weight, std=0.02)
        if hasattr(self, "bias"):
            nn.init.constant_(self.bias, 0.0)

    # since this method is full of custom autograd, it cannot be jitted from torch frontend.
    @torch.jit.ignore
    def forward(self, x):
        x_cp = copy_to_parallel_region(x, self.comm_out_name)
        x_loc = F.linear(x_cp, self.weight, bias=None)
        x_out = reduce_from_parallel_region(x_loc, self.comm_inp_name)
        if hasattr(self, "bias"):
            x_out = x_out + self.bias
        return x_out

    
class DistributedMLP(nn.Module):
    """Distributed MLP layer"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        comm_inp_name="col_matmul",
        comm_hidden_name="row_matmul",
        act_layer=nn.GELU,
        drop=0.0
    ):

        super(DistributedMLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # get effective embedding size:
        comm_inp_size = comm.get_size(comm_inp_name)
        comm_hid_size = comm.get_size(comm_hidden_name)

        self.fc1 = DistributedMatmul(
            in_features,
            hidden_features,
            comm_inp_name=comm_inp_name,
            comm_out_name=comm_hidden_name,
            bias=True,
        )

        self.fc2 = DistributedMatmul(
            hidden_features,
            out_features,
            comm_inp_name=comm_hidden_name,
            comm_out_name=comm_inp_name,
            bias=True,
        )

        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DistributedLayerNorm(nn.Module):
    """Distributed LayerNorm layer"""

    def __init__(
            self,
            normalized_shape,
            comm_names,
            comm_name_meta=None,
            eps=1e-05,
            elementwise_affine=True):

        super(DistributedLayerNorm, self).__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        
        # this can be tricky for arbitrary shapes, we would need to
        # make sure the comm names we give it are correct:
        # make sure each dim is associated with a comm, none is allowed:
        assert(len(comm_names) == len(normalized_shape))
        self.normalized_shape = normalized_shape
        self.normalized_dims = [i for i in range(-len(self.normalized_shape), 0)]
        self.comm_names = comm_names
        self.comm_name_meta = comm_name_meta
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        self.gather_mode = 'welford'
        
        # get local shapes
        self.normalized_shape_local = [s // comm.get_size(c) for s,c in zip(self.normalized_shape, self.comm_names)]
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(*self.normalized_shape_local))
            self.bias = nn.Parameter(torch.ones(*self.normalized_shape_local))

            # set sharing
            comm_names_shared = [c for c in comm.get_names(meta=False) if c not in comm_names]
            self.weight.is_shared_mp = comm_names_shared
            self.bias.is_shared_mp = comm_names_shared
            

    def _stats_naive(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the statistics in the naive way by first gathering the tensors and then computing them"""
        if self.comm_name_meta is None:
            for dim, cname in zip(self.normalized_dims, self.comm_names):
                x = gather_from_parallel_region(x, dim, cname)
            var, mean = torch.var_mean(x, dim=self.normalized_dims, unbiased=False, keepdim=True)
        else:
            x = x.unsqueeze(0)
            x = gather_from_parallel_region(x, 0, self.comm_name_meta)
            var, mean = torch.var_mean(x, dim=0, unbiased=False, keepdim=False)            

        return var, mean

    def _stats_welford(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the statistics locally, then uses the Welford online algorithm to reduce them"""
        var, mean = torch.var_mean(x, dim=self.normalized_dims, unbiased=False, keepdim=True)
        # workaround to not use shapes, as otherwise cuda graphs won't work
        with torch.no_grad():
            count = torch.ones_like(x, requires_grad=False)
            count = torch.sum(count, dim=self.normalized_dims, keepdim=True)

        if self.comm_name_meta is None:
            for dim, cname in zip(self.normalized_dims, self.comm_names):
                vars = gather_from_parallel_region(var, dim, cname)
                means = gather_from_parallel_region(mean, dim, cname)
                counts = gather_from_parallel_region(count, dim, cname)
                m2s = vars * counts

                m2s = [x.squeeze(dim) for x in torch.split(m2s, 1, dim)]
                means = [x.squeeze(dim) for x in torch.split(means, 1, dim)]
                counts = [x.squeeze(dim) for x in torch.split(counts, 1, dim)]

                # initial values
                mean = means[0]
                m2 = m2s[0]
                count = counts[0]

                # use Welford's algorithm to accumulate them into a single mean and variance
                for meani, m2i, counti in zip(means[1:], m2s[1:], counts[1:]):
                    delta = meani - mean
                    m2 = m2 + m2i + delta**2 * count * counti / (count + counti)
                    mean = mean + delta * counti / (count + counti)

                    # update the current count
                    count = count + counti

        else:
            vars = gather_from_parallel_region(var.unsqueeze(0), 0, self.comm_name_meta)
            means = gather_from_parallel_region(mean.unsqueeze(0), 0, self.comm_name_meta)
            counts = gather_from_parallel_region(count.unsqueeze(0), 0, self.comm_name_meta)

            m2s = vars * counts

            # split into lists
            m2s = [x.squeeze(0) for x in torch.split(m2s, 1, 0)]
            means = [x.squeeze(0) for x in torch.split(means, 1, 0)]
            counts = [x.squeeze(0) for x in torch.split(counts, 1, 0)]

            # initial values
            mean = means[0]
            m2 = m2s[0]
            count = counts[0]

            # use Welford's algorithm to accumulate them into a single mean and variance
            for meani, m2i, counti in zip(means[1:], m2s[1:], counts[1:]):                
                delta = meani - mean
                m2 = m2 + m2i + delta**2 * count * counti / (count + counti)
                mean = mean + delta * counti / (count + counti)
                
		# update the current count
                count = count + counti

        # finalize
        var = m2 / count

        return var, mean


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        with amp.autocast(enabled=False):
            dtype = x.dtype
            x = x.float()

            # start by computing std and mean
            if self.gather_mode == 'naive':
                var, mean = self._stats_naive(x)
            elif self.gather_mode == 'welford':
                var, mean = self._stats_welford(x)
            else:
                raise ValueError(f"Unknown gather mode {self.gather_mode}")

            # this is absolutely necessary to get the correct graph in the backward pass
            if self.comm_name_meta is None:
                for cname in self.comm_names:
                    mean = copy_to_parallel_region(mean, cname)
                    var = copy_to_parallel_region(var, cname)
            else:
                mean = copy_to_parallel_region(mean, self.comm_name_meta)
                var = copy_to_parallel_region(var, self.comm_name_meta)
                
        # convert back
        x = x.to(dtype)
        mean = mean.to(dtype)
        var = var.to(dtype)

        # apply the normalization
        x = (x - mean) / torch.sqrt(var + self.eps)
        
        # affine transform if we use it
        if self.elementwise_affine:
            # reshape parameters
            padlen = x.dim() - len(self.normalized_shape_local)
            newshape = [1 for _ in range(padlen)] + [*self.normalized_shape_local]
            x = self.weight.reshape(*newshape) * x + self.bias.reshape(*newshape)

        return x

    
class DistributedAttention(nn.Module):
    """Distributed Attention layer"""

    def __init__(
            self,
            dim,
            comm_inp_name,
            comm_hidden_name,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):

        super(DistributedAttention, self).__init__()

        assert(dim % num_heads == 0), 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        assert( (num_heads % comm.get_size(comm_hidden_name)) == 0), 'num_heads should be divisible by size of comm_hidden_name'
        self.num_heads_local = num_heads // comm.get_size(comm_hidden_name)
        self.head_dim = dim // num_heads
        self.dim_local = self.head_dim * self.num_heads_local
        #self.scale = 1. / sqrt(self.head_dim)

        self.comm_inp_name = comm_inp_name
        self.comm_hidden_name = comm_hidden_name
        
        self.qkv = DistributedMatmul(dim, dim * 3, comm_inp_name, comm_hidden_name, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.dim_local) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.dim_local) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = DistributedMatmul(dim, dim, comm_hidden_name, comm_inp_name, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        
        # qkx
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads_local, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p,
        )

        x = x.transpose(1, 2).reshape(B, -1, self.dim_local)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
        

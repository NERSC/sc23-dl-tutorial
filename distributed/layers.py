import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import comm

from networks.helpers import trunc_normal_

# matmul parallel
from distributed.mappings import copy_to_parallel_region
from distributed.mappings import reduce_from_parallel_region


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
        comm_names_shared = [c for c,_ in comm.get_names(meta=False).items() if c not in [comm_inp_name, comm_out_name]]
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
            eps=1e-05,
            elementwise_affine=True):

        super(self, DistributedLayerNorm).__init__()
        # this can be tricky for arbitrary shapes, we would need to
        # make sure the comm names we give it are correct:
        # make sure each dim is associated with a comm, none is allowed:
        assert(len(comm_names) == len(normalized_shape))
        self.normalized_shape = normalized_shape
        self.comm_names = comm_names
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        self.gather_mode = 'welford'

        # get local shapes
        normalized_shapes_local = [s // comm.get_size(c) for s,z in zip(self.normalized_shape, self.comm_names)]
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(*normalized_shapes_local))
            self.bias = nn.Parameter(torch.ones(*normalized_shapes_local))

            # set sharing
            comm_names_shared = [c for c,_ in comm.get_names(meta=False).items() if c not in comm_names]
            self.weight.is_shared_mp = comm_names_shared
            self.bias.is_shared_mp = comm_names_shared
            

    def _stats_naive(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the statistics in the naive way by first gathering the tensors and then computing them"""
        for dim, cname in zip(self.normalized_shape, self.comm_names):
            x = gather_from_parallel_region(x, dim, cname)
        var, mean = torch.var_mean(x, dim=normalized_shape, unbiased=False, keepdim=True)

        return var, mean

    def _stats_welford(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the statistics locally, then uses the Welford online algorithm to reduce them"""
        var, mean = torch.var_mean(x, dim=self.normlized_shape, unbiased=False, keepdim=True)
        # workaround to not use shapes, as otherwise cuda graphs won't work
        count = torch.ones_like(x, requires_grad=False)
        count = torch.sum(count, dim=self.normalized_shape, keepdim=True)

        m2 = var * count
        for dim, cname in zip(self.normalized_shape, self.comm_names):
            m2s = torch.split(gather_from_parallel_region(m2, dim, cname), 1, dim)
            means = torch.split(gather_grom_parallel_region(mean, dim, cname), 1, dim)
            counts = torch.split(gather_from_parallel_region(count, dim, cname), 1, dim)

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

        #var = var.reshape(1, -1, 1, 1)
        #mean = mean.reshape(1, -1, 1, 1)

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
            for cname in self.comm_names:
                mean = copy_to_parallel_region(mean, cname)
                var = copy_to_parallel_region(var, cname)

        x = x.to(dtype)
        mean = mean.to(dtype)
        var = var.to(dtype)

        # apply the normalization
        x = (x - mean) / torch.sqrt(var + self.eps)

        # affine transform if we use it
        if self.elementwise_affine:
            # reshape parameters
            padlen = x.dims() - len(self.normalized_shape)
            newshape = [1 for range(padlen)] + [*self.normalized_shape]
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
            norm_layer=DistributedLayerNorm,
    ):

        super(DistributedAttention, self).__init__()

        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.head_dim_local = self.head_dim // comm.get_size(comm_out_name)
        self.scale = self.head_dim ** -0.5

        self.comm_inp_name = comm_inp_name
        self.comm_hidden_name = comm_hidden_name

        self.qkv = DistributedMatmul(dim, dim * 3, comm_inp_name, comm_hidden_name, bias=qkv_bias)
        self.q_norm = norm_layer([self.head_dim], comm_hidden_name) if qk_norm else nn.Identity()
	self.k_norm = norm_layer([self.head_dim], comm_hidden_name) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
	self.proj = DistributedMatmul(dim, dim, comm_hidden_name, comm_inp_name, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        # qkx
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim_local).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = reduce_from_parallel_region(attn, comm_hidden_name)

        # apply softmax, this is a local op
        attn = attn.softmax(dim=-1)

        # this is local too
        attn = self.attn_drop(attn)

        # this is a local op since we contract over tokens
        x = attn @ v

        # transpose back
        x = x.transpose(1, 2).reshape(B, N, C)

        # this is distributed again
        x = self.proj(x)

        # generally we have to be super careful with dropout layers, since
        # those are normalized over the dropouts. That would need to be reduced across nodes
        x = self.proj_drop(x)
        
        return x
        

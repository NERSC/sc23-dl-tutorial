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
#        print("before matmul, shape = {}".format(x.shape))
        x_cp = copy_to_parallel_region(x, self.comm_out_name)
        x_loc = F.linear(x_cp, self.weight, bias=None)
        x_out = reduce_from_parallel_region(x_loc, self.comm_inp_name)
        if hasattr(self, "bias"):
            x_out = x_out + self.bias
#        print("after matmul, shape = {}".format(x_out.shape))
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

        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads

        assert num_heads % comm.get_size(comm_hidden_name) == 0, 'heads are not evenly split across model ranks'
        self.num_heads_local = num_heads // comm.get_size(comm_hidden_name)
        self.head_dim = dim // self.num_heads
        self.scale = (dim // self.num_heads) ** -0.5
        self.fused_attn = True 

        self.comm_inp_name = comm_inp_name
        self.comm_hidden_name = comm_hidden_name
        
        self.qkv = DistributedMatmul(dim, dim * 3, comm_inp_name, comm_hidden_name, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = DistributedMatmul(dim, dim, comm_hidden_name, comm_inp_name, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads_local, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        # transpose back
        x = x.transpose(1, 2).reshape(B, N, self.num_heads_local * self.head_dim)

        # this is distributed again
        x = self.proj(x)

        # generally we have to be super careful with dropout layers, since
        # those are normalized over the dropouts. That would need to be reduced across nodes
        x = self.proj_drop(x)
        
        return x
        

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
        self.weight = nn.Parameter(torch.ones(out_dim_local, inp_dim_local))
        self.weight.is_shared_mp = ["spatial"]
        self.weight.sharded_dims_mp = [
            self.comm_out_name,
            self.comm_inp_name,
            None,
            None,
        ]
        if bias:
            self.bias = nn.Parameter(torch.ones(1, 1, out_dim_local))
            self.bias.is_shared_mp = ["spatial"]
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
        comm_inp_name="mlpi_matmul",
        comm_hidden_name="mlph_matmul",
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


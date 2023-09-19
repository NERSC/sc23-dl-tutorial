
import torch
import torch.distributed as dist
from utils import comm


def get_memory_format(tensor):
    """Helper routine to get the memory format"""
    if tensor.is_contiguous(memory_format=torch.channels_last):
        return torch.channels_last
    else:
        return torch.contiguous_format

def sync_params(model):
    """Helper routine to ensure shared weights are the same after initialization"""
    with torch.no_grad():
        # distributed sync step
        for param in model.parameters():
            if not hasattr(param, "is_shared_mp"):
                param.is_shared_mp = ["model"]

            for comm_group in param.is_shared_mp:
                if comm.get_size(comm_group) > 1:
                    tlist = [
                        torch.empty_like(param)
                        for x in range(comm.get_size(comm_group))
                    ]
                    tlist[comm.get_rank(comm_group)] = param
                    # gather all weights in the comm group
                    dist.all_gather(tlist, param, group=comm.get_group(comm_group))
                    # use weight of rank 0
                    # important to use copy here otherwise the handle gets detaches from the optimizer
                    param.copy_(tlist[0])
 
# distributed primitives
def _reduce(input_, use_fp32=True, group=None):
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if dist.get_world_size(group=group) == 1:
        return input_

    # All-reduce.
    if use_fp32:
        dtype = input_.dtype
        inputf_ = input_.float().contiguous()
        dist.all_reduce(inputf_, group=group)
        input_ = inputf_.to(dtype)
    else:
        input_ = input_.contiguous()
        dist.all_reduce(input_, group=group)

    return input_


def split_tensor_along_dim(tensor, dim, num_chunks):
    """Helper routine to split a tensor along a given dimension"""
    assert (
        dim < tensor.dim()
    ), f"Error, tensor dimension is {tensor.dim()} which cannot be split along {dim}"
    assert (
        tensor.shape[dim] % num_chunks == 0
    ), f"Error, cannot split dim {dim} evenly. Dim size is \
                                                  {tensor.shape[dim]} and requested numnber of splits is {num_chunks}"
    chunk_size = tensor.shape[dim] // num_chunks
    tensor_list = torch.split(tensor, chunk_size, dim=dim)

    return tensor_list

def _split(input_, dim_, group=None):  
    """Split the tensor along dim."""
    # get input format
    input_format = get_memory_format(input_)

    # Bypass the function if we are using only 1 GPU.
    comm_size = dist.get_world_size(group=group)
    if comm_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_dim(input_, dim_, comm_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = dist.get_rank(group=group)
    output = input_list[rank].contiguous(memory_format=input_format)

    return output

def _gather(input_, dim_, group=None):
    """Gather tensors and concatinate along the last dimension."""
    # get input format
    input_format = get_memory_format(input_)

    comm_size = dist.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if comm_size == 1:
        return input_

    # sanity checks
    assert (
        dim_ < input_.dim()
    ), f"Error, cannot gather along {dim_} for tensor with {input_.dim()} dimensions."

    # Size and dimension.
    comm_rank = dist.get_rank(group=group)

    input_ = input_.contiguous(memory_format=input_format)
    tensor_list = [torch.empty_like(input_) for _ in range(comm_size)]
    tensor_list[comm_rank] = input_
    dist.all_gather(tensor_list, input_, group=group)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=dim_).contiguous(memory_format=input_format)

    return output

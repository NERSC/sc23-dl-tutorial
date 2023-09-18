import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from utils import comm

# torch utils
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

# helper functions
from distributed.helpers import _reduce, _split, _gather


class _CopyToParallelRegion(torch.autograd.Function):
    """Pass the input to the parallel region."""

    @staticmethod
    def symbolic(graph, input_, comm_id_):
        """symbolic method"""
        return input_

    @staticmethod
    def forward(ctx, input_, comm_id_): 
        ctx.comm_id = comm_id_
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if comm.is_distributed(ctx.comm_id):
            return _reduce(grad_output, group=comm.get_group(ctx.comm_id)), None
        else:
            return grad_output, None


class _ReduceFromParallelRegion(torch.autograd.Function):
    """All-reduce the input from the parallel region."""

    @staticmethod
    def symbolic(graph, input_, comm_id_):  # pragma: no cover
        """symbolic method"""
        if comm.is_distributed(comm_id_):
            return _reduce(input_, group=comm.get_group(comm_id_))
        else:
            return input_

    @staticmethod
    def forward(ctx, input_, comm_id_):  # pragma: no cover
        if comm.is_distributed(comm_id_):
            return _reduce(input_, group=comm.get_group(comm_id_))
        else:
            return input_

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        return grad_output, None

    
class _GatherFromParallelRegion(torch.autograd.Function):
    """Gather the input from parallel region and concatenate."""

    @staticmethod
    def symbolic(graph, input_, dim_, comm_id_): # pragma: no cover 
        if comm.is_distributed(comm_id_):
            return _gather(input_, dim_, group=comm.get_group(comm_id_))
        else:
            return input_

    @staticmethod
    def forward(ctx, input_, dim_, comm_id_): # pragma: no cover 
        ctx.dim = dim_
        ctx.comm_id = comm_id_
        if comm.is_distributed(comm_id_):
            return _gather(input_, dim_, group=comm.get_group(comm_id_))
        else:
            return input_

    @staticmethod
    def backward(ctx, grad_output): # pragma: no cover 
        if comm.is_distributed(ctx.comm_id):
            return _split(grad_output, ctx.dim, group=comm.get_group(ctx.comm_id)), None, None
        else:
            return grad_output, None, None
    
# matmul parallel
def copy_to_parallel_region(input_, comm_name):  # pragma: no cover
    """Parallel copy helper"""
    return _CopyToParallelRegion.apply(input_, comm_name)


def reduce_from_parallel_region(input_, comm_name):  # pragma: no cover
    """Parallel reduction helper"""
    return _ReduceFromParallelRegion.apply(input_, comm_name)


def gather_from_parallel_region(input_, dim, comm_name):
    """Parallel gather helper"""
    return _GatherFromParallelRegion.apply(input_, dim, comm_name)


def init_ddp_model_and_reduction_hooks(model,
                                       device_ids,
                                       output_device,
                                       bucket_cap_mb = 25,
                                       broadcast_buffers = True,
                                       find_unused_parameters = False,
                                       gradient_as_bucket_view = True,
                                       static_graph = False):
    # early exit if we are not in a distributed setting:
    if not dist.is_initialized():
        return model

    # set this to false in init and then find out if we can use it:
    need_hooks = False
    ddp_group = comm.get_group("data")
    # this is the trivial case
    if comm.get_size("model") == 1:
        # the simple case, we can just continue then
        ddp_group = None
    else:
        # count parameters and reduction groups
        num_parameters_total = 0
        num_parameters_shared_model = 0
        for param in model.parameters():
            # if it does not have any annotation, we assume it is shared between all model ranks
            if not hasattr(param, "is_shared_mp"):
                param.is_shared_mp = ["model"]
            # add the sharing type to the dict
            num_parameters_total += 1
            if "model" in param.is_shared_mp:
                num_parameters_shared_model += 1

        # if all parameters are shared between all model ranks, then the situation is easy
        if (num_parameters_shared_model == num_parameters_total):
            # we can always use DDP
            ddp_group = None
            # register some pre-multiply reduction hooks
            print("Setting up gradient hooks to account for shared parameter multiplicity")
            for param in model.parameters():
                param.register_hook(lambda grad: grad * float(comm.get_size("model")))
        else:
            ddp_group = comm.get_group("data")
            broadcast_buffers = False
            need_hooks = True

    model = DistributedDataParallel(model,
                                    device_ids = device_ids,
                                    output_device = output_device,
                                    bucket_cap_mb = bucket_cap_mb,
                                    broadcast_buffers = broadcast_buffers,
                                    find_unused_parameters = find_unused_parameters,
                                    gradient_as_bucket_view = gradient_as_bucket_view,
                                    static_graph = static_graph,
                                    process_group = ddp_group)
    if not need_hooks:
        return model

    # define comm hook:
    def reduction_comm_hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
        # allreduce everything first:
        buff = bucket.buffer()
        # get future for allreduce
        fut = dist.all_reduce(buff, op=dist.ReduceOp.AVG, group=comm.get_group("data"), async_op=True).get_future()
        # get grads for shared weights
        params = bucket.parameters()
        def grad_reduction(fut, grads, group):
            # reduce remaining gradients
            coalesced = _flatten_dense_tensors(grads)
            dist.all_reduce(coalesced, op=dist.ReduceOp.SUM, group=comm.get_group(group), async_op=False)
            for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)
            return bucket.buffer()

        for group in comm.get_names():
            if group == "data":
                continue

            # build list
            grads = []
            for p in params:
                if group in p.is_shared_mp:
                    if p.grad is not None:
                        grads.append(p.grad.data)
            if not grads:
                continue
            
            # append the new reduction functions
            fut = fut.then(lambda x: grad_reduction(x, grads=grads, group=group))

        return fut
    # register model comm hook
    model.register_comm_hook(state=None, hook=reduction_comm_hook)
    return model

## Model parallelism
Now that we are familiar with distributed data parallel training, we are ready to move to more advanced parallelism in the form of model parallelism. One of the main motivations to explore this dimension is the need to use a larger model and/or process higher resolution images: both these can lead to higher accuracies and/or better emulation of physical phenomena. However, they will inflate the memory consumption (activation and weights) as well as computational cost.  At some point, the model (activation and weights) will no longer fit on a single GPU and we need to partition/shard the model across multiple GPUs. 
We will increase our model size to motivate this partition and show you the building blocks of implementing model parallelism, motivated by the megatron-style model parallelism. We will focus on tensor parallelism here. Our goal is not to build the most efficient model parallel network (which can require significantly more care and development and would parallelize on other dimensions as well) but rather to serve as an instructive blueprint on how to get started on parallelizing your own model in PyTorch. For all the bells and whistles, see [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) for deep details.

### Setting up the communicator groups
We assume a `MxD` grid of GPUs where we use data parallelism (as before) across D GPUs and split the model across `M` GPUs. Take a look at [`utils/comm.py`](utils/comm.py) where this is setup. The logic is more general where we could split the `M` GPUs into more orthogonal groups (example: `M = M_1 x M_2`) for parallelism on more dimensions. This is not done in this tutorial but we have left the logic in, so that the code can be extended for more levels of parallelism. 

A quick example: Let's say we have 8 GPUs in total and we want to do 4-way model parallelism and 2-way data parallelism. The logic would simply have the model parallel group (each has 4 GPUs) ranks as `[0, 1, 2, 3], [4, 5, 6, 7]` and data parallel in the orthogonal dimension (each has 2 GPUs) as: `[0, 4], [1, 5], [2, 6], [3, 7]`. So, let's say, we are looking at the work rank `5` is doing -- then, all model parallel communications will happen within the group `[4, 5, 6, 7]` and data parallel gradient reduction in `[1, 5]`.  For this communication, we tell`torch.distributed` about the groups by creating them with `torch.distributed.new_group(ranks = grp)` and for any communication collectives such as `torch.distributed.all_reduce`, we simply pass the group to the [function call](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce). 

Anothing thing to note is that we need to only use the data parallel groups for the data loading purposes -- this means that the data for each model parallel group (example: `[4, 5, 6, 7]` should be the same). This is taken care of in [`train_mp.py`](train_mp.py) with the lines:
```
params.data_num_shards = comm.get_size("data")
params.data_shard_id = comm.get_rank("data")
```
`get_rank()` and `get_size()` are only within the data parallel group.

### Setting up the model parallel utilities
Now that we have our groups setup, we just have to tell PyTorch to additionally communicate local results within the groups. All tensor parallel distributed utilities are at [`distributed/`](distributed/). Start off with seeing how the distributed matrix multiply is implemented here [`distributed/layers.py`]. Note that there is a call to `reduce_from_parallel_region()` which does an `all_reduce` of the partial sums. Note that you will need to implement both the forward and backward functions for this new operation that will be used to evaluate and compute the gradient seamlessly. We can do this easily in PyTorch by adding our custom `autograd.Function` class in PyTorch.  This is implemented in [`distributed/mappings.py`](distributed/mappings.py). See the [PyTorch docs](https://pytorch.org/docs/stable/notes/extending.html#how-to-use) for the steps to do this. Check out the `copy_to_parallel_region()` function as well and see the forward and backward operations for them and how they align with what we saw in the slides. Note that we have also implemented other routines that are not used (such as gathers and scatters) but will be if you partition/shard on other dimensions. We leave it in so the code can be extended.

### Running the model parallel code
The train script is at [`train_mp.py`](train_mp.py). The model parallel size is defined by `row_parallel_size`. Setting this to `4`, for example, will enable 4-way model parallelism. Let's run a larger model by increasing our `embed_dim` to `1024` .  The config for this is called `mp` which trains the larger model assuming a global batch size of `64` with 4 GPUs for data parallelism (hence local batch size is `16`) . Let's use no model parallelism: so set `row_parallel_size=1` and run it on 4 GPUs with the following command:
```
sbatch --nodes 1 submit_pm_mp.sh --config=mp --row_parallel_size=1
```
 We can see from the logs that the job crashes with an OOM signal because the model is too big. 
```
File "/usr/local/lib/python3.10/dist-packages/torch/_tensor.py", line 491, in backward
torch.autograd.backward(
File "/usr/local/lib/python3.10/dist-packages/torch/autograd/__init__.py", line 204, in backward
Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 318.00 MiB. GPU 1 has a total capacty of 39.39 GiB of which 130.56 MiB is free. Including non-PyTorch memory, this process has 39.25 GiB memory in use. Of the allocated memory 31.34 GiB is allocated by PyTorch, and 1.40 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```
If we run it on an 80G GPU, we can see the estimated memory usage to be around 41GB and hence just overflows the 40G GPU. While this example is instructive, larger models (and/or larger inputs) can push the memory consumption significantly higher. 
```
Scaffolding memory high watermark: 10.2071533203125 GB.
2023-11-03 04:14:58,115 - root - INFO - Starting Training Loop...
2023-11-03 04:15:08,450 - torch.nn.parallel.distributed - INFO - Reducer buckets have been rebuilt in this iteration.
2023-11-03 04:15:08,450 - torch.nn.parallel.distributed - INFO - Reducer buckets have been rebuilt in this iteration.
2023-11-03 04:15:08,451 - torch.nn.parallel.distributed - INFO - Reducer buckets have been rebuilt in this iteration.
2023-11-03 04:15:08,452 - torch.nn.parallel.distributed - INFO - Reducer buckets have been rebuilt in this iteration.
2023-11-03 04:15:08,580 - root - INFO -  Memory usage after forward pass: 40.0841064453125 GB.
2023-11-03 04:28:15,944 - root - INFO - Time taken for epoch 1 is 791.9564564228058 sec, avg 47.8410141021346 samples/sec
2023-11-03 04:28:15,945 - root - INFO - Avg train loss=0.336905
2023-11-03 04:28:35,199 - root - INFO - Avg val loss=0.265976
2023-11-03 04:28:35,199 - root - INFO - Total validation time: 18.541259050369263 sec
2023-11-03 04:28:36,641 - root - INFO -  Memory usage after forward pass: 41.20123291015625 GB.
2023-11-03 04:41:45,640 - root - INFO - Time taken for epoch 2 is 790.3432559967041 sec, avg 48.0196417341964 samples/sec
2023-11-03 04:41:45,642 - root - INFO - Avg train loss=0.224150
2023-11-03 04:42:03,306 - root - INFO - Avg val loss=0.197870
2023-11-03 04:42:03,307 - root - INFO - Total validation time: 17.101737022399902 sec
2023-11-03 04:42:04,724 - root - INFO -  Memory usage after forward pass: 41.20123291015625 GB.
2023-11-03 04:55:13,705 - root - INFO - Time taken for epoch 3 is 790.3928642272949 sec, avg 48.01662782862127 samples/sec
2023-11-03 04:55:13,706 - root - INFO - Avg train loss=0.179888
2023-11-03 04:55:29,698 - root - INFO - Avg val loss=0.169655
2023-11-03 04:55:29,698 - root - INFO - Total validation time: 15.423459529876709 sec
2023-11-03 04:55:31,077 - root - INFO -  Memory usage after forward pass: 41.20123291015625 GB.
```

Let's run it with `row_parallel_size=4`, which will partition/shard the hidden dimensions of the MLP weights and biases as well as the attention heads. Note here that 4 GPUs are used for model parallelism. Recall our global batch size is `64`. How many GPUs do we need? We also want 4-way data parallel, in addition to model parallelism, here: therefore, we should run on 16 GPUs (or 4 nodes on Perlmutter). Remember that we are assuming `M x D` GPUs always. Run this config with the command:
```
sbatch --nodes 4 submit_pm_mp.sh --config=mp --row_parallel_size=4
```
```
Scaffolding memory high watermark: 8.9056396484375 GB.
2023-11-04 04:55:27,609 - root - INFO - Starting Training Loop...
2023-11-04 04:55:38,372 - root - INFO -  Memory usage after forward pass: 28.6165771484375 GB.
2023-11-04 05:01:08,265 - root - INFO - Time taken for epoch 1 is 334.90422677993774 sec, avg 113.13085046518637 samples/sec
2023-11-04 05:01:08,266 - root - INFO - Avg train loss=0.332298
2023-11-04 05:01:21,165 - root - INFO - Avg val loss=0.246270
2023-11-04 05:01:21,166 - root - INFO - Total validation time: 12.243930101394653 sec
2023-11-04 05:01:21,829 - root - INFO -  Memory usage after forward pass: 32.7415771484375 GB.
2023-11-04 05:06:52,733 - root - INFO - Time taken for epoch 2 is 331.56011605262756 sec, avg 114.46491348789367 samples/sec
2023-11-04 05:06:52,734 - root - INFO - Avg train loss=0.205112
2023-11-04 05:07:03,417 - root - INFO - Avg val loss=0.178545
2023-11-04 05:07:03,417 - root - INFO - Total validation time: 10.102246046066284 sec
2023-11-04 05:07:04,120 - root - INFO -  Memory usage after forward pass: 32.7415771484375 GB.
2023-11-04 05:12:35,057 - root - INFO - Time taken for epoch 3 is 331.6335074901581 sec, avg 114.43958207729146 samples/sec
2023-11-04 05:12:35,058 - root - INFO - Avg train loss=0.160458
2023-11-04 05:12:46,143 - root - INFO - Avg val loss=0.148919
2023-11-04 05:12:46,144 - root - INFO - Total validation time: 10.509092807769775 sec
2023-11-04 05:12:46,776 - root - INFO -  Memory usage after forward pass: 32.7415771484375 GB.
```

We see that the memory has reduced to 32.7G. Also note that the throughput is higher.

*Question: Can we drop the memory consumed more? What tensors have we left un-partitioned?*

We also see that the bigger model gets a better RMSE compared to the batch size `64` run from before (with the smaller model):
![model parallel logs](tutorial_images/mp_comp.png)

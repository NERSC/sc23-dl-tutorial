import logging
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch import Tensor

#concurrent futures
import concurrent.futures as cf

# distributed stuff
import torch.distributed as dist
from utils import comm

#dali stuff
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

# es helper
import utils.dali_es_helper as esh

def get_data_loader(params, files_pattern, distributed, train):
    dataloader = DaliDataLoader(params, files_pattern, train)

    if train:
        return dataloader, None, None
    else:
        return dataloader, None

class DaliDataLoader(object):
    def get_pipeline(self):
        pipeline = Pipeline(batch_size = self.batch_size,
                            num_threads = 2,
                            device_id = self.device_index,
                            py_num_workers = self.num_data_workers,
                            py_start_method='spawn',
                            seed = self.model_seed)
        
     
        with pipeline: # get input and target 
            # get input and target
            inp, tar = fn.external_source(source = esh.HDF5ES(self.location,
                                                              self.train,
                                                              self.batch_size,
                                                              self.dt,
                                                              self.img_size,
                                                              self.n_in_channels,
                                                              self.n_out_channels,
                                                              self.num_shards,
                                                              self.shard_id,
                                                              self.limit_nsamples,
                                                              enable_logging = False,
                                                              seed=self.global_seed),
                                          num_outputs = 2,
                                          layout = ["CHW", "CHW"],
                                          batch = False,
                                          no_copy = True,
                                          parallel = True)
            
            # upload to GPU
            inp = inp.gpu()
            tar = tar.gpu()

            if self.normalize:
                inp = fn.normalize(inp,
                                   device = "gpu",
                                   axis_names = "HW",
                                   batch = False,
                                   mean = self.in_bias,
                                   stddev = self.in_scale)

                tar = fn.normalize(tar,
                                   device = "gpu",
                                   axis_names = "HW",
                                   batch = False,
                                   mean = self.out_bias,
                                   stddev = self.out_scale)

            pipeline.set_outputs(inp, tar)
        return pipeline

    def __init__(self, params, location, train, seed = 333):
        # set up seeds
        # this one is the same on all ranks
        self.global_seed = seed
        # this one is the same for all ranks of the same model
        model_id = comm.get_world_rank() // comm.get_size("model")
        self.model_seed = self.global_seed + model_id
        # this seed is supposed to be diffferent for every rank
        self.local_seed = self.global_seed + comm.get_world_rank()

        self.num_data_workers = params.num_data_workers
        self.device_index = torch.cuda.current_device()
        self.batch_size = int(params.local_batch_size)

        self.location = location
        self.train = train
        self.dt = params.dt
        self.n_in_channels = params.n_in_channels
        self.n_out_channels = params.n_out_channels
        self.img_size = params.img_size
        self.limit_nsamples = params.limit_nsamples if train else params.limit_nsamples_val

        # load stats
        self.normalize = True
        means = np.load(params.global_means_path)[0][:self.n_in_channels]
        stds = np.load(params.global_stds_path)[0][:self.n_in_channels]
        self.in_bias = means
        self.in_scale = stds
        means = np.load(params.global_means_path)[0][:self.n_out_channels]
        stds = np.load(params.global_stds_path)[0][:self.n_out_channels]
        self.out_bias = means
        self.out_scale = stds

        # set sharding
        if dist.is_initialized():
            self.num_shards = params.data_num_shards
            self.shard_id = params.data_shard_id
        else:
            self.num_shards = 1
            self.shard_id = 0

        # get img source data
        extsource = esh.HDF5ES(self.location,
                               self.train,
                               self.batch_size,
                               self.dt,
                               self.img_size,
                               self.n_in_channels,
                               self.n_out_channels,
                               self.num_shards,
                               self.shard_id,
                               self.limit_nsamples,
                               seed=self.global_seed)
        self.num_batches = extsource.num_steps_per_epoch
        del extsource
 
        # create pipeline
        self.pipeline = self.get_pipeline()
        self.pipeline.start_py_workers()
        self.pipeline.build()

        # create iterator
        self.iterator = DALIGenericIterator([self.pipeline], ['inp', 'tar'],
                                            auto_reset = True,
                                            last_batch_policy = LastBatchPolicy.DROP,
                                            prepare_first_batch = True)
        
    def __len__(self):
        return self.num_batches

    def __iter__(self):
        #self.iterator.reset()
        for token in self.iterator:
            inp = token[0]['inp']
            tar = token[0]['tar']
            
            yield inp, tar

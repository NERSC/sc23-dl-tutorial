import os
import glob
import numpy as np
import cupyx as cpx
import h5py
import logging

class HDF5ES(object):
    # very important: the seed has to be constant across the workers, or otherwise mayhem:
    def __init__(self, location, 
                train, batch_size, 
                dt, img_size,
                n_in_channels, n_out_channels, 
                num_shards,
                shard_id,
                enable_logging = True,
                seed=333):
        self.batch_size = batch_size
        self.location = location
        self.img_size = img_size
        self.train = train
        self.dt = dt
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.rng = np.random.default_rng(seed = seed)
        self.num_shards = num_shards
        self.shard_id = shard_id
        
        self._get_files_stats(enable_logging)
        self.shuffle = True if train else False
        
    def _get_files_stats(self, enable_logging):
        self.files_paths = glob.glob(self.location + "/*.h5")
        self.files_paths.sort()
        self.years = [int(os.path.splitext(os.path.basename(x))[0][-4:]) for x in self.files_paths]
        self.n_years = len(self.files_paths)

        with h5py.File(self.files_paths[0], 'r') as _f:
            logging.info("Getting file stats from {}".format(self.files_paths[0]))
            self.n_samples_per_year = _f['fields'].shape[0]
            self.img_shape_x = self.img_size[0]
            self.img_shape_y = self.img_size[1]
            assert(self.img_shape_x <= _f['fields'].shape[2] and self.img_shape_y <= _f['fields'].shape[3]), 'image shapes are greater than dataset image shapes'

        self.n_samples_total = self.n_years * self.n_samples_per_year
        self.n_samples_shard = self.n_samples_total // self.num_shards
        self.files = [None for _ in range(self.n_years)]
        self.dsets = [None for _ in range(self.n_years)]
        if enable_logging:
            logging.info("Number of samples per year: {}".format(self.n_samples_per_year))
            logging.info("Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(self.location, self.n_samples_total, self.img_shape_x, self.img_shape_y, self.n_in_channels))
            if self.num_shards > 1:
                logging.info("Using shards of size {} per rank".format(self.n_samples_shard))

        # number of steps per epoch
        self.num_steps_per_epoch = self.n_samples_shard // self.batch_size
        self.last_epoch = None

        self.index_permutation = None
        # prepare buffers for double buffering
        self.current_buffer = 0
        self.inp_buffs = [cpx.zeros_pinned((self.n_in_channels, self.img_shape_x, self.img_shape_y), dtype=np.float32),
                        cpx.zeros_pinned((self.n_in_channels, self.img_shape_x, self.img_shape_y), dtype=np.float32)]
        self.tar_buffs = [cpx.zeros_pinned((self.n_out_channels, self.img_shape_x, self.img_shape_y), dtype=np.float32),
                        cpx.zeros_pinned((self.n_out_channels, self.img_shape_x, self.img_shape_y), dtype=np.float32)]    
    
    def __len__(self):
        return self.num_steps_per_epoch

    def __del__(self):
        for f in self.files:
            if f is not None:
                f.close()

    def __call__(self, sample_info):
        # check if epoch is done
        if sample_info.iteration >= self.num_steps_per_epoch:
            raise StopIteration

        # check if we need to shuffle again
        if sample_info.epoch_idx != self.last_epoch:
            self.last_epoch = sample_info.epoch_idx
            self.index_permutation = self.rng.permutation(self.n_samples_total)
            # shard the data
            start = self.n_samples_shard * self.shard_id
            end = start + self.n_samples_shard
            self.index_permutation = self.index_permutation[start:end]

        # determine local and sample idx
        sample_idx = self.index_permutation[sample_info.idx_in_epoch]
        year_idx = int(sample_idx / self.n_samples_per_year) #which year we are on
        local_idx = int(sample_idx % self.n_samples_per_year) #which sample in that year we are on - determines indices for centering 

        step = self.dt # time step
        
        # boundary conditions to ensure we don't pull data that is not in a specific year
        local_idx = local_idx % (self.n_samples_per_year - step)
        if local_idx < step:
            local_idx += step

        if self.files[year_idx] is None:
            self.files[year_idx] = h5py.File(self.files_paths[year_idx], 'r')
            self.dsets[year_idx] = self.files[year_idx]['fields']
        
        tmp_inp = self.dsets[year_idx][local_idx, ...]
        tmp_tar = self.dsets[year_idx][local_idx+step, ...]

        # handles to buffers buffers
        inp = self.inp_buffs[self.current_buffer]
        tar = self.tar_buffs[self.current_buffer]
        self.current_buffer = (self.current_buffer + 1) % 2
        
        # crop the pixels:
        inp[...] = tmp_inp[..., :self.img_shape_x, :self.img_shape_y]
        tar[...] = tmp_tar[..., :self.img_shape_x, :self.img_shape_y]


        return inp, tar

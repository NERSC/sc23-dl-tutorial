import os
import logging
import glob
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch import Tensor
import h5py

def worker_init(wrk_id):
    np.random.seed(torch.utils.data.get_worker_info().seed%(2**32 - 1))

def get_data_loader(params, files_pattern, distributed, train):
    dataset = GetDataset(params, files_pattern, train)
    sampler = DistributedSampler(dataset, shuffle=train) if distributed else None
    
    dataloader = DataLoader(dataset,
                            batch_size=int(params.local_batch_size),
                            num_workers=params.num_data_workers,
                            shuffle=(sampler is None),
                            sampler=sampler if train else None,
                            worker_init_fn=worker_init,
                            drop_last=True,
#                            persistent_workers=train,
                            pin_memory=torch.cuda.is_available())

    if train:
        return dataloader, dataset, sampler
    else:
        return dataloader, dataset

class GetDataset(Dataset):
    def __init__(self, params, location, train):
        self.params = params
        self.location = location
        self.train = train
        self.dt = params.dt
        self.n_in_channels = params.n_in_channels
        self.n_out_channels = params.n_out_channels
        self.normalize = True
        self.means = np.load(params.global_means_path)[0]
        self.stds = np.load(params.global_stds_path)[0]
        self.limit_nsamples = params.limit_nsamples if train else params.limit_nsamples_val
        self._get_files_stats()

    def _get_files_stats(self):
        self.files_paths = glob.glob(self.location + "/*.h5")
        self.files_paths.sort()
        self.years = [int(os.path.splitext(os.path.basename(x))[0][-4:]) for x in self.files_paths]
        self.n_years = len(self.files_paths)

        with h5py.File(self.files_paths[0], 'r') as _f:
            logging.info("Getting file stats from {}".format(self.files_paths[0]))
            self.n_samples_per_year = _f['fields'].shape[0]
            self.img_shape_x = self.params.img_size[0]
            self.img_shape_y = self.params.img_size[1]
            assert(self.img_shape_x <= _f['fields'].shape[2] and self.img_shape_y <= _f['fields'].shape[3]), 'image shapes are greater than dataset image shapes'

        self.n_samples_total = self.n_years * self.n_samples_per_year
        if self.limit_nsamples is not None:
            self.n_samples_total = min(self.n_samples_total, self.limit_nsamples)
            logging.info("Overriding total number of samples to: {}".format(self.n_samples_total))
        self.files = [None for _ in range(self.n_years)]
        logging.info("Number of samples per year: {}".format(self.n_samples_per_year))
        logging.info("Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(self.location, self.n_samples_total, self.img_shape_x, self.img_shape_y, self.n_in_channels))

    def _open_file(self, year_idx):
        _file = h5py.File(self.files_paths[year_idx], 'r')
        self.files[year_idx] = _file['fields']  
    
    def __len__(self):
        return self.n_samples_total

    def _normalize(self, img):
        if self.normalize:
            img -= self.means
            img /= self.stds
        return torch.as_tensor(img)

    def __getitem__(self, global_idx):
        year_idx = int(global_idx / self.n_samples_per_year)  # which year
        local_idx = int(global_idx % self.n_samples_per_year) # which sample in that year 

        # open image file
        if self.files[year_idx] is None:
            self._open_file(year_idx)
        step = self.dt # time step

        # boundary conditions to ensure we don't pull data that is not in a specific year
        local_idx = local_idx % (self.n_samples_per_year - step)
        if local_idx < step:
            local_idx += step
        
        # pre-process and get the image fields
        inp_field = self.files[year_idx][local_idx,:,0:self.img_shape_x,0:self.img_shape_y]
        tar_field = self.files[year_idx][local_idx+step,:,0:self.img_shape_x,0:self.img_shape_y]
        inp, tar = self._normalize(inp_field), self._normalize(tar_field)

        return inp, tar

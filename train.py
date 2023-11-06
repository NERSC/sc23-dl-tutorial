import sys
import os
import time
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

import logging
from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
from utils import get_data_loader_distributed
from utils.loss import l2_loss, l2_loss_opt
from utils.metrics import weighted_rmse
from utils.plots import generate_images
from networks import vit

import apex.optimizers as aoptim

def compute_grad_norm(p_list, device):
    norm_type = 2.0
    grads = [p.grad for p in p_list if p.grad is not None]
    total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
    return total_norm

def compute_parameter_norm(p_list, device):
    norm_type = 2.0
    total_norm = torch.norm(torch.stack([torch.norm(p.detach(), norm_type).to(device) for p in p_list]), norm_type)
    return total_norm

def train(params, args, local_rank, world_rank, world_size):
    # set device and benchmark mode
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda:%d'%local_rank)

    # get data loader
    logging.info('rank %d, begin data loader init'%world_rank)
    train_data_loader, train_dataset, train_sampler = get_data_loader_distributed(params, params.train_data_path, params.distributed, train=True)
    val_data_loader, valid_dataset = get_data_loader_distributed(params, params.valid_data_path, params.distributed, train=False)
    logging.info('rank %d, data loader initialized'%(world_rank))

    # create model
    model = vit.ViT(params).to(device)
    
    if params.amp_dtype == torch.float16: 
        scaler = GradScaler()
    if params.distributed and not args.noddp:
        if args.disable_broadcast_buffers: 
            model = DistributedDataParallel(model, device_ids=[local_rank],
                                            bucket_cap_mb=args.bucket_cap_mb,
                                            broadcast_buffers=False,
                                            gradient_as_bucket_view=True)
        else:
            model = DistributedDataParallel(model, device_ids=[local_rank],
                                            bucket_cap_mb=args.bucket_cap_mb)

    if params.enable_apex:
        optimizer = optim.FusedAdam(model.parameters(), lr = params.lr,
                                    adam_w_mode=False, set_grad_none=True)
    else:
        optimizer = optim.Adam(model.parameters(), lr = params.lr)

    iters = 0
    startEpoch = 0

    if params.lr_schedule == 'cosine':
        if params.warmup > 0:
            lr_scale = lambda x: min(params.lr*((x+1)/params.warmup), 0.5*params.lr*(1 + np.cos(np.pi*x/params.num_epochs)))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.num_epochs, last_epoch=startEpoch-1)
    else:
        scheduler = None

    if params.enable_jit:
        model_handle = model.module if (params.distributed and not args.noddp) else model
        model_handle = torch.jit.script(model_handle)  

    # select loss function
    if params.enable_jit:
        loss_func = l2_loss_opt
    else:
        loss_func = l2_loss

    if world_rank==0: 
        logging.info("Starting Training Loop...")

    # Log initial loss on train and validation to tensorboard
    with torch.no_grad():
        inp, tar = map(lambda x: x.to(device), next(iter(train_data_loader)))
        gen = model(inp)
        tr_loss = loss_func(gen, tar)
        inp, tar = map(lambda x: x.to(device), next(iter(val_data_loader)))
        gen = model(inp)
        val_loss = loss_func(gen, tar)
        val_rmse = weighted_rmse(gen, tar)
        if params.distributed:
            torch.distributed.all_reduce(tr_loss)
            torch.distributed.all_reduce(val_loss)
            torch.distributed.all_reduce(val_rmse)
        if world_rank==0:
            args.tboard_writer.add_scalar('Loss/train', tr_loss.item()/world_size, 0)
            args.tboard_writer.add_scalar('Loss/valid', val_loss.item()/world_size, 0)
            args.tboard_writer.add_scalar('RMSE(u10m)/valid', val_rmse.cpu().numpy()[0]/world_size, 0)

    iters = 0
    t1 = time.time()
    for epoch in range(startEpoch, startEpoch + params.num_epochs):
        torch.cuda.synchronize() # device sync to ensure accurate epoch timings
        if params.distributed and (train_sampler is not None):
            train_sampler.set_epoch(epoch)
        start = time.time()
        tr_loss = []
        tr_time = 0.
        dat_time = 0.
        log_time = 0.

        model.train()
        step_count = 0
        for i, data in enumerate(train_data_loader, 0):
            if (args.enable_manual_profiling and world_rank==0):
                if (epoch == 3 and i == 0):
                    torch.cuda.profiler.start()
                if (epoch == 3 and i == 59):
                    torch.cuda.profiler.stop()

            if args.enable_manual_profiling: torch.cuda.nvtx.range_push(f"step {i}")
            iters += 1
            dat_start = time.time()
            if args.enable_manual_profiling: torch.cuda.nvtx.range_push(f"data copy in {i}")

            inp, tar = map(lambda x: x.to(device), data)
            if args.enable_manual_profiling: torch.cuda.nvtx.range_pop() # copy in

            tr_start = time.time()
            b_size = inp.size(0)
            
            optimizer.zero_grad()

            if args.enable_manual_profiling: torch.cuda.nvtx.range_push(f"forward")
            with autocast(enabled=params.amp_enabled, dtype=params.amp_dtype):
                gen = model(inp)
                loss = loss_func(gen, tar)
            if args.enable_manual_profiling: torch.cuda.nvtx.range_pop() #forward

            if params.amp_dtype == torch.float16: 
                scaler.scale(loss).backward()
                if args.enable_manual_profiling: torch.cuda.nvtx.range_push(f"optimizer")
                scaler.step(optimizer)
                if args.enable_manual_profiling: torch.cuda.nvtx.range_pop() # optimizer
                scaler.update()
            else:
                loss.backward()
                if args.enable_manual_profiling: torch.cuda.nvtx.range_push(f"optimizer")
                optimizer.step()
                if args.enable_manual_profiling: torch.cuda.nvtx.range_pop() # optimizer

            if params.distributed:
                torch.distributed.all_reduce(loss)
            tr_loss.append(loss.item()/world_size)

            if args.enable_manual_profiling: torch.cuda.nvtx.range_pop() # step

#            g_norm = compute_grad_norm(model.parameters(), device)
#            p_norm = compute_parameter_norm(model.parameters(), device)

            tr_end = time.time()
            tr_time += tr_end - tr_start
            dat_time += tr_start - dat_start
            step_count += 1

        # lr step
        scheduler.step()
        torch.cuda.synchronize() # device sync to ensure accurate epoch timings
        end = time.time()

        if world_rank==0:
            logging.info('Time taken for epoch {} is {} sec, avg {} samples/sec'.format(epoch + 1, end - start,
                                                                                        (step_count * params["global_batch_size"]) / (end - start)))
            logging.info('  Avg train loss=%f'%np.mean(tr_loss))
            args.tboard_writer.add_scalar('Loss/train', np.mean(tr_loss), iters)
            args.tboard_writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], iters)
            args.tboard_writer.add_scalar('Avg iters per sec', step_count/(end-start), iters)
            fig = generate_images([inp, tar, gen])
            args.tboard_writer.add_figure('Visualization, t2m', fig, iters, close=True)

        val_start = time.time()
        val_loss = torch.zeros(1, device=device)
        val_rmse = torch.zeros((params.n_out_channels), dtype=torch.float32, device=device)
        val_samples = torch.zeros(1, device=device)
        model.eval()

        with torch.inference_mode():
            with torch.no_grad():
                for i, data in enumerate(val_data_loader, 0):
                    with autocast(enabled=params.amp_enabled, dtype=params.amp_dtype):
                        inp, tar = map(lambda x: x.to(device), data)
                        gen = model(inp)
                    val_loss += loss_func(gen, tar)
                    val_rmse += weighted_rmse(gen, tar)
                    val_samples += float(inp.shape[0])
                    valid_steps += 1
                
                if params.distributed:
                    torch.distributed.all_reduce(val_loss)
                    torch.distributed.all_reduce(val_rmse)
                    torch.distributed.all_reduce(val_samples)
                    
                val_rmse_avg = (val_rmse / val_samples)
                val_loss_avg = (val_loss / val_samples)

                val_end = time.time()
                if world_rank==0:
                    logging.info(f'  Avg val loss={val_loss_avg}')
                    logging.info(f'  Total validation time: {val_end - val_start} sec')
                    args.tboard_writer.add_scalar('Loss/valid', val_loss_avg, iters)
                    args.tboard_writer.add_scalar('RMSE(u10m)/valid', val_rmse_avg[0], iters)
                    args.tboard_writer.flush()

    t2 = time.time()
    tottime = t2 - t1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='00', type=str, help='tag for indexing the current experiment')
    parser.add_argument("--yaml_config", default='./config/ViT.yaml', type=str, help='path to yaml file containing training configs')
    parser.add_argument("--config", default='base', type=str, help='name of desired config in yaml file')
    parser.add_argument("--amp_mode", default='none', type=str, choices=['none', 'fp16', 'bf16'], help='select automatic mixed precision mode')  
    parser.add_argument("--enable_apex", action='store_true', help='enable apex fused Adam optimizer')
    parser.add_argument("--enable_jit", action='store_true', help='enable JIT compilation')
    parser.add_argument("--enable_manual_profiling", action='store_true', help='enable manual nvtx ranges and profiler start/stop calls')
    parser.add_argument("--local_batch_size", default=None, type=int, help='local batchsize (manually override global_batch_size config setting)')
    parser.add_argument("--num_epochs", default=None, type=int, help='number of epochs to run')
    parser.add_argument("--num_data_workers", default=None, type=int, help='number of data workers for data loader')
    parser.add_argument("--bucket_cap_mb", default=25, type=int, help='max message bucket size in mb')
    parser.add_argument("--disable_broadcast_buffers", action='store_true', help='disable syncing broadcasting buffers')
    parser.add_argument("--noddp", action='store_true', help='disable DDP communication')
    args = parser.parse_args()
 
    run_num = args.run_num

    params = YParams(os.path.abspath(args.yaml_config), args.config)

    # Update config with modified args
    # set up amp
    if args.amp_mode != 'none':
        params.update({"amp_mode": args.amp_mode})
    amp_dtype = torch.float32
    if params.amp_mode == "fp16":
        amp_dtype = torch.float16
    elif params.amp_mode == "bf16":
        amp_dtype = torch.bfloat16    
    params.update({"amp_enabled": amp_dtype is not torch.float32,
                    "amp_dtype" : amp_dtype, 
                    "enable_apex" : args.enable_apex,
                    "enable_jit" : args.enable_jit
                    })
    
    if args.num_epochs:
        params.update({"num_epochs" : args.num_epochs})

    if args.num_data_workers:
        params.update({"num_data_workers" : args.num_data_workers})

    params.distributed = False
    if 'WORLD_SIZE' in os.environ:
        params.distributed = int(os.environ['WORLD_SIZE']) > 1
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        world_size = 1

    world_rank = 0
    local_rank = 0
    if params.distributed:
        torch.distributed.init_process_group(backend='nccl',
                                            init_method='env://')
        world_rank = torch.distributed.get_rank()
        local_rank = int(os.environ['LOCAL_RANK'])

    if args.local_batch_size:
        # Manually override batch size
        params.local_batch_size = args.local_batch_size
        params.update({"global_batch_size" : world_size*args.local_batch_size})
    else:
        # Compute local batch size based on number of ranks
        params.local_batch_size = params.global_batch_size//world_size

    # Set up directory
    baseDir = params.expdir
    expDir = os.path.join(baseDir, args.config + '/%dGPU/'%(world_size) + str(run_num) + '/')
    if world_rank==0:
        if not os.path.isdir(expDir):
            os.makedirs(expDir)
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'out.log'))
        params.log()
        args.tboard_writer = SummaryWriter(log_dir=os.path.join(expDir, 'logs/'))

    params.experiment_dir = os.path.abspath(expDir)

    train(params, args, local_rank, world_rank, world_size)

    if params.distributed:
        torch.distributed.barrier()
    logging.info('DONE ---- rank %d'%world_rank)


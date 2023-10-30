#!/bin/bash
export MASTER_ADDR=$(hostname)

image=nersc/pytorch:ngc-23.04-v0

DATADIR=/pscratch/sd/s/shas1693/data/sc23_tutorial_data/downsampled
LOGDIR=${SCRATCH}/sc23-dl-tutorial/logs
mkdir -p ${LOGDIR}

ngpu=4
config_file=./config/ViT.yaml
config="mp"
run_num="test"
col_parallel_size=1
row_parallel_size=4
#cmd="python train.py --amp_mode=$amp_mode --yaml_config=$config_file --config=$config --run_num=$run_num --local_batch_size=$local_batch_size"
cmd="python train_mp.py --row_parallel_size=$row_parallel_size --col_parallel_size=$col_parallel_size --yaml_config=$config_file --config=$config --run_num=$run_num"


srun -n $ngpu --cpus-per-task=32 --gpus-per-node $ngpu shifter --image=${image} --module=gpu,nccl-2.18 -V ${DATADIR}:/data -V ${LOGDIR}:/logs  bash -c "source export_DDP_vars.sh && $cmd"

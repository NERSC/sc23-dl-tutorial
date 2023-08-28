#!/bin/bash
export MASTER_ADDR=$(hostname)

image=nersc/pytorch:ngc-23.04-v0
env=/global/homes/s/shas1693/.local/perlmutter/nersc_pytorch_ngc-23.04-v0

DATADIR=/pscratch/sd/s/shas1693/data/sc23_tutorial_data/
LOGDIR=${SCRATCH}/sc23-dl-tutorial/logs
mkdir -p ${LOGDIR}

ngpu=4
config_file=./config/ViT.yaml
config="short"
run_num="interactive_b64_dataworkers4"
amp_mode="fp16"
cmd="python train.py --amp_mode=$amp_mode --yaml_config=$config_file --config=$config --run_num=$run_num"


srun -n $ngpu --cpus-per-task=32 --gpus-per-node $ngpu shifter --image=${image} --env PYTHONUSERBASE=${env} --module=gpu,nccl-2.15 -V ${DATADIR}:/data -V ${LOGDIR}:/logs  bash -c "source nccl-2.17.sh && source export_DDP_vars.sh && $cmd"

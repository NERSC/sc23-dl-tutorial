#!/bin/bash 
#SBATCH -C gpu 
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH -A nvendor
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH --time=00:15:00
#SBATCH --image=nersc/pytorch:ngc-23.07-v0
#SBATCH --module=gpu,nccl-2.18
#SBATCH -J vit-era5
#SBATCH -o %x-%j.out

DATADIR=/pscratch/sd/s/shas1693/data/sc23_tutorial_data/downsampled
LOGDIR=${SCRATCH}/sc23-dl-tutorial/logs
env=~/.local/perlmutter/nersc_pytorch_ngc-23.07-v0
mkdir -p ${LOGDIR}

config_file=./config/ViT.yaml
config="short_opt" #"short_opt"
run_num="test"
amp_mode="fp16"
col_parallel_size=2
row_parallel_size=2
args="--col_parallel_size=$col_parallel_size --row_parallel_size=$row_parallel_size --amp_mode=$amp_mode --yaml_config=$config_file --config=$config --run_num=$run_num"

#args="${@}"

export FI_MR_CACHE_MONITOR=userfaultfd
export NCCL_NET_GDR_LEVEL=PHB
export HDF5_USE_FILE_LOCKING=FALSE

# Profiling
if [ "${ENABLE_PROFILING:-0}" -eq 1 ]; then
    echo "Enabling profiling..."
    NSYS_ARGS="--trace=cuda,cublas,nvtx --kill none -c cudaProfilerApi -f true"
    NSYS_OUTPUT=${PROFILE_OUTPUT:-"profile"}
    export PROFILE_CMD="nsys profile $NSYS_ARGS -o $NSYS_OUTPUT"
fi

export MASTER_ADDR=$(hostname)

# Reversing order of GPUs to match default CPU affinities from Slurm
export CUDA_VISIBLE_DEVICES=3,2,1,0

set -x
srun -u shifter --env PYTHONUSERBASE=${env} -V ${DATADIR}:/data -V ${LOGDIR}:/logs \
    bash -c "
    pip install timm pynvml;
    source export_DDP_vars.sh
    ${PROFILE_CMD} python train_mp.py ${args}
    "

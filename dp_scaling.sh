#!/bin/bash

#sbatch -n 1 -t 12:00:00 submit_pm.sh --config=bs32_opt_nh2 --amp_mode=fp16
#sbatch -n 1 -t 24:00:00 -C gpu\&hbm80g submit_pm.sh --config=bs32_opt_e1024 --amp_mode=fp16

sbatch -n 1 -t 24:00:00 submit_pm.sh --config=bs16_opt_b95_40k --amp_mode=fp16
#sbatch -n 1 -t 08:00:00 submit_pm.sh --config=bs32_opt_b95 --amp_mode=fp16
#sbatch -n 2 -t 08:00:00 submit_pm.sh --config=bs64_opt_b95 --amp_mode=fp16
#sbatch -n 4 -t 08:00:00 submit_pm.sh --config=bs128_opt_b95 --amp_mode=fp16
#sbatch -n 8 -t 08:00:00 submit_pm.sh --config=bs256_opt_b95 --amp_mode=fp16
#sbatch -n 16 -t 08:00:00 submit_pm.sh --config=bs512_opt_b95 --amp_mode=fp16
#sbatch -n 32 -t 08:00:00 submit_pm.sh --config=bs1024_opt_b95 --amp_mode=fp16
#sbatch -n 64 -t 08:00:00 submit_pm.sh --config=bs2048_opt_b95 --amp_mode=fp16

#for opt in lamb b95
#do 
#
#   sbatch -n 8 -t 08:00:00 submit_pm.sh --config=bs256_opt_${opt} --amp_mode=fp16 --enable_apex
#   sbatch -n 8 -t 08:00:00 submit_pm.sh --config=bs256_opt_w1_ni15_${opt} --amp_mode=fp16 --enable_apex
#
#   sbatch -n 16 -t 08:00:00 submit_pm.sh --config=bs512_opt_${opt} --amp_mode=fp16 --enable_apex
#   sbatch -n 16 -t 08:00:00 submit_pm.sh --config=bs512_opt_w1_ni15_${opt} --amp_mode=fp16 --enable_apex
#
#   sbatch -n 32 -t 08:00:00 submit_pm.sh --config=bs1024_opt_${opt} --amp_mode=fp16 --enable_apex
#   sbatch -n 32 -t 08:00:00 submit_pm.sh --config=bs1024_opt_w1_ni15_${opt} --amp_mode=fp16 --enable_apex
#
#   sbatch -n 64 -t 08:00:00 submit_pm.sh --config=bs2048_opt_${opt} --amp_mode=fp16 --enable_apex
#   sbatch -n 64 -t 08:00:00 submit_pm.sh --config=bs2048_opt_w1_ni15_${opt} --amp_mode=fp16 --enable_apex
#
#done

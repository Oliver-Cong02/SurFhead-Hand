#!/bin/bash

# job name
#SBATCH -J 1120_color2_grasp1_300_720_740_obj_50k

#SBATCH --partition=ssrinath-gcondo --gres=gpu:1 --gres-flags=enforce-binding
#SBATCH --account=ssrinath-gcondo
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00

# output
#SBATCH -o sbatch_info/1120_color2_grasp1_300_720_740_obj_50k.out

# error
#SBATCH -e sbatch_info/1120_color2_grasp1_300_720_740_obj_50k.err

# formatted=$(printf "%08d" $SLURM_ARRAY_TASK_ID)

module load cuda/11.8.0-lpttyok
module load ffmpeg

source /users/xcong2/data/users/xcong2/miniconda3/etc/profile.d/conda.sh
conda activate surfhead

cd /users/xcong2/data/users/xcong2/projects/SurFhead

sh train.sh 0
# sh test_circle.sh 0
# sh test.sh 0
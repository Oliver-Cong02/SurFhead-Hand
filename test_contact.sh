#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <gpu_id>"
    exit 1
fi

# Extract the GPU ID from the command-line argument
GPU_ID=$1

port=60100 #! example port number

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python render_contact.py \
-s "" -m output/1120_color2_grasp1_200_720_740_obj_50k --bind_to_mesh --depth_ratio 1 --rm_bg --sg_type 'asg'  \
# --iteration 120000 \

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python render_contact.py \
# -s "" -m output/color2_grasp1_1110_obj_50k_isotropic_1 --bind_to_mesh --depth_ratio 1 --rm_bg --sg_type 'asg'  \
# # --iteration 300000 \

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python render_contact.py \
# -s "" -m output/color2_grasp1_1111_obj_20k_handobjisotropic1_normal001 --bind_to_mesh --depth_ratio 1 --rm_bg --sg_type 'asg'  \
# # --iteration 300000 \

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python render_contact.py \
# -s "" -m output/color2_grasp1_1111_obj_20k_handobjisotropic1_normal00001 --bind_to_mesh --depth_ratio 1 --rm_bg --sg_type 'asg'   \
# # --iteration 300000 \

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python render_contact.py \
# -s "" -m output/color2_grasp1_1111_obj_50k_handobjisotropic1 --bind_to_mesh --depth_ratio 1 --rm_bg --sg_type 'asg'  \
# # --iteration 300000 \

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python render_contact.py \
# -s "" -m output/color2_grasp1_1111_obj_50k_handobjisotropic1_normal1 --bind_to_mesh --depth_ratio 1 --rm_bg --sg_type 'asg'  \
# # --iteration 300000 \

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python render_contact.py \
# -s "" -m output/color2_grasp1_1111_obj_50k_handobjisotropic1_normal001 --bind_to_mesh --depth_ratio 1 --rm_bg --sg_type 'asg'  \
# # --iteration 300000 \

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python render_contact.py \
# -s "" -m output/color2_grasp1_1111_obj_50k_handobjisotropic1_normal00001 --bind_to_mesh --depth_ratio 1 --rm_bg --sg_type 'asg'  \
# # --iteration 300000 \
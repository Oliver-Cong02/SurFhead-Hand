#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <gpu_id>"
    exit 1
fi

# Extract the GPU ID from the command-line argument
GPU_ID=$1

port=60001 #! example port number

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python render.py \
-s "" -m output/1110_onlymaskGT_obj --bind_to_mesh --depth_ratio 1 --rm_bg --sg_type 'asg'  \
# --iteration 120000 \
# --skip_train --skip_val 

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python render.py \
# -s "" -m output/obj_largemask_isotropic1 --bind_to_mesh --depth_ratio 1 --rm_bg --sg_type 'asg'  \
# # --iteration 120000 \
# # --skip_train --skip_val 

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python render.py \
# -s "" -m output/obj_smallmask --bind_to_mesh --depth_ratio 1 --rm_bg --sg_type 'asg'  \
# # --iteration 120000 \
# # --skip_train --skip_val 

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python render.py \
# -s "" -m output/obj_smallmask_isotropic1 --bind_to_mesh --depth_ratio 1 --rm_bg --sg_type 'asg'  \
# # --iteration 120000 \
# # --skip_train --skip_val 


# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python render.py \
# -s "" -m output/obj_densify_isotropic1 --bind_to_mesh --depth_ratio 1 --rm_bg --sg_type 'asg' --spec_only_eyeball --render_mesh \
# # --iteration 120000 \
# # --skip_train --skip_val 

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python render.py \
# -s "" -m output/obj_densify_shuffle --bind_to_mesh --depth_ratio 1 --rm_bg --sg_type 'asg' --spec_only_eyeball --render_mesh \
# # --iteration 120000 \
# # --skip_train --skip_val 

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python render.py \
# -s "" -m output/obj_densify --bind_to_mesh --depth_ratio 1 --rm_bg --sg_type 'asg' --spec_only_eyeball --render_mesh \
# # --iteration 120000 \
# # --skip_train --skip_val 

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python render.py \
# -s "" -m output/cam16_400_520_560_600k_obj_xyz_lr_densify --bind_to_mesh --depth_ratio 1 --rm_bg --sg_type 'asg' --spec_only_eyeball --render_mesh \
# # --iteration 120000  \
# # --skip_test --skip_val 
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python render.py \
# -s "" -m output/cam20_400_520_560_600k_obj_xyz_lr_densify --bind_to_mesh --depth_ratio 1 --rm_bg --sg_type 'asg' --spec_only_eyeball --render_mesh \
# # --iteration 120000 \
# # --skip_test --skip_val 

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python render.py \
# -s "" -m output/00_520_560_300k_obj_xyz_lr_normal5e-2 --bind_to_mesh --depth_ratio 1 --rm_bg --sg_type 'asg' --spec_only_eyeball 
# # --skip_train --skip_val \

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python render.py \
# -s "" -m output/400_520_560_300k_obj_xyz_lr_norma5e-2_dist100 --bind_to_mesh --depth_ratio 1 --rm_bg --sg_type 'asg' --spec_only_eyeball 
# # --skip_train --skip_val \

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python render.py \
# -s "" -m output/400_520_560_300k_obj_xyz_lr_norma5e-4 --bind_to_mesh --depth_ratio 1 --rm_bg --sg_type 'asg' --spec_only_eyeball 
# # --skip_train --skip_val \

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python render.py \
# -s "" -m output/400_520_560_300k_obj_xyz_lr_disk1e-2 --bind_to_mesh --depth_ratio 1 --rm_bg --sg_type 'asg' --spec_only_eyeball 
# # --skip_train --skip_val \

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python render.py \
# -s "" -m output/400_520_560_300k_obj_xyz_lr_disk1 --bind_to_mesh --depth_ratio 1 --rm_bg --sg_type 'asg' --spec_only_eyeball 
# # --skip_train --skip_val \

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python render.py \
# -s "" -m output/400_520_560_300k_obj_xyz_lr_normal5e-6 --bind_to_mesh --depth_ratio 1 --rm_bg --sg_type 'asg' --spec_only_eyeball 
# # --skip_train --skip_val \

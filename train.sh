#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <gpu_id>"
    exit 1
fi

# Extract the GPU ID from the command-line argument
GPU_ID=$1

port=60000 #! example port number

# large mask: "/users/xcong2/data/users/xcong2/data/brics/book1/dialated_masks"
# small mask "/users/xcong2/data/users/xcong2/projects/SurFhead/.cache/books1_grasp1/obj_dilated_hand_combine_mask"
# 1109 processed mask: "/users/xcong2/data/users/xcong2/projects/SurFhead/.cache/books1_grasp1/processed_obj_mask"

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
# -s "" \
# -m output/1110_onlymaskGT_obj_100k_isotropic1 \
# --port ${port} --eval --bind_to_mesh --lambda_normal 0.0 --lambda_dist 0.0 --depth_ratio 1 \
# --interval 60000 \
# --iterations 300000 \
# --sample_obj_pts_num 100000 \
# --raw_video_dir "/users/xcong2/data/datasets/BRICS/BRICS-DATA-02/neural-hands/chandradeep/grasps/2023-10-26_session_books1_grasp1/synced" \
# --grasp_path "/users/xcong2/data/datasets/MANUS_data/chandradeep/grasps/books1_grasp1/meta_data.pkl" \
# --cam_path "/users/xcong2/data/datasets/MANUS_data/chandradeep/calib.object/optim_params.txt" \
# --masks_path "/users/xcong2/data/users/xcong2/projects/SurFhead/.cache/books1_grasp1/processed_obj_mask" \
# --obj_mesh_paths "/users/xcong2/data/datasets/MANUS_data/chandradeep/objects/books1/mesh/ngp_mesh" \
# --start_frame_idx 400 --test_start_frame_idx 520 --end_frame_idx 560 --jump 1 --mask_iterations 0 --rm_bg --isotropic_loss_iter 0 --lambda_isotropic_weight 1.0 \
# # --merge_train_val \
# # --DTF --invT_Jacobian --lambda_normal_norm 0.0


# colored mesh : color2_grasp1
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
-s "" \
-m output/1120_color2_grasp1_300_720_740_obj_50k \
--port ${port} --eval --bind_to_mesh --lambda_normal 0.0001 --lambda_dist 0.0 --depth_ratio 1 \
--interval 60000 \
--sample_obj_pts_num 50000 \
--raw_video_dir "/users/xcong2/data/datasets/BRICS/BRICS-DATA-02/neural-hands/chandradeep/grasps/2023-10-27_session_color2_grasp1/synced" \
--grasp_path "/users/xcong2/data/datasets/MANUS_data/chandradeep/grasps/color2_grasp1/meta_data.pkl" \
--cam_path "/users/xcong2/data/datasets/MANUS_data/chandradeep/calib.object/optim_params.txt" \
--masks_path "/users/xcong2/data/users/xcong2/projects/SurFhead-Hand/.cache/color2_grasp1/processed_obj_mask" \
--obj_mesh_paths "/users/xcong2/data/datasets/MANUS_data/chandradeep/objects/color2/mesh/ngp_mesh/" \
--start_frame_idx 300 --test_start_frame_idx 720 --end_frame_idx 740 --jump 1 --mask_iterations 0 --rm_bg --isotropic_loss_iter 0 --lambda_isotropic_weight 1.0 \
# --merge_train_val \
# --DTF --invT_Jacobian --lambda_normal_norm 0.0
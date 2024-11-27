#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self.sg_degree = 24
        self._source_path = ""  # Path to the source data set
        self._target_path = ""  # Path to the target data set for pose and expression transfer
        self._model_path = ""  # Path to the folder to save trained models
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.skip_train_ = False
        self.skip_test_ = False
        self.skip_val_ = False
        self.bind_to_mesh = False
        self.disable_flame_static_offset = False
        self.not_finetune_flame_params = False
        self.select_camera_id = -1

        # ------- xiaoyan -------- #
        self.subject = "xcong2"
        self.exp_name = "test"
        self.grasp_path = "/users/xcong2/data/datasets/MANUS_data/chandradeep/grasps/color2_grasp1/meta_data.pkl"
        self.raw_video_dir = "/users/xcong2/data/datasets/BRICS/BRICS-DATA-02/neural-hands/chandradeep/grasps/2023-10-27_session_color2_grasp1/synced"
        self.obj_mesh_paths = ""
        # self.raw_video_dir = "/users/xcong2/data/brics/non-pii/brics-mini/2024-09-11-action-yixin-instrument-varun-files"
        # self.ith = 28
        # self.mano_param_path = f"/users/xcong2/data/datasets/Action/brics-mini/2024-09-11-action-yixin-instrument-varun-files/params/{self.ith:03d}.json"
        # self.cam_path = "/users/xcong2/data/datasets/Action/brics-mini/2024-09-11-action-yixin-instrument-varun-files/optim_params.txt"
        self.cam_path = "/users/xcong2/data/datasets/MANUS_data/chandradeep/calib.object/optim_params.txt"
        self.masks_path = "/users/xcong2/data/users/xcong2/projects/SurFhead/.cache/color2_grasp1/dilated_masks"
        self.width = 1280
        self.height = 720
        self.near = 0.01
        self.far = 100
        self.step_size = 10
        self.start_frame_idx = 300 # 300, debug: 500
        self.bg_color = "black"
        self.val_camera_name = "brics-sbc-005_cam0" # 
        self.merge_train_val = False
        self.test_start_frame_idx = 500 # 600
        self.end_frame_idx = 520 # 540 # 660
        self.not_finetune_MANO_params = False
        self.sample_obj_pts_num = 100000 # 20000
        self.jump = 1
        # ------------------------ #
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 0.0
        self.debug = False
        self.tight_pruning = False
        self.tight_pruning_threshold = 0.1
        self.train_kinematic = False
        self.DTF = False
        self.rm_bg = False
        self.invT_Jacobian = False
        self.SGs = False
        self.sg_type = 'asg'
        self.rotSH = False
        self.detach_boundary = False
        self.train_kinematic_dist = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        # 3D Gaussians
        self.iterations = 120_000  # 30_000 (original)
        self.mask_iterations = self.iterations // 3
        self.position_lr_init = 0.016  # (scaled up according to mean triangle scale)  #0.00016 (original)#! *1/0.032
        self.position_lr_final = 0.00016  # (scaled up according to mean triangle scale) # 0.0000016 (original)
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = self.iterations  # 30_000 (original)
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.023  # (scaled up according to mean triangle scale)  # 0.005 (original)
        self.rotation_lr = 0.001
        self.blend_weight_lr = 0.001
        self.densification_interval = self.iterations // 300 # 400  # 100 (original)
        self.opacity_reset_interval = self.iterations // 10 # 12_000 # 3000 (original)
        self.densify_from_iter = self.iterations // 15 # 2_000  # 500 (original)
        self.densify_until_iter = self.iterations * 2 // 3  # 15_000 (original)
        self.densify_grad_threshold = 0.0002

        # isotropic loss
        self.lambda_isotropic_weight = 0.001
        self.isotropic_loss_iter = 0

        
        # Object parameters
        self.obj_position_lr_init = 0.00016
        self.obj_position_lr_final = 0.0000016
        self.obj_position_lr_delay_mult = 0.01
        self.obj_position_lr_max_steps = self.iterations
        self.obj_feature_lr = 0.0025
        self.obj_opacity_lr = 0.05
        self.obj_scaling_lr = 0.005
        self.obj_rotation_lr = 0.001
        self.obj_percent_dense = 0.01


        # GaussianAvatars
        self.flame_expr_lr = 1e-3
        self.flame_trans_lr = 1e-6
        self.flame_pose_lr = 1e-5
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_xyz = 1e-2
        self.threshold_xyz = 1.
        self.metric_xyz = False
        self.lambda_scale = 1.
        self.threshold_scale = 0.6
        self.metric_scale = False
        self.lambda_dynamic_offset = 0.
        self.lambda_laplacian = 0.
        self.lambda_dynamic_offset_std = 0  #1.

        self.lambda_normal_iteration = int(self.iterations * 0.5) # 0.3
        self.lambda_dist_iteration = int(self.iterations * 0.1)
        
             # Xiaoyan: MANO optimization parameters
        self.mano_pose_lr = 0.0001
        self.mano_shape_lr = 0.0001
        self.mano_trans_lr = 0.000001
        self.mano_scale_lr = 0.000001


        #! for 2dgs
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_dist = 0.  # 100.0
        self.lambda_normal = 0.  # 0.05
        self.opacity_cull = 0.05
        
        #! for binding inheritance
        self.lambda_blend_weight = 0.
        self.lambda_normal_norm = 0.

        self.lambda_ids = 0.
        self.half_res = False
        self.densification_type = 'arithmetic_mean'

        self.lambda_eye_alpha = 0.
        # self.lambda_eye_scale = 0.
        # self.lambda_teeth_alpha = 0.
        self.lambda_laplacian_Jacobian = 0.
        self.specular_lr_max_steps = 300_000
        self.lambda_erank = 0.
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

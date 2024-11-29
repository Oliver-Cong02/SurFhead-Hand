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
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim, laplacian_loss, laplacian_loss_U, get_effective_rank

from gaussian_renderer import render, network_gui
from mesh_renderer import NVDiffRenderer
import sys
from scene import Scene, GaussianModel, UniGaussians
from utils.general_utils import safe_state, colormap
import uuid
from tqdm import tqdm


from utils.image_utils import psnr, error_map, visualize_gaussians_with_tensor
from lpipsPyTorch import lpips
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from kornia.losses import inverse_depth_smoothness_loss

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    
    specular_mlp = None
    tb_writer = prepare_output_and_logger(dataset)
    if dataset.bind_to_mesh:#!
        
        train_normal = False
        cal_laplacian = False
       
    
        uni_gaussians = UniGaussians(sh_degree=dataset.sh_degree, not_finetune_MANO_params=dataset.not_finetune_MANO_params, sample_obj_pts_num=dataset.sample_obj_pts_num,
                                     train_normal = train_normal, train_kinematic=pipe.train_kinematic, train_kinematic_dist = pipe.train_kinematic_dist, 
                                     DTF = pipe.DTF, invT_Jacobian=pipe.invT_Jacobian, obj_mesh_paths=dataset.obj_mesh_paths)

        try:
            mesh_renderer = NVDiffRenderer()
        except:
            mesh_renderer = None
            print("Mesh renderer not available")
    else:
        gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, uni_gaussians)
    uni_gaussians.hand_gaussian_model.training_setup(opt)
    uni_gaussians.obj_gaussian_model.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        uni_gaussians.hand_gaussian_model.restore(model_params, opt)
        uni_gaussians.obj_gaussian_model.restore(model_params, opt)
    # breakpoint()
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    num_workers = 16 # 8
    shuffle = False

    loader_camera_train = DataLoader(scene.getTrainCameras(), batch_size=None, shuffle=shuffle, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    iter_camera_train = iter(loader_camera_train)
   
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    for iteration in range(first_iter, opt.iterations + 1):     
      
        # han_window_iter = iteration * 2/(opt.iterations + 1)  
        han_window_iter = iteration /(opt.iterations + 1)  

        iter_start.record()

        uni_gaussians.hand_gaussian_model.update_learning_rate(iteration)
        uni_gaussians.obj_gaussian_model.update_learning_rate(iteration)


        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            uni_gaussians.hand_gaussian_model.oneupSHdegree()
            uni_gaussians.obj_gaussian_model.oneupSHdegree()

        try:
            viewpoint_cam = next(iter_camera_train)
        except StopIteration:
            iter_camera_train = iter(loader_camera_train)
            viewpoint_cam = next(iter_camera_train)

        if uni_gaussians.hand_gaussian_model.binding != None:
            uni_gaussians.hand_gaussian_model.select_mesh_by_timestep(viewpoint_cam.timestep)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
     
        specular_color = None
        
        render_pkg = render(viewpoint_cam, uni_gaussians, pipe, background,
                                            iter = han_window_iter,
                                            specular_color= specular_color)
                                        
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
       
        
        # visibility_filter_tight = render_pkg.get("visibility_filter_tight", None)
        
        gt_image = viewpoint_cam.original_image.cuda()
        mask = viewpoint_cam.original_mask.cuda()
        # breakpoint()
        if iteration > opt.mask_iterations:
            # print("Masked")
            # image = image * mask
            gt_image = gt_image * mask

        gaussians = uni_gaussians.hand_gaussian_model
        obj_visibility_filter = visibility_filter[:-gaussians.binding.shape[0]]
        obj_radii = radii[:-gaussians.binding.shape[0]]

        visibility_filter = visibility_filter[-gaussians.binding.shape[0]:]
        radii = radii[-gaussians.binding.shape[0]:]
       
        losses = {}
        losses['l1'] = l1_loss(image, gt_image) * (1.0 - opt.lambda_dssim)
        losses['ssim'] = (1.0 - ssim(image, gt_image)) * opt.lambda_dssim
        
        if iteration == 1 or ((iteration - 1) % opt.densification_interval == 0 \
            and (iteration - 1) >= opt.densify_from_iter):
            if opt.lambda_laplacian != 0:
                pre_compute_laplacian = gaussians.recalculate_points_laplacian()
        #! from 2dgs

        if opt.lambda_isotropic_weight != 0 and iteration > opt.isotropic_loss_iter:
            # breakpoint()
            losses['isotropic'] = opt.lambda_isotropic_weight * uni_gaussians.obj_gaussian_model.compute_isotropic_loss()
            # losses['isotropic'] = opt.lambda_isotropic_weight * (uni_gaussians.obj_gaussian_model.compute_isotropic_loss() + uni_gaussians.hand_gaussian_model.compute_isotropic_loss()) / 2.0

        # regularization

        if opt.iterations < 600000:
            lambda_normal = opt.lambda_normal if iteration > (opt.iterations // 2) else 0.0
            lambda_dist = opt.lambda_dist if iteration > (opt.iterations // 3) else 0.0
        else:
            lambda_normal = opt.lambda_normal if iteration > 140000 else 0.0
            lambda_dist = opt.lambda_dist if iteration > 60000 else 0.0

        if opt.lambda_blend_weight != 0:
        
            losses['blend_weight'] = opt.lambda_blend_weight * \
                F.relu(F.normalize(gaussians.get_blend_weight[visibility_filter],dim=-1,p=1)[...,1:] - 0.1).norm(dim=1).mean()
          
        if opt.lambda_normal_norm != 0 and (pipe.DTF):
            if pipe.train_kinematic or pipe.train_kinematic_dist:
                view_dir = gaussians.get_blended_xyz - viewpoint_cam.camera_center.cuda()
            else:
                view_dir = gaussians.get_xyz - viewpoint_cam.camera_center.cuda()
            normal = gaussians.get_normal#[visibility_filter]
            normal_normalised=  F.normalize(normal,dim=-1).detach()
            normal = normal * ((((view_dir * normal_normalised).sum(dim=-1) < 0) * 1 - 0.5) * 2)[...,None]
            normal = normal[visibility_filter]
            losses['normal_norm'] \
                 = torch.abs(normal.norm(dim=1) - 1).mean() * opt.lambda_normal_norm
             
        if lambda_normal != 0:
           
            rend_normal = render_pkg['surfel_rend_normal']
            surf_normal = render_pkg['surfel_surf_normal']
            
            if pipe.rm_bg:
                gt_mask = viewpoint_cam.original_mask.cuda()
                # surf_normal = F.normalize(surf_normal * gt_mask.repeat(3,1,1), dim =0)
                surf_normal = surf_normal * gt_mask.repeat(3,1,1)
                
                rend_normal = rend_normal * gt_mask.repeat(3,1,1)
                # if False:
                rend_normal = F.normalize(rend_normal, dim = 0)
            else:
                # surf_normal = render_pkg['surfel_surf_normal']
                pass
            
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            normal_loss = lambda_normal * (normal_error).mean()
            losses['surfel_normal_loss'] = normal_loss

        if lambda_dist != 0:
            rend_dist = render_pkg["surfel_rend_dist"]
            dist_loss = lambda_dist * (rend_dist).mean()
            losses['surfel_dist_loss'] = dist_loss
        
        if opt.lambda_ids != 0:
            # breakpoint()
            depth = render_pkg["surfel_surf_depth"]
            if opt.half_res:
                
                losses['ids'] = inverse_depth_smoothness_loss(F.interpolate(depth[None],scale_factor=0.5)
                                        , F.interpolate(image[None],scale_factor=0.5)) * opt.lambda_ids
            else:
                losses['ids'] = inverse_depth_smoothness_loss(depth[None], image[None]) * opt.lambda_ids
        
        
        if gaussians.binding != None:
                
            if opt.lambda_xyz != 0:
                if opt.metric_xyz:
                    losses['xyz'] = F.relu((gaussians._xyz*gaussians.face_scaling[gaussians.binding])[visibility_filter] - opt.threshold_xyz).norm(dim=1).mean() * opt.lambda_xyz
            
                else:
                    losses['xyz'] = F.relu(gaussians._xyz[visibility_filter].norm(dim=1) - opt.threshold_xyz).mean() * opt.lambda_xyz

            if opt.lambda_scale != 0:
                # breakpoint()
                if opt.metric_scale:
                    losses['scale'] = F.relu(gaussians.get_scaling[visibility_filter] - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale
                else:
                    losses['scale'] = F.relu(torch.exp(gaussians._scaling[visibility_filter]) - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale
                

            if opt.lambda_dynamic_offset != 0:
                losses['dy_off'] = gaussians.compute_dynamic_offset_loss() * opt.lambda_dynamic_offset

            if opt.lambda_dynamic_offset_std != 0:
                ti = viewpoint_cam.timestep
                t_indices =[ti]
                if ti > 0:
                    t_indices.append(ti-1)
                if ti < gaussians.num_timesteps - 1:
                    t_indices.append(ti+1)
                losses['dynamic_offset_std'] = gaussians.flame_param['dynamic_offset'].std(dim=0).mean() * opt.lambda_dynamic_offset_std
                
        losses['total'] = sum([v for k, v in losses.items()])
        
        losses['total'].backward()
        
        if pipe.detach_boundary:
            boundary_mask = torch.isin(gaussians.binding, gaussians.flame_model.mask.f.boundary)
            boundary_indices = torch.nonzero(boundary_mask).squeeze(1)
            if pipe.train_kinematic:
                gaussians.blend_weight.grad[boundary_indices] = 0

        iter_end.record()
       
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * losses['total'].item() + 0.6 * ema_loss_for_log
            if lambda_dist != 0:
                ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            else:
                ema_dist_for_log = 0.0
            if lambda_normal != 0:
                ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            else:
                ema_normal_for_log = 0.0

            if iteration % 10 == 0:
                postfix = {"Loss": f"{ema_loss_for_log:.{7}f}"}
                if 'xyz' in losses:
                    postfix["xyz"] = f"{losses['xyz']:.{7}f}"
                if 'scale' in losses:
                    postfix["scale"] = f"{losses['scale']:.{7}f}"
                if 'dy_off' in losses:
                    postfix["dy_off"] = f"{losses['dy_off']:.{7}f}"
                if 'lap' in losses:
                    postfix["lap"] = f"{losses['lap']:.{7}f}"
                if 'dynamic_offset_std' in losses:
                    postfix["dynamic_offset_std"] = f"{losses['dynamic_offset_std']:.{7}f}"
                if 'alpha' in losses:
                    postfix["alpha"] = f"{losses['alpha']:.{7}f}"
                if 'surfel_normal_loss' in losses:
                    postfix["surfel_normal_loss"] = f"{ema_normal_for_log:.{7}f}"
                    # postfix["surfel_normal_loss"] = f"{losses['surfel_normal_loss']:.{7}f}"
                if 'surfel_dist_loss' in losses:
                    # postfix["surfel_dist_loss"] = f"{losses['surfel_dist_loss']:.{7}f}"
                    postfix["surfel_dist_loss"] = f"{ema_dist_for_log:.{7}f}"
                if 'normal' in losses:
                    postfix["normal"] = f"{losses['normal']:.{7}f}"
             
                if 'normal_norm' in losses:
                    postfix['normal_norm'] = f"{losses['normal_norm']:.{7}f}"
       
                if 'lap_lbs' in losses:
                    postfix['lap_lbs'] = f"{losses['lap_lbs']:.{7}f}"
                if 'isotropic' in losses:
                    postfix['isotropic'] = f"{losses['isotropic']:.{7}f}"
            
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            render_args = (pipe, background, 1.0, None, han_window_iter)
            
       
            training_report(tb_writer, iteration, losses, iter_start.elapsed_time(iter_end), \
                            testing_iterations, scene, render, render_args, specular_mlp)
            
            if (iteration in saving_iterations):
                print("[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            viewspace_point_tensor_grad = viewspace_point_tensor.grad
            obj_viewspace_point_tensor_grad = viewspace_point_tensor_grad[:-gaussians.binding.shape[0]]
            hand_viewspace_point_tensor_grad = viewspace_point_tensor_grad[-gaussians.binding.shape[0]:]
            # breakpoint()
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats_my(hand_viewspace_point_tensor_grad, visibility_filter)
               
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    #! 10000 12000 ...
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    #! 10000 60000 120000 ...
                    
                    gaussians.reset_opacity()

            # # Object Densification
            # if iteration < opt.densify_until_iter:
            #     # import ipdb; ipdb.set_trace()
            #     gaussians = uni_gaussians.obj_gaussian_model
            #     visibility_filter = obj_visibility_filter
            #     radii = obj_radii
            #     # Keep track of max radii in image-space for pruning
            #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            #     gaussians.add_densification_stats_my(obj_viewspace_point_tensor_grad, visibility_filter)

            #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #         gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
            #     if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            #         gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                uni_gaussians.hand_gaussian_model.optimizer.step()
                uni_gaussians.hand_gaussian_model.optimizer.zero_grad(set_to_none = True)
                uni_gaussians.obj_gaussian_model.optimizer.step()
                uni_gaussians.obj_gaussian_model.optimizer.zero_grad(set_to_none = True)
            

            if (iteration in checkpoint_iterations):
                print("[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((uni_gaussians.hand_gaussian_model.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + "_hand.pth")
                torch.save((uni_gaussians.obj_gaussian_model.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + "_obj.pth")


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, losses, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, specular_mlp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', losses['l1'].item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss', losses['ssim'].item(), iteration)
        if 'xyz' in losses:
            tb_writer.add_scalar('train_loss_patches/xyz_loss', losses['xyz'].item(), iteration)
        if 'scale' in losses:
            tb_writer.add_scalar('train_loss_patches/scale_loss', losses['scale'].item(), iteration)
        if 'dynamic_offset' in losses:
            tb_writer.add_scalar('train_loss_patches/dynamic_offset', losses['dynamic_offset'].item(), iteration)
        if 'laplacian' in losses:
            tb_writer.add_scalar('train_loss_patches/laplacian', losses['laplacian'].item(), iteration)
        if 'dynamic_offset_std' in losses:
            tb_writer.add_scalar('train_loss_patches/dynamic_offset_std', losses['dynamic_offset_std'].item(), iteration)
        if 'alpha' in losses:
            tb_writer.add_scalar('train_loss_patches/alpha_loss', losses['alpha'].item(), iteration)
    
        if 'normal' in losses:
            tb_writer.add_scalar('train_loss_patches/normal_loss', losses['normal'].item(), iteration)
        if 'surfel_normal_loss' in losses:
            tb_writer.add_scalar('train_loss_patches/surfel_normal_loss', losses['surfel_normal_loss'].item(), iteration)
        if 'surfel_dist_loss' in losses:
            tb_writer.add_scalar('train_loss_patches/surfel_dist_loss', losses['surfel_dist_loss'].item(), iteration)

        if 'blend_weight' in losses:
            tb_writer.add_scalar('train_loss_patches/blend_weight', losses['blend_weight'].item(), iteration)
        if 'normal_norm' in losses:
            tb_writer.add_scalar('train_loss_patches/normal_norm', losses['normal_norm'].item(), iteration)
     
        if 'convex_eyeballs' in losses:
            tb_writer.add_scalar('train_loss_patches/convex_eyeballs', losses['convex_eyeballs'].item(), iteration)
        if 'eye_alpha' in losses:
            tb_writer.add_scalar('train_loss_patches/eye_alpha', losses['eye_alpha'].item(), iteration)

        tb_writer.add_scalar('train_loss_patches/total_loss', losses['total'].item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    testing_iterations_rough = [_iter * 2 for _iter in testing_iterations]
    if iteration in testing_iterations: #or iteration in [5000, 10000]:
        print("[ITER {}] Evaluating".format(iteration))
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'val', 'cameras' : scene.getValCameras()},
            {'name': 'test', 'cameras' : scene.getTestCameras()},
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                num_vis_img = 10 if len(config['cameras']) > 10 else len(config['cameras'])
                image_cache = []
                gt_image_cache = []
                vis_ct = 0
                for idx, viewpoint in tqdm(enumerate(DataLoader(config['cameras'], shuffle=False, batch_size=None, num_workers=8)), total=len(config['cameras'])):
                    if scene.uni_gaussians.hand_gaussian_model.num_timesteps > 1:
                        scene.uni_gaussians.hand_gaussian_model.select_mesh_by_timestep(viewpoint.timestep)
                    
                
                    
              
                    specular_color=None

                    render_pkg = renderFunc(viewpoint, scene.uni_gaussians, *renderArgs, specular_color)
                    
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    gt_mask = torch.clamp(viewpoint.original_mask.to("cuda"), 0.0, 1.0)
                    
                    if tb_writer and (idx % (len(config['cameras']) // num_vis_img) == 0):
                        tb_writer.add_images(config['name'] + "_{}/render".format(vis_ct), image[None], global_step=iteration)
                        error_image = error_map(image, gt_image)
                        tb_writer.add_images(config['name'] + "_{}/error".format(vis_ct), error_image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_{}/ground_truth".format(vis_ct), gt_image[None], global_step=iteration)
                        #! from NRFF
                    
                        try:
                            diffuse = render_pkg["rend_diffuse"]
                            specular = render_pkg['rend_specular']
                            tb_writer.add_images(config['name'] + "_{}/render_diffuse".format(vis_ct), diffuse[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_{}/render_specular".format(vis_ct), specular[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_{}/render_specular_2x".format(vis_ct), specular[None] * 2, global_step=iteration)
                            tb_writer.add_images(config['name'] + "_{}/render_specular_4x".format(vis_ct), specular[None] * 4, global_step=iteration)
                        except:
                            pass
                
                        #! from 2dgs
                        
                        depth = render_pkg["surfel_surf_depth"]
                        alpha = render_pkg['surfel_rend_alpha']
                        # breakpoint()
                        valid_depth_area = depth[gt_mask > 0.1]
                        max_depth_value = valid_depth_area.max()
                        min_depth_value = valid_depth_area.min()
                        
                        norm = max_depth_value - min_depth_value
                        depth[alpha < 0.1] = max_depth_value #! fill bg with max depth
                        depth = (depth - min_depth_value) / norm
                        # from torchvision.utils import save_image as si
                        # breakpoint()
                        # depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        depth = depth * gt_mask.repeat(3,1,1).cpu() + (1 - gt_mask.repeat(3,1,1).cpu())
                        tb_writer.add_images(config['name'] + "_{}/surfel_depth".format(vis_ct), depth[None], global_step=iteration)
                
                            
                        try:
                            rend_alpha = render_pkg['surfel_rend_alpha']
                            rend_normal = render_pkg["surfel_rend_normal"] * 0.5 + 0.5
                            surf_normal = (render_pkg["surfel_surf_normal"] * 0.5 + 0.5) * gt_mask.repeat(3,1,1).cuda + (1 - gt_mask.repeat(3,1,1).cuda())
                            tb_writer.add_images(config['name'] + "_{}/surfel_surfel_render_normal".format(vis_ct), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_{}/surfel_surfel_normal".format(vis_ct), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_{}/surfel_surfel_alpha".format(vis_ct), rend_alpha[None], global_step=iteration)
                          
                            rend_dist = render_pkg["surfel_rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass
                        
                        vis_ct += 1
                        
                        
                    image = viewpoint.original_mask.cuda() * image
                    gt_image = viewpoint.original_mask.cuda() * gt_image
                    # breakpoint()
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                    image_cache.append(image)
                    gt_image_cache.append(gt_image)

                    if idx == len(config['cameras']) - 1 or len(image_cache) == 16:
                        batch_img = torch.stack(image_cache, dim=0)
                        batch_gt_img = torch.stack(gt_image_cache, dim=0)
                        lpips_test += lpips(batch_img, batch_gt_img).sum().double()
                        image_cache = []
                        gt_image_cache = []

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                lpips_test /= len(config['cameras'])          
                ssim_test /= len(config['cameras'])          
                print("[ITER {}] Evaluating {}: L1 {:.4f} PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.uni_gaussians.get_opacity, iteration)
            tb_writer.add_histogram("scene/scale_histogram", torch.mean(scene.uni_gaussians.get_scaling, dim=-1), iteration)
            tb_writer.add_scalar('total_points', scene.uni_gaussians.get_xyz.shape[0], iteration)
            try:
                tb_writer.add_histogram('scene/blend_weight_primary',\
                    F.normalize(scene.uni_gaussians.hand_gaussian_model.get_blend_weight, dim=-1)[...,:1], iteration)
            except:
                pass
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--interval", type=int, default=30_000, help="A shared iteration interval for test and saving results and checkpoints.")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    if args.interval > op.iterations:
        args.interval = op.iterations // 5
    if len(args.test_iterations) == 0:
        args.test_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    if len(args.save_iterations) == 0:
        args.save_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    if len(args.checkpoint_iterations) == 0:
        args.checkpoint_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")

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

import torch
from torch.utils.data import DataLoader
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import concurrent.futures
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import joblib
from copy import deepcopy
# from utils.general_utils import colormap

from gaussian_renderer import render, render_contact
from utils.general_utils import safe_state
from utils.graphics_utils import focal2fov
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, UniGaussians
from mesh_renderer import NVDiffRenderer
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
from utils.image_utils import apply_depth_colormap, frames2video
from utils.brics_utils.vis_util import plot_points_in_image, project, get_colors_from_cmap
from utils.brics_utils.extra import *
from utils.brics_utils.cam_utils import get_opengl_camera_attributes
from scene.dataset_readers import CameraInfo
from scene import SpecularModel
#! import F
import torch.nn.functional as F
import glob
import subprocess
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import open3d as o3d
from torchvision.utils import save_image as si
try:
    # breakpoint()
    mesh_renderer = NVDiffRenderer()
except:
    print("Cannot import NVDiffRenderer. Mesh rendering will be disabled.")
    mesh_renderer = None


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def pose_spherical(theta, phi, radius, offset=None):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    if offset is not None:
        c2w[:3, 3] += offset
    return c2w

def pose_spherical_with_look_at(theta, phi, radius, center, look_at):
    """
    生成一个围绕给定中心旋转并朝向目标点的相机变换矩阵。
    
    参数：
    - theta: 水平角度，单位为度数。
    - phi: 垂直角度，单位为度数。
    - radius: 距离中心点的距离。
    - center: 围绕旋转的中心点 (x, y, z)。
    - look_at: 相机要朝向的目标点 (a, b, c)。

    返回：
    - c2w: 4x4 相机到世界的变换矩阵。
    """
    # 基于球面坐标计算相机的基本位置和旋转
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w

    # 将相机位置偏移到围绕 `center` 旋转
    c2w[:3, 3] += torch.Tensor(center)

    # 计算朝向矩阵，使相机指向 `look_at` 位置
    forward = torch.Tensor(look_at) - c2w[:3, 3]
    forward = forward / torch.norm(forward)
    right = torch.cross(torch.Tensor([0, 1, 0]), forward)
    right = right / torch.norm(right)
    up = torch.cross(forward, right)

    # 设置 c2w 的旋转部分，使其朝向 `look_at`
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward

    return c2w
def pose_spherical_xz_plane(theta, phi, radius, center, look_at):
    """
    生成一个在 xz 平面上围绕给定中心旋转并朝向目标点的相机变换矩阵。
    
    参数：
    - theta: 水平角度，单位为度数。
    - phi: 垂直角度，单位为度数。
    - radius: 距离中心点的距离。
    - center: 围绕旋转的中心点 (x, y, z)。
    - look_at: 相机要朝向的目标点 (a, b, c)。

    返回：
    - c2w: 4x4 相机到世界的变换矩阵。
    """
    # 设置初始的 z 轴平移，并在 xz 平面上围绕中心旋转
    c2w = trans_t(radius)
    c2w = rot_theta(theta / 180. * np.pi) @ c2w

    # 计算相机在 xz 平面上的位置
    c2w[1, 3] += radius * np.sin(phi / 180. * np.pi)  # 控制相机的 y 坐标偏移
    c2w[:3, 3] += torch.Tensor(center)  # 将相机位置偏移到 `center` 位置

    # 计算朝向矩阵，使相机指向 `look_at` 位置
    forward = torch.Tensor(look_at) - c2w[:3, 3]
    forward = forward / torch.norm(forward)
    right = torch.cross(torch.Tensor([0, 1, 0]), forward)
    right = right / torch.norm(right)
    up = torch.cross(forward, right)

    # 设置 c2w 的旋转部分，使其朝向 `look_at`
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward

    return c2w

# 生成一个围绕 (x, y, z) 旋转并朝向 (a, b, c) 的相机序列
def generate_c2w_sequence(center, look_at, radius, num_frames=40):
    """
    生成围绕 `center` 旋转并朝向 `look_at` 的相机变换矩阵序列。
    
    参数：
    - center: 围绕旋转的中心点 (x, y, z)。
    - look_at: 相机要朝向的目标点 (a, b, c)。
    - radius: 距离中心点的距离。
    - num_frames: 生成的相机帧数。
    
    返回：
    - c2w_sequence: 包含 `num_frames` 个相机变换矩阵的列表。
    """
    c2w_sequence = []
    for angle in np.linspace(-180, 180, num_frames, endpoint=False):
        # c2w = pose_spherical_with_look_at(angle, 0, radius, center, look_at)
        c2w = pose_spherical_xz_plane(angle, 0, radius, center, look_at)
        c2w_sequence.append(c2w)
    return c2w_sequence


def is_camera_facing_point_cloud(c2w, xyz):

    point_cloud_center = np.mean(xyz, axis=0)
    camera_position = c2w[:3, 3]
    direction_to_point_cloud = point_cloud_center - camera_position
    direction_to_point_cloud /= np.linalg.norm(direction_to_point_cloud)  # Normalize
    camera_forward = -c2w[:3, 2]
    dot_product = np.dot(camera_forward, direction_to_point_cloud)
    is_facing = dot_product > 0
    return is_facing

def write_data(path2data):
    for path, data in path2data.items():
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix in [".png", ".jpg"]:
            data = data.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            Image.fromarray(data).save(path)
        elif path.suffix in [".obj"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".txt"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".npz"]:
            np.savez(path, **data)
        elif path.suffix in [".npy"]:
            np.save(path, data)
        else:
            raise NotImplementedError(f"Unknown file type: {path.suffix}")
    
def render_set(dataset, name, iteration, views, gaussians, pipeline, background, render_mesh, extract_mesh, random_camera=0, specular= None):
    if dataset.select_camera_id != -1:
        name = f"{name}_camera_{dataset.select_camera_id}"
    iter_path = Path(dataset.model_path) / name / f"ours_{iteration}_circle"
    
    
    render_path = iter_path / "renders"
    render_alpha_path = iter_path / "render_alphas"
    render_depth_path = iter_path / "render_depths"
    render_analytic_normal_path = iter_path / "render_analytic_normals"
    render_tangent_normal_path = iter_path / "render_tangent_normals"

    os.makedirs(render_path, exist_ok=True)
    os.makedirs(render_alpha_path, exist_ok=True)
    os.makedirs(render_depth_path, exist_ok=True)
    os.makedirs(render_analytic_normal_path, exist_ok=True)
    os.makedirs(render_tangent_normal_path, exist_ok=True)

    views_loader = DataLoader(views, batch_size=None, shuffle=False, num_workers=8)
    max_threads = multiprocessing.cpu_count()
    print('Max threads: ', max_threads)
    worker_args = []
    ONLY_IMAGE = False
    timesteps = []
    for idx, view in enumerate(views_loader):
        if view.timestep not in timesteps:
            timesteps.append(view.timestep)

    timesteps = sorted(timesteps)
    gt_cams = []

    K = view.K
    width = view.image_width
    height = view.image_height
    mean_pts = torch.mean(gaussians.obj_gaussian_model.get_xyz.detach().cpu(), dim=0)
    center = mean_pts + torch.Tensor([0, 0.52, 0])
    # render_poses = [pose_spherical(angle, -30.0, 1, offset=mean_pts) for angle in np.linspace(-180,180,40+1)[:-1]]
    render_poses = generate_c2w_sequence(center, mean_pts, 0.4, num_frames=120)

    for idx, c2w in enumerate(render_poses):
        # is_camera_facing_point_cloud(c2w.cpu().numpy(), gaussians.obj_gaussian_model.get_xyz.detach().cpu().numpy())
        # blender -> opencv
        c2w = c2w @ torch.Tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        w2c = np.linalg.inv(c2w.cpu().numpy())
        RT = w2c
        # T = w2c[:3, 3] + np.array([0.47846505, 0.7205611 , 0.5177064 ])
        # R = np.transpose(w2c[:3, :3])
        # RT = np.eye(4)
        # RT[:3, :3] = R
        # RT[:3, 3] = T
        attr_dict = get_opengl_camera_attributes(K, RT[:3, :4], width, height)
        cam = deepcopy(view)
        cam.image_width = attr_dict['width']
        cam.image_height = attr_dict['height']
        cam.FoVx = attr_dict['fovx']
        cam.FoVy = attr_dict['fovy']
        cam.world_view_transform = to_tensor(attr_dict['world_view_transform'])
        cam.full_proj_transform = to_tensor(attr_dict['full_proj_transform'])
        cam.camera_center = to_tensor(attr_dict['camera_center'])
        # breakpoint()
        gt_cams.append(cam)

    for timestep in timesteps[-2:]:
        if gaussians.hand_gaussian_model.binding != None:
            gaussians.hand_gaussian_model.select_mesh_by_timestep(timestep)
        save_path = os.path.join(render_path, f'{timestep:05d}')
        os.makedirs(save_path, exist_ok=True)
        for idx, view in enumerate(gt_cams):
            # breakpoint()
            K = 600_000  # Temporarily fixed
            han_window_iter = iteration * 2 / (K + 1)
            specular_color = None
            render_bucket = render(view, gaussians, pipeline, background,
                                iter=han_window_iter,
                                specular_color= specular_color)

            rendering = render_bucket["render"]
        
            # render_alpha_path = render_alpha_path / f'{timestep:05d}'
            # render_depth_path = render_depth_path / f'{timestep:05d}'
            # render_analytic_normal_path = render_analytic_normal_path / f'{timestep:05d}'
            # render_tangent_normal_path = render_tangent_normal_path / f'{timestep:05d}'

            path2data = {
                Path(save_path) / f'{idx:05d}.png': rendering,
                # Path(render_alpha_path) / f'{idx:05d}.png': render_bucket['surfel_rend_alpha'].repeat(3,1,1) ,
                # Path(render_analytic_normal_path) / f'{idx:05d}.png': (render_bucket["surfel_surf_normal"] * 0.5 + 0.5) ,
                # Path(render_analytic_normal_path) / f'{idx:05d}.npy': render_bucket["surfel_surf_normal"].detach().cpu().numpy(),
                # Path(render_tangent_normal_path) / f'{idx:05d}.png': (render_bucket["surfel_rend_normal"] * 0.5 + 0.5),
                # Path(render_tangent_normal_path) / f'{idx:05d}.npy': render_bucket["surfel_rend_normal"].detach().cpu().numpy(),
            }
            
            # depth = render_bucket["surfel_surf_depth"]
            # alpha = render_bucket['surfel_rend_alpha']
            # valid_depth_area = depth
            # max_depth_value = valid_depth_area.max()
            # min_depth_value = valid_depth_area.min()
            
            # norm = max_depth_value - min_depth_value
            # depth[alpha < 0.1] = max_depth_value #! fill bg with max depth
            # depth = (depth - min_depth_value) / norm
            # # from torchvision.utils import save_image as si
            # # breakpoint()
            # # depth = depth / norm
            # depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
            # path2data[Path(render_depth_path) / f'{idx:05d}.png'] = depth
            worker_args.append(path2data)
            # breakpoint()
            if idx == len(gt_cams) - 1:
                with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
                    futures = [executor.submit(write_data, args) for args in worker_args]
                    for future in futures:
                        future.result()
                worker_args = []
    
        if timestep == timesteps[-1]:
            # Define the command as a list of strings
            command = [
                'ffmpeg',
                '-framerate', '30',
                '-i', os.path.join(save_path, '%05d.png'),
                '-c:v', 'libx264',
                '-r', '30',
                '-pix_fmt', 'yuv420p',
                os.path.join(render_path, f'{timestep:05d}.mp4')
            ]

            # Run the command
            subprocess.run(command)


            

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_val : bool, skip_test : bool, \
    render_mesh: bool, extract_mesh: bool, random_camera : int = 0):
    with torch.no_grad():
        if dataset.bind_to_mesh:
            uni_gaussians = UniGaussians(sh_degree=dataset.sh_degree, obj_mesh_paths=dataset.obj_mesh_paths)
            # gaussians = MANOGaussianModel(dataset.sh_degree, dataset.not_finetune_MANO_params)
            mesh_renderer = NVDiffRenderer()
        else:
            exit()
        scene = Scene(dataset, uni_gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        specular = None
        if pipeline.SGs:
            specular = SpecularModel()
            specular.load_weights(dataset.model_path)
        if dataset.target_path != "":
             name = os.path.basename(os.path.normpath(dataset.target_path))
             # when loading from a target path, test cameras are merged into the train cameras
             render_set(dataset, f'{name}', scene.loaded_iter, scene.getTrainCameras(), uni_gaussians, pipeline, background, render_mesh, extract_mesh, random_camera, specular)
        else:
            if not skip_train:
                # if random_camera !=0:
                render_set(dataset, "train", scene.loaded_iter, scene.getTrainCameras(), uni_gaussians, pipeline, background, render_mesh, extract_mesh, random_camera, specular)
                # else:
                    # render_set(dataset, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, render_mesh, extract_mesh)
            
            if not skip_val:
                render_set(dataset, "val", scene.loaded_iter, scene.getValCameras(), uni_gaussians, pipeline, background, render_mesh, extract_mesh, random_camera, specular)

            if not skip_test:
                render_set(dataset, "test", scene.loaded_iter, scene.getTestCameras(), uni_gaussians, pipeline, background, render_mesh, extract_mesh, random_camera, specular)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_val", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_mesh", action="store_true")
    parser.add_argument("--extract_mesh", action="store_true")
    parser.add_argument("--random_camera", default=20, type=int)
    # parser.add_argument("--render_normal", action="store_true")
    # parser.add_argument("--render_depth", action="store_true")
    # parser.add_argument("--render_neigh_normal", action="store_true")
    

    # # load camera json
    # import json
    # with open("/users/xcong2/data/users/xcong2/projects/SurFhead/output/color2_grasp1_1110_obj_20k/cameras.json", 'r') as f:
    #     camera_json = json.load(f)
    # breakpoint()


    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_val, args.skip_test,\
        args.render_mesh, args.extract_mesh, args.random_camera)
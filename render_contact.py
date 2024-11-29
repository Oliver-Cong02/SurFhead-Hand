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
import json
import joblib
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
from scene.cameras import Camera
from copy import deepcopy
#! import F
import torch.nn.functional as F
import glob
import matplotlib.pyplot as plt
import seaborn as sns
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


import taichi as ti
ti.init(arch=ti.cuda, device_memory_fraction=0.8)
def get_contact_dist(pt1, pt2):
    device = pt1.device

    ti_pt1 = ti.ndarray(shape=(pt1.shape[0], 3), dtype=ti.f32)
    ti_pt2 = ti.ndarray(shape=(pt2.shape[0], 3), dtype=ti.f32)
    ti_contact_map = ti.ndarray(shape=pt1.shape[0], dtype=ti.f32)
    ti_contact_indices = ti.ndarray(shape=pt1.shape[0], dtype=ti.f32)

    ti_pt1.from_numpy(to_numpy(pt1))
    ti_pt2.from_numpy(to_numpy(pt2))

    @ti.kernel
    def calculate_distances(
        ti_pt1: ti.types.ndarray(ndim=2),
        ti_pt2: ti.types.ndarray(ndim=2),
        ti_contact_map: ti.types.ndarray(ndim=1),
        ti_contact_indices: ti.types.ndarray(ndim=1),
    ):
        for i in range(pt1.shape[0]):
            min_dist = 1e9
            for j in range(pt2.shape[0]):
                dist = 0.0
                for k in range(3):
                    dist += (ti_pt1[i, k] - ti_pt2[j, k]) ** 2
                dist = ti.sqrt(dist)
                if dist < min_dist:
                    min_dist = dist
                    ti_contact_indices[i] = j
            ti_contact_map[i] = min_dist

    calculate_distances(ti_pt1, ti_pt2, ti_contact_map, ti_contact_indices)
    contact_map = attach(to_tensor(ti_contact_map.to_numpy()), device)
    contact_indices = attach(to_tensor(ti_contact_indices.to_numpy()), device)
    return contact_map, contact_indices

def get_colormap(pt1, pt2, c_thresh=0.004, cmap_type="gray"):
    dist, indices = get_contact_dist(pt1, pt2)
    # save_distribution_visualization(to_numpy(dist), './vis/contact/data_distribution_visualization.jpg')
    # breakpoint()
    dist = torch.clamp(dist.clone(), 0, c_thresh) / c_thresh
    dist = 1 - dist
    alpha = 4
    dist = (dist ** alpha) / (dist ** alpha + (1 - dist) ** alpha)
    colors = get_colors_from_cmap(to_numpy(dist), cmap_name=cmap_type)[..., :3]
    colors = attach(to_tensor(colors), pt1.device)
    return dist, indices, colors
def calculate_cmap(obj_xyz, hand_xyz, c_thresh=0.004, cmap_type = 'gray'): 
    dist_h, indices_h, cmap_h = get_colormap(hand_xyz, obj_xyz, c_thresh= c_thresh, cmap_type = cmap_type)
    return dist_h, cmap_h


def save_distribution_visualization(data, output_path='data_distribution_visualization.jpg'):
    """
    Generates and saves a distribution visualization of a 1D numpy array.
    
    Parameters:
        data (np.array): Input array with shape (N,)
        output_path (str): Path to save the output JPG image
    """
    # Create figure and axes for multiple plots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Data Distribution Visualizations")

    # Histogram
    axs[0, 0].hist(data, bins=30, edgecolor='black', alpha=0.7)
    axs[0, 0].set_title("Histogram")
    axs[0, 0].set_xlabel("Value")
    axs[0, 0].set_ylabel("Frequency")

    # KDE Plot
    sns.kdeplot(data, ax=axs[0, 1])
    axs[0, 1].set_title("KDE Plot")
    axs[0, 1].set_xlabel("Value")
    axs[0, 1].set_ylabel("Density")

    # Box Plot
    axs[1, 0].boxplot(data, vert=False)
    axs[1, 0].set_title("Box Plot")
    axs[1, 0].set_xlabel("Value")

    # Violin Plot
    sns.violinplot(data, orient="h", ax=axs[1, 1])
    axs[1, 1].set_title("Violin Plot")
    axs[1, 1].set_xlabel("Value")

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure as a JPG file
    fig.savefig(output_path, format='jpg')
    plt.close(fig)  # Close the plot to free memory

# Example usage
# A = np.random.randn(1000)
# save_distribution_visualization(A, 'output.jpg')



from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap

def colormap(img, cmap='jet'):
    # Normalize the image data to 0-1 range
    norm = Normalize(vmin=img.min(), vmax=img.max())
    img_normalized = norm(img)
    
    # Get the colormap
    cmap = get_cmap(cmap)
    
    # Apply the colormap
    img_colormap = cmap(img_normalized)
    
    # Remove the alpha channel
    img_colormap = img_colormap[:, :, :3]
    
    # Convert to torch tensor and permute dimensions to match (C, H, W) format
    img_colormap_torch = torch.from_numpy(img_colormap).float().permute(2, 0, 1)
    # breakpoint()
    return img_colormap_torch

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
        
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import math, random
class DummyCamera:
    def __init__(self, projection_matrix, world_view_transform, W, H, FoVx, FoVy):
        # self.projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=FoVx, fovY=FoVy).transpose(0,1).cuda()
        # self.R = R
        # self.T = T
        self.projection_matrix = projection_matrix
        self.world_view_transform = world_view_transform
        # self.world_view_transform = torch.tensor(getWorld2View2(R, T, np.array([0,0,0]), 1.0)).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.image_width = W
        self.image_height = H
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.original_mask = None

def LookAtPoseSampler(horizontal_mean, vertical_mean, lookat_position, horizontal_stddev=0, vertical_stddev=0, radius=1, batch_size=1, device='cpu'):
    h = torch.randn((batch_size, 1), device=device) * horizontal_stddev + horizontal_mean
    v = torch.randn((batch_size, 1), device=device) * vertical_stddev + vertical_mean
    v = torch.clamp(v, 1e-5, math.pi - 1e-5)

    theta = h
    v = v / math.pi
    phi = torch.arccos(1 - 2*v)

    camera_origins = torch.zeros((batch_size, 3), device=device)

    camera_origins[:, 0:1] = radius*torch.sin(phi) * torch.cos(math.pi-theta)
    camera_origins[:, 2:3] = radius*torch.sin(phi) * torch.sin(math.pi-theta)
    camera_origins[:, 1:2] = radius*torch.cos(phi)

    # forward_vectors = math_utils.normalize_vecs(-camera_origins)
    def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
        '''
        # Normalize vector lengths.
        '''
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True))
    
    forward_vectors = normalize_vecs(lookat_position - camera_origins)
    
    
    forward_vector = normalize_vecs(forward_vectors)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=camera_origins.device).expand_as(forward_vector)

    right_vector = -normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = normalize_vecs(torch.cross(forward_vector, right_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=camera_origins.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=camera_origins.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = camera_origins
    cam2world = (translation_matrix @ rotation_matrix)[:, :, :]
    assert(cam2world.shape[1:] == (4, 4))
    return cam2world

    
def render_set(dataset, name, iteration, gaussians, pipeline, background, render_mesh, extract_mesh, random_camera=0, specular= None):
    if dataset.select_camera_id != -1:
        name = f"{name}_camera_{dataset.select_camera_id}"
    iter_path = Path(dataset.model_path) / name / f"ours_{iteration}"

    timesteps = []
    # json_file = Path(dataset.model_path) / "cameras.json"
    # if json_file.exists():
    #     with open(json_file, "r") as f:
    #         cameras = json.load(f)
    #         for cam in cameras:
    #             if cam["timestep"] not in timesteps:
    #                 timesteps.append(cam["timestep"])
    # else:
    #     exit()

    start = 0
    end = 2

    timesteps = [i for i in range(start, end)]
    timesteps = sorted(timesteps)
    acc_dist = []
    c_threshold = 0.002 # 0.002

    render_path = iter_path / f"renders_c_thresh_{c_threshold}_start_{start}_end_{end}"
    os.makedirs(render_path, exist_ok=True)
    
    for timestep in timesteps:
        if gaussians.hand_gaussian_model.binding != None:
            gaussians.hand_gaussian_model.select_mesh_by_timestep(timestep)
        obj_xyz = gaussians.obj_gaussian_model.get_xyz
        hand_xyz = gaussians.hand_gaussian_model.get_xyz
        dist_h, cmap_h = calculate_cmap(obj_xyz, hand_xyz, c_thresh=c_threshold)
        acc_dist.append(dist_h)
    acc_dist = torch.stack(acc_dist).sum(axis=0)
    # breakpoint()
    root_dir = '/'.join(dataset.grasp_path.split('/')[:-3])
    seq_name = dataset.grasp_path.split('/')[-2].split('_')[0]
    gt_contact_masks = glob.glob(f"{root_dir}/evals/{seq_name}_action/gt_contacts_seg/*.png")
    gt_contacts = glob.glob(f"{root_dir}/evals/{seq_name}_action/gt_contacts/*.png")
    mano_poses = glob.glob(f"{root_dir}/evals/{seq_name}_action/mano/mano_params/*.npz")
    gt_cam_path = f"{root_dir}/evals/{seq_name}_action/gt_cam.pkl"

    cam_data = joblib.load(gt_cam_path)
    intrs = cam_data['intrs']
    extrs = cam_data['extrs']
    view = Camera(colmap_id=0, R=np.eye(3), T=[0, 0, 0], 
                    FoVx=1, FoVy=1, 
                    image_width=1080, image_height=1080,
                    bg=[0, 0, 0], 
                    image=None, 
                    image_path=None,
                    image_name=None,
                    mask=None,
                    mask_path=None,
                    mask_name=None,
                    normal = None,
                    normal_path = None,
                    normal_name = None,
                    uid=0, K=None,
                    timestep=0, data_device=args.data_device)
    gt_cam = []
    for i in range(len(intrs)): 
        fx, fy, cx, cy = intrs[i]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        RT = extrs[i]
        attr_dict = get_opengl_camera_attributes(K, RT[:3, :4], 1080, 1080)
        cam = deepcopy(view)
        cam.image_width = attr_dict['width']
        cam.image_height = attr_dict['height']
        cam.FoVx = attr_dict['fovx']
        cam.FoVy = attr_dict['fovy']
        cam.world_view_transform = to_tensor(attr_dict['world_view_transform'])
        cam.full_proj_transform = to_tensor(attr_dict['full_proj_transform'])
        cam.camera_center = to_tensor(attr_dict['camera_center'])
        gt_cam.append(cam)

    m_dict = defaultdict(list)
    for pose_path in mano_poses: 
        data = np.load(pose_path)
        fno = int(pose_path.split('/')[-1].split('.')[0])
        m_dict['pose'].append(data['angle'])
        m_dict['shape'].append(data['shape'])
        m_dict['trans'].append(data['trans'])
        m_dict['scale'].append(data['scale'])
        m_dict['fno'].append(fno)
    for k, v in m_dict.items():
        m_dict[k] = to_tensor(np.stack(v, axis=0)).cuda()
    
    gaussians.hand_gaussian_model.apply_mano_param(m_dict)
    # breakpoint()
    # save_distribution_visualization(to_numpy(acc_dist), './vis/contact/data_distribution_visualization.jpg')
    colors = get_colors_from_cmap(to_numpy(acc_dist), cmap_name='gray')[..., :3]
    colors = to_tensor(colors).cuda()

    for idx, (img_path, mask_path) in enumerate(zip(gt_contacts, gt_contact_masks)): 
        cam = gt_cam[idx]
        render_bucket = render_contact(cam, gaussians, pipeline, background, contact_color=colors)

        rendered_mask = render_bucket["render"].permute(1,2,0).detach().cpu().numpy()
        ## render only hand
        # rendered_mask = to_numpy(rendered['render'])
        # breakpoint()
        img = np.asarray(Image.open(img_path).convert("RGB")) / 255.0
        gt_mask = np.asarray(Image.open(mask_path).convert("L"))[..., None] / 255.0
        alpha = 0.3
        overlay_gt = img * alpha + (1 - alpha) * gt_mask
        overlay_rendered = img * alpha + (1 - alpha) * rendered_mask
        combined = np.hstack((img, overlay_rendered, overlay_gt))
        Image.fromarray((combined * 255).astype(np.uint8)).save(os.path.join(render_path, f"{idx:03d}.png"))
        # dump_image(combined * 255, 'test.png')

    exit()


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_val : bool, skip_test : bool, \
    render_mesh: bool, extract_mesh: bool, random_camera : int = 0):
    with torch.no_grad():
        if dataset.bind_to_mesh:
            uni_gaussians = UniGaussians(sh_degree=dataset.sh_degree, obj_mesh_paths=dataset.obj_mesh_paths)
            # gaussians = MANOGaussianModel(dataset.sh_degree, dataset.not_finetune_MANO_params)
            mesh_renderer = NVDiffRenderer()
        else:
            exit()

        scene = Scene(dataset, uni_gaussians, load_iteration=iteration, shuffle=False, eval_contact=True)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        specular = None

        render_set(dataset, "contact", iteration, uni_gaussians, pipeline, background, render_mesh, extract_mesh, random_camera, specular)

        # if dataset.target_path != "":
        #      name = os.path.basename(os.path.normpath(dataset.target_path))
        #      # when loading from a target path, test cameras are merged into the train cameras
        #      render_set(dataset, f'{name}', scene.loaded_iter, scene.getTrainCameras(), uni_gaussians, pipeline, background, render_mesh, extract_mesh, random_camera, specular)
        # else:
        #     render_set(dataset, "contact", scene.loaded_iter, [scene.getTrainCameras(), scene.getTestCameras()], uni_gaussians, pipeline, background, render_mesh, extract_mesh, random_camera, specular)

        

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
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_val, args.skip_test,\
        args.render_mesh, args.extract_mesh, args.random_camera)
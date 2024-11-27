from pathlib import Path
import numpy as np
import torch

from .obj_gaussian_model import ObjGaussianModel
from .mano_gaussian_model import MANOGaussianModel

import cv2
import os
from PIL import Image

def project(keypoints3d: np.ndarray, P: np.ndarray):
    """
    Project keypoints to 2D using

    Inputs -
        keypoints3d (N, 3): 3D keypoints
        P (V,3,4): Projection matrices
    Outputs -
        keypoints2d (V, N, 2): Projected 2D keypoints
    """
    hom = np.hstack((keypoints3d, np.ones((keypoints3d.shape[0], 1))))
    projected = np.matmul(P, hom.T).transpose(0, 2, 1)  # (V, N, 2)
    projected = (projected / projected[:, :, -1:])[:, :, :-1]
    return projected

def plot_mesh_in_image(points, faces, image, color=(255, 255, 255)):
    """
    Render the mesh onto the image by drawing the mesh faces.
    
    :param points: Projected 2D points of the mesh vertices.
    :param faces: Array of face indices defining the mesh triangles.
    :param image: The image onto which the mesh will be rendered.
    :param color: The color to fill the mesh triangles.
    :return: Image with the mesh rendered.
    """
    image = image.copy()
    for face in faces:
        pts = points[face].astype(np.int32)
        cv2.fillConvexPoly(image, pts, color)
    return image

def dump_image(img, path=None, return_img=False):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.max() <= 1.0:
        img = img * 255
    img = img.astype(np.uint8)
    if return_img:
        return img
    if path is None:
        raise Exception("Save path is None!!")

    img = Image.fromarray(img)
    img.save(path)

class UniGaussians:
    def __init__(self, sh_degree : int, ncomps=30, not_finetune_MANO_params=False, start_frame_idx=0, step_size=10, sample_obj_pts_num=50000, 
                 train_normal=False, train_kinematic=False, train_kinematic_dist=False, DTF=False, invT_Jacobian=False, obj_mesh_paths=None):

        self.obj_gaussian_model = ObjGaussianModel(sh_degree, sample_obj_pts_num=sample_obj_pts_num, obj_mesh_paths=obj_mesh_paths)
        self.hand_gaussian_model = MANOGaussianModel(sh_degree, ncomps, not_finetune_MANO_params, start_frame_idx, step_size,
                                                     train_normal=train_normal, train_kinematic=train_kinematic, train_kinematic_dist=train_kinematic_dist, DTF=DTF, invT_Jacobian=invT_Jacobian)

    @property
    def get_xyz(self):
        return torch.cat([self.obj_gaussian_model.get_xyz, self.hand_gaussian_model.get_xyz], dim=0)

    @property
    def active_sh_degree(self):
        return self.hand_gaussian_model.active_sh_degree
    
    @property
    def get_opacity(self):
        return torch.cat([self.obj_gaussian_model.get_opacity, self.hand_gaussian_model.get_opacity], dim=0)

    @property
    def get_scaling(self):
        return torch.cat([self.obj_gaussian_model.get_scaling, self.hand_gaussian_model.get_scaling], dim=0)
    
    @property
    def get_rotation(self):
        return torch.cat([self.obj_gaussian_model.get_rotation, self.hand_gaussian_model.get_rotation], dim=0)
    
    @property
    def get_features(self):
        return torch.cat([self.obj_gaussian_model.get_features, self.hand_gaussian_model.get_features], dim=0)
    
    @property
    def max_sh_degree(self):
        return self.hand_gaussian_model.max_sh_degree
    
    def get_covariance(self, scaling_modifier):
        return torch.cat([self.obj_gaussian_model.get_covariance(scaling_modifier), self.hand_gaussian_model.get_covariance(scaling_modifier)], dim=0)

    def generate_mask(self, camera, start_frame_idx=0, step_size=10):
        # import ipdb; ipdb.set_trace()
        timestep = camera.timestep
        print(f"Generating mask for timestep {timestep}")
  
        verts, joints, bone_T, pose_delta, shape_delta, verts_cano = self.hand_gaussian_model.mano_model(
            self.hand_gaussian_model.mano_param['pose'][timestep].unsqueeze(0),
            self.hand_gaussian_model.mano_param['shape'][timestep].unsqueeze(0),
        )
        verts = (verts/1000) * self.hand_gaussian_model.mano_param['scale'][timestep] + self.hand_gaussian_model.mano_param['trans'][timestep]  
        faces = self.hand_gaussian_model.mano_model.th_faces.cpu().numpy()

        K = camera.K
        R = camera.R
        T = camera.T[..., None]
        R = np.transpose(R)
        RT = np.concatenate([R, T], axis=1)
        P = K @ RT[:3, :4]

        verts = verts[0].detach().cpu().numpy()

        pts = project(verts, P[None])[0]

        # obj + hand

        # mano_plot = plot_mesh_in_image(pts, faces, (np.zeros((camera.height, camera.width, 3)) * 255).astype(np.uint8))
        # obj_pts = self.obj_gaussian_model.obj_mesh.vertices
        # obj_faces = self.obj_gaussian_model.obj_mesh.faces
        # obj_pts = project(obj_pts, P[None])[0]
        # mano_plot = plot_mesh_in_image(obj_pts, obj_faces, mano_plot)
        # save_dir = f"/users/xcong2/data/users/xcong2/projects/SurFhead/.cache/books1_grasp1/mano_mask/{(timestep * step_size + start_frame_idx):04d}"
        # os.makedirs(save_dir, exist_ok = True)
        # dump_image(mano_plot, os.path.join(save_dir, f'{camera.image_name}.png'))

        # hand
        mano_plot = plot_mesh_in_image(pts, faces, (np.zeros((camera.height, camera.width, 3)) * 255).astype(np.uint8))
        save_dir = f"/users/xcong2/data/users/xcong2/projects/SurFhead/.cache/color2_grasp1/mano_mask/{(timestep * step_size + start_frame_idx):04d}"
        os.makedirs(save_dir, exist_ok = True)
        dump_image(mano_plot, os.path.join(save_dir, f'{camera.image_name}.png'))

        # obj
        obj_pts = self.obj_gaussian_model.obj_mesh.vertices
        obj_faces = self.obj_gaussian_model.obj_mesh.faces
        obj_pts = project(obj_pts, P[None])[0]
        obj_plot = plot_mesh_in_image(obj_pts, obj_faces, (np.zeros((camera.height, camera.width, 3)) * 255).astype(np.uint8))
        save_dir = f"/users/xcong2/data/users/xcong2/projects/SurFhead/.cache/color2_grasp1/obj_mask/{(timestep * step_size + start_frame_idx):04d}"
        os.makedirs(save_dir, exist_ok = True)
        dump_image(obj_plot, os.path.join(save_dir, f'{camera.image_name}.png'))


class Bi_UniGaussians:
    def __init__(self, sh_degree : int, ncomps=30, not_finetune_MANO_params=False, start_frame_idx=0, step_size=10, sample_obj_pts_num=50000):
        self.obj_gaussian_model = ObjGaussianModel(sh_degree, sample_obj_pts_num=sample_obj_pts_num)
        self.left_hand_gaussian_model = MANOGaussianModel(sh_degree, ncomps, not_finetune_MANO_params, start_frame_idx, step_size)
        self.right_hand_gaussian_model = MANOGaussianModel(sh_degree, ncomps, not_finetune_MANO_params, start_frame_idx, step_size)

    @property
    def get_xyz(self):
        return torch.cat([self.obj_gaussian_model.get_xyz, self.left_hand_gaussian_model.get_xyz, self.right_hand_gaussian_model.get_xyz], dim=0)

    @property
    def active_sh_degree(self):
        return self.left_hand_gaussian_model.active_sh_degree
    
    @property
    def get_opacity(self):
        return torch.cat([self.obj_gaussian_model.get_opacity, self.left_hand_gaussian_model.get_opacity, self.right_hand_gaussian_model.get_opacity], dim=0)

    @property
    def get_scaling(self):
        return torch.cat([self.obj_gaussian_model.get_scaling, self.left_hand_gaussian_model.get_scaling, self.right_hand_gaussian_model.get_scaling], dim=0)
    
    @property
    def get_rotation(self):
        return torch.cat([self.obj_gaussian_model.get_rotation, self.left_hand_gaussian_model.get_rotation, self.right_hand_gaussian_model.get_rotation], dim=0)
    
    @property
    def get_features(self):
        return torch.cat([self.obj_gaussian_model.get_features, self.left_hand_gaussian_model.get_features, self.right_hand_gaussian_model.get_features], dim=0)
    
    @property
    def max_sh_degree(self):
        return self.left_hand_gaussian_model.max_sh_degree
    
    def get_covariance(self, scaling_modifier):
        return torch.cat([self.obj_gaussian_model.get_covariance(scaling_modifier), self.left_hand_gaussian_model.get_covariance(scaling_modifier), self.right_hand_gaussian_model.get_covariance(scaling_modifier)], dim=0)

    def load_meshes(self, train_meshes, test_meshes, tgt_train_meshes, tgt_test_meshes):
        self.left_hand_gaussian_model.load_meshes(train_meshes, test_meshes, tgt_train_meshes, tgt_test_meshes)
        self.right_hand_gaussian_model.load_meshes(train_meshes, test_meshes, tgt_train_meshes, tgt_test_meshes)

    def training_setup(self, opt):
        self.left_hand_gaussian_model.training_setup(opt)
        self.right_hand_gaussian_model.training_setup(opt)
        self.obj_gaussian_model.training_setup(opt)
    
    def restore(self, model_params, opt):
        self.left_hand_gaussian_model.restore(model_params, opt)
        self.right_hand_gaussian_model.restore(model_params, opt)
        self.obj_gaussian_model.restore(model_params, opt)

    def update_learning_rate(self, iteration):
        self.left_hand_gaussian_model.update_learning_rate(iteration)
        self.right_hand_gaussian_model.update_learning_rate(iteration)
        self.obj_gaussian_model.update_learning_rate(iteration)
    
    def oneupSHdegree(self):
        self.left_hand_gaussian_model.oneupSHdegree()
        self.right_hand_gaussian_model.oneupSHdegree()
        self.obj_gaussian_model.oneupSHdegree()
    
    def select_mesh_by_timestep(self, timestep):
        self.left_hand_gaussian_model.select_mesh_by_timestep(timestep)
        self.right_hand_gaussian_model.select_mesh_by_timestep(timestep)
        self.obj_gaussian_model.select_mesh_by_timestep(timestep)

    def generate_mask(self, camera):
        # import ipdb; ipdb.set_trace()
        timestep = camera.timestep
        K = camera.K
        R = camera.R
        T = camera.T[..., None]
        R = np.transpose(R)
        RT = np.concatenate([R, T], axis=1)
        P = K @ RT[:3, :4]

        print(f"Generating mask for timestep {timestep}")
        # left hand
        verts, joints, bone_T, pose_delta, shape_delta, verts_cano = self.left_hand_gaussian_model.mano_model(
            self.left_hand_gaussian_model.mano_param['pose'][timestep].unsqueeze(0),
            self.left_hand_gaussian_model.mano_param['shape'][timestep].unsqueeze(0),
        )
        left_verts = (verts/1000) * self.left_hand_gaussian_model.mano_param['scale'][timestep] + self.left_hand_gaussian_model.mano_param['trans'][timestep]  
        left_faces = self.left_hand_gaussian_model.mano_model.th_faces.cpu().numpy()
        left_verts = left_verts[0].detach().cpu().numpy()
        pts = project(left_verts, P[None])[0]

        # right hand
        verts, joints, bone_T, pose_delta, shape_delta, verts_cano = self.right_hand_gaussian_model.mano_model(
            self.right_hand_gaussian_model.mano_param['pose'][timestep].unsqueeze(0),
            self.right_hand_gaussian_model.mano_param['shape'][timestep].unsqueeze(0),
        )
        right_verts = (verts/1000) * self.right_hand_gaussian_model.mano_param['scale'][timestep] + self.right_hand_gaussian_model.mano_param['trans'][timestep]
        right_faces = self.right_hand_gaussian_model.mano_model.th_faces.cpu().numpy()
        right_verts = right_verts[0].detach().cpu().numpy()
        pts = project(right_verts, P[None])[0]
        mano_plot = plot_mesh_in_image(pts, right_faces, mano_plot)

        # object
        obj_pts = self.obj_gaussian_model.obj_mesh.vertices
        obj_faces = self.obj_gaussian_model.obj_mesh.faces
        obj_pts = project(obj_pts, P[None])[0]
        mano_plot = plot_mesh_in_image(obj_pts, obj_faces, mano_plot)
        save_dir = f"/users/xcong2/data/users/xcong2/data/brics/book1/mano_mask/{timestep:04d}"
        os.makedirs(save_dir, exist_ok = True)
        dump_image(mano_plot, os.path.join(save_dir, f'{camera.image_name}.png'))
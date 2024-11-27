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

import os
from copy import deepcopy
import random
import json
from typing import Union, List
import numpy as np
import torch
from utils.system_utils import searchForMaxIteration
# from projects.SurFhead.scene.dataset_readers_cp import sceneLoadTypeCallbacks
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from scene.uni_gaussians import UniGaussians
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.general_utils import PILtoTorch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Original_CameraDataset(torch.utils.data.Dataset):
    def __init__(self, cameras: List[Camera]):
        self.cameras = cameras

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            # ---- from readCamerasFromTransforms() ----
            camera = deepcopy(self.cameras[idx])
            
            if camera.image is None:
                image = Image.open(camera.image_path)
            else:
                image = camera.image
            # breakpoint()
            if camera.mask is None:
                mask = Image.open(camera.mask_path)
            else:
                mask = camera.mask
                
            if camera.normal_path is not None:
                
                normal = Image.open(camera.normal_path) 
                resized_normal_rgb = PILtoTorch(normal, (camera.image_width, camera.image_height))
                normal = resized_normal_rgb[:3, ...]
                camera.original_normal = normal.clamp(0.0, 1.0)
            else:
                camera.original_normal = None
            
                
            im_data = np.array(image.convert("RGBA"))
            
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + camera.bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
 
            # ---- from loadCam() and Camera.__init__() ----
            resized_image_rgb = PILtoTorch(image, (camera.image_width, camera.image_height))
        
            resized_mask_gray = PILtoTorch(mask, (camera.image_width, camera.image_height))[0:1, ...]

            image = resized_image_rgb[:3, ...]
            mask = resized_mask_gray
    
            if resized_image_rgb.shape[1] == 4:
                gt_alpha_mask = resized_image_rgb[3:4, ...]
                image *= gt_alpha_mask
            # camera.original_mask = torch.ones_like(mask)
            camera.original_mask = mask.clamp(0.0, 1.0)
            camera.original_image = image.clamp(0.0, 1.0)
            return camera
        elif isinstance(idx, slice):
            return CameraDataset(self.cameras[idx])
        else:
            raise TypeError("Invalid argument type")

class CameraDataset(torch.utils.data.Dataset):
    def __init__(self, cameras: List[Camera]):
        self.cameras = cameras

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            # ---- from readCamerasFromTransforms() ----
            camera = deepcopy(self.cameras[idx])
            
            if camera.image is None:
                image = Image.open(camera.image_path)
            else:
                image = camera.image

            if camera.mask is None:
                mask = Image.open(camera.mask_path)
            else:
                mask = camera.mask

            if camera.normal_path is not None:
                
                normal = Image.open(camera.normal_path) 
                resized_normal_rgb = PILtoTorch(normal, (camera.image_width, camera.image_height))
                normal = resized_normal_rgb[:3, ...]
                camera.original_normal = normal.clamp(0.0, 1.0)
            else:
                camera.original_normal = None
 
            # ---- from loadCam() and Camera.__init__() ----
            resized_image_rgb = PILtoTorch(image, (camera.image_width, camera.image_height))
        
            resized_mask_gray = PILtoTorch(mask, (camera.image_width, camera.image_height))[0:1, ...]

            image = resized_image_rgb
            mask = resized_mask_gray
            camera.original_mask = mask.clamp(0.0, 1.0)
            camera.original_image = image.clamp(0.0, 1.0)
            return camera
        elif isinstance(idx, slice):
            return CameraDataset(self.cameras[idx])
        else:
            raise TypeError("Invalid argument type")
        
class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, uni_gaussians : Union[GaussianModel, UniGaussians], load_iteration=None, shuffle=True, resolution_scales=[1.0], eval_contact=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        # breakpoint()
        self.model_path = args.model_path
        self.loaded_iter = None
        self.uni_gaussians = uni_gaussians
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        if not eval_contact:
            # load dataset
            assert os.path.exists(args.source_path), "Source path does not exist: {}".format(args.source_path)
            if os.path.exists(os.path.join(args.source_path, "sparse")):
                scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
            elif os.path.exists(os.path.join(args.source_path, "canonical_flame_param.npz")): #!
                print("Found FLAME parameter, assuming dynamic NeRF data set!")
                scene_info = sceneLoadTypeCallbacks["DynamicNerf"](args.source_path, args.white_background, args.eval, target_path=args.target_path)
            elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
                print("Found transforms_train.json file, assuming Blender data set!")
                scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
            elif os.path.basename(args.source_path).startswith('FaceTalk'):
                print("Found FaceTalk data set!")
                scene_info = sceneLoadTypeCallbacks["MakeHuman"](args.source_path, args.white_background, args.eval)
            else:
                print("Found BRICS data set!")
                scene_info = sceneLoadTypeCallbacks["BRICS"](args)

            # process cameras
            self.train_cameras = {}
            self.val_cameras = {}
            self.test_cameras = {}
            
            if not self.loaded_iter:
                if uni_gaussians.hand_gaussian_model.binding == None:
                    with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                        dest_file.write(src_file.read())
                json_cams = []
                camlist = []
                if scene_info.test_cameras:
                    camlist.extend(scene_info.test_cameras)
                if scene_info.train_cameras:
                    camlist.extend(scene_info.train_cameras)
                if scene_info.val_cameras:
                    camlist.extend(scene_info.val_cameras)
                for id, cam in enumerate(camlist):
                    json_cams.append(camera_to_JSON(id, cam))
                with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                    json.dump(json_cams, file)

            if shuffle:
                random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
                random.shuffle(scene_info.val_cameras)  # Multi-res consistent random shuffling
                random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

            self.cameras_extent = scene_info.nerf_normalization["radius"]

            for resolution_scale in resolution_scales:
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
                print("Loading Validation Cameras")
                self.val_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.val_cameras, resolution_scale, args)
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            
            # process meshes
            if uni_gaussians.hand_gaussian_model.binding != None:
                self.uni_gaussians.hand_gaussian_model.load_meshes(scene_info.train_meshes, scene_info.test_meshes, 
                                        scene_info.tgt_train_meshes, scene_info.tgt_test_meshes)
            

        # # breakpoint()
        # # generate masks
        # for id, cam in enumerate(camlist):
        #     uni_gaussians.generate_mask(cam, start_frame_idx=args.start_frame_idx, step_size=args.step_size)
        # exit()
        # # generate masks

        # create gaussians
        if self.loaded_iter:
            self.uni_gaussians.hand_gaussian_model.load_ply(
                os.path.join(self.model_path,
                            "point_cloud",
                            "iteration_" + str(self.loaded_iter),
                            "point_cloud.ply"),
                has_target=args.target_path != "",
            )
            self.uni_gaussians.obj_gaussian_model.load_ply(
                os.path.join(self.model_path,
                            "point_cloud",
                            "iteration_" + str(self.loaded_iter),
                            "point_cloud_obj.ply")
            )
        else:
            self.uni_gaussians.hand_gaussian_model.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            self.uni_gaussians.obj_gaussian_model.create_from_pcd(self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.uni_gaussians.hand_gaussian_model.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.uni_gaussians.obj_gaussian_model.save_ply(os.path.join(point_cloud_path, "point_cloud_obj.ply"))

    def getTrainCameras(self, scale=1.0):
        return CameraDataset(self.train_cameras[scale])
    
    def getValCameras(self, scale=1.0):
        return CameraDataset(self.val_cameras[scale])

    def getTestCameras(self, scale=1.0):
        return CameraDataset(self.test_cameras[scale])
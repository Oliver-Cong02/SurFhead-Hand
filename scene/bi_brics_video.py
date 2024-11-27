import cv2
import numpy as np
import torch
import json
import glob
from natsort import natsorted
from PIL import Image
import h5py
from tqdm import tqdm, trange
import os 
import joblib
from loguru import logger
import math
from utils.brics_utils import params as param_utils
from utils.brics_utils.extra import *
from utils.brics_utils.transforms import project_points, apply_constraints_to_poses, build_kintree, euler_angles_to_quats, \
    convert_armature_space_to_world_space, get_pose_wrt_root, euler_angles_to_matrix, get_keypoints
from utils.brics_utils.cam_utils import get_opengl_camera_attributes, get_scene_extent, load_brics_cameras
from utils.brics_utils.extra import create_skinning_grid, create_voxel_grid
# from utils.brics_utils.reader import Reader
from utils.brics_utils.rao_reader_v2 import Reader

class Bi_Brics_Dataset(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    def __init__(self, opts, split='train'):
        self.resize_factor = 1.0
        self.near = opts.near
        self.far = opts.far
        self.bg_color = opts.bg_color
        self.opts = opts
        self.subject_id = opts.subject
        self.width = opts.width
        self.height = opts.height
        self.grasp_path = opts.grasp_path
        self.cam_dir = '/'.join(self.grasp_path.split('/')[:-3])
        self.grasp_dir = os.path.join(self.cam_dir, 'grasps')
        self.grasp_seq = self.grasp_path.split('/')[-2]
        self.jump = opts.jump
        self.masks_path = opts.masks_path
        self.mano_file_path = opts.mano_param_path

        self.step_size = opts.step_size
        cam_to_remove = ['brics-odroid-004_cam0', "brics-odroid-008_cam0", "brics-odroid-008_cam1", "brics-odroid-009_cam0",
                        "brics-odroid-013_cam0", "brics-odroid-013_cam1", "brics-odroid-014_cam0", "brics-odroid-018_cam0",
                        "brics-odroid-018_cam1", "brics-odroid-019_cam0", "brics-odroid-026_cam0", "brics-odroid-026_cam1",
                        "brics-odroid-027_cam0", "brics-odroid-027_cam1", "brics-odroid-028_cam0", "brics-odroid-029_cam0",
                        "brics-odroid-029_cam1", "brics-odroid-030_cam0", "brics-odroid-030_cam1"]
        self.reader = Reader("video", opts.raw_video_dir, undistort=False, cams_to_remove=cam_to_remove, ith=opts.ith)

        self.start_frame_idx = opts.start_frame_idx
        self.test_start_frame_idx = opts.test_start_frame_idx
        self.end_frame_idx = opts.end_frame_idx

        ## load hdf5
        self.h5_path = os.path.join(os.getcwd(), f'.cache/{self.grasp_seq}/annot.hdf5')
        
        cam_path = opts.cam_path
        self.load_all_cameras(cam_path)
        self.load_mano_poses(self.mano_file_path)
        self.color_bkgd = self.get_bg_color()

        super().__init__()

    def __len__(self):
        "return frame count for the sequence"
        return (self.reader.frame_count//self.step_size)

    def __getitem__(self, idx):
        data = self.fetch_data(idx)
        return data

    def load_mano_poses(self, mano_poses_file): 
        left_dict = defaultdict(list)
        right_dict = defaultdict(list)

        with open(mano_poses_file, 'r') as f:
            data = json.load(f)

        left_hand_mano = data['left']
        right_hand_mano = data['right']

        left_dict['pose'] = left_hand_mano['pose']
        left_dict['shape'] = left_hand_mano['shape']
        left_dict['']
        for idx, mano_params in enumerate(left_hand_mano):
            fno = mano_params['fno']
            angle = mano_params['angle']
            shape = mano_params['shape']
                                
        for pose_path in mano_poses: 
            # load json
            with open(pose_path, 'r') as f:
                data = json.load(f)
            left_hand_mano = data['left']
            right_hand_mano = data['right']
            fno = int(pose_path.split('/')[-1].split('.')[0])
            angle = data['angle']
            shape = data['shape']
            trans = data['trans']
            scale = data['scale']
            m_dict['pose'].append(data['angle'])
            m_dict['shape'].append(data['shape'])
            m_dict['trans'].append(data['trans'])
            m_dict['scale'].append(data['scale'])
            m_dict['fno'].append(fno)

        for k, v in m_dict.items():
            m_dict[k] = np.stack(v, axis=0)
        self.mano_poses = MANO(**m_dict)

    def get_bg_color(self):
        if self.bg_color == "random":
            color_bkgd = np.random.rand(3).astype(np.float32)
        elif self.bg_color == "white":
            color_bkgd = np.ones(3).astype(np.float32)
        elif self.bg_color == "black":
            color_bkgd = np.zeros(3).astype(np.float32)
        return color_bkgd

    def get_images_for_this_frame_no(self, fno): 
        frame, cur_frame = next(self.reader([fno]))
        assert (cur_frame == fno)

        f_dict = self.process_frames(frame, fno)
        return f_dict

    def process_frames(self, frames, fno):
        f_dict = { }
        for cam_name, image in frames.items(): 
            f_dict[cam_name] = {}

            cam = self.all_cameras[cam_name]
            image = param_utils.undistort_image(cam.distK, cam.K, cam.dist, image)
            image = image[..., :3] / 255.0
            image = image[..., ::-1]
            # mask = self.masks[cam_name][fno - self.start_frame_idx]
            #TODO: Fix thi

            # with h5py.File(self.h5_path, "r", ) as file:
            #     frames = file.get('frames')
            #     mask = frames[cam_name][fno - 200]
            # hmask = (mask == 1) * 1
            # omask  = (mask == 2) * 1
            # alpha = cv2.bitwise_or(hmask, omask)
            # image = np.concatenate([image, alpha[..., None]], axis = -1)

            f_dict[cam_name]['image'] = image

        return f_dict


    def fetch_data(self, idx):
        # fno is from 0 to len(frame_sequence) for which poses are given
        # atleast for now. 
        # ideally it should be nth frame

        fno = idx * self.step_size + self.start_frame_idx
        f_dict = self.get_images_for_this_frame_no(fno)
        info = [self.subject_id, self.grasp_seq, fno]

        data_dict = {
            'info': info,
            "images": to_tensor(f_dict), 
        }

        return data_dict
    
    def load_all_cameras(self, cam_path): 
        cameras = param_utils.read_params(cam_path)
        d_dict = defaultdict(list)
        for idx, cam in enumerate(cameras):
            extr = param_utils.get_extr(cam)
            K, dist = param_utils.get_intr(cam)
            cam_name = cam["cam_name"]
            new_K, roi = param_utils.get_undistort_params(K, dist, (self.width, self.height))
            new_K = new_K.astype(np.float32)
            extr = extr.astype(np.float32)

            attr_dict = get_opengl_camera_attributes(new_K, extr, self.width, self.height,
                                                     resize_factor=self.resize_factor)
            for k, v in attr_dict.items():
                d_dict[k].append(v)
            d_dict['cam_name'].append(cam_name)
            d_dict['dist'].append(dist)
            d_dict['distK'].append(K)

        for k, v in d_dict.items():
            d_dict[k] = np.stack(v, axis=0)

        self.all_cameras = Cameras(**d_dict)
        self.extent = get_scene_extent(self.all_cameras.camera_center)
import cv2
import os
import ffmpeg
import numpy as np
from glob import glob
from natsort import natsorted
from typing import Dict, Iterable, Generator, Tuple
import json
import ipdb
import datetime
from utils.brics_utils import params as param_utils

def parse_timestamp(filename):
    # Strip the extension and parse the remaining part as a datetime
    timestamp_str = filename.split('_')[-1].split('.')[0]
    return datetime.datetime.fromtimestamp(int(timestamp_str) / 1e6)

def find_closest_video(folder, anchor_timestamp):
    min_diff = datetime.timedelta(seconds=1)  # Max allowed difference
    closest_video = None

    for filename in os.listdir(folder):
        if filename.endswith('.mp4'):
            video_timestamp = parse_timestamp(filename)
            time_diff = abs(video_timestamp - anchor_timestamp)
            
            if time_diff < min_diff:
                min_diff = time_diff
                closest_video = filename
    
    return closest_video, min_diff

class Reader():
    iterator = []

    def __init__(
            self, inp_type: str, path: str, undistort: bool=False, cams_to_remove=[], ith: int=0, start_frame_path=None, cam_path=None
        ):
        """ith: the ith video in each folder will be processed."""
        self.type = inp_type
        self.ith = ith
        self.frame_count = int(1e9)
        self.start_frames = None
        self.path = path
        self.cams_to_remove = cams_to_remove
        self.to_delete = []
        self.undistort = undistort

        if self.undistort: 
            assert (cam_path is not None)
            self.cameras = param_utils.read_params(cam_path)
        else: 
            self.cameras = None

        if self.type == "video":
            self.streams = {}
            self.vids = []

            for cam in os.listdir(path):
                if 'imu' not in cam and len(glob(f"{path}/{cam}/*.mp4")) > self.ith:
                    if cam not in cams_to_remove:
                        self.vids.append(natsorted(glob(f"{path}/{cam}/*.mp4"))[self.ith])
                if len(self.vids) > 0:
                    break
            self.anchor_timestamp = parse_timestamp(self.vids[0])
            self.check_timestamp()
            self.init_videos()
            if start_frame_path:
                with open(start_frame_path, 'r') as file:
                    start_frames = json.load(file)
                self.start_frames = start_frames
        else:
            pass

        # Sanity checks
        assert(self.frame_count < int(1e9)), "No frames found"
        if self.frame_count <= 0:
            print("No frames found")
            
        self.cur_frame = 0
    
    def _get_next_frame(self, frame_idx) -> Dict[str, np.ndarray]:
        """ Get next frame (stride 1) from each camera"""
        self.cur_frame = frame_idx
        
        if self.cur_frame == self.frame_count:
            return {}

        frames = {}
        for cam_name, cam_cap in self.streams.items():
            if self.start_frames:
                start_frame = self.start_frames.get(cam_name, [0, 0])[0]
            else:
                start_frame = 0
            cam_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + start_frame)
            suc, frame = cam_cap.read()
            if not suc:
                raise RuntimeError(f"Couldn't retrieve frame from {cam_name}")
            
            if self.undistort: 
                idx = np.where(self.cameras[:]['cam_name']==cam_name)[0][0]
                cam = self.cameras[idx]
                assert (cam['cam_name'] == cam_name)
                extr = param_utils.get_extr(cam)
                K, dist = param_utils.get_intr(cam)
                new_K, roi = param_utils.get_undistort_params(K, dist, (frame.shape[1], frame.shape[0]))
                frame = param_utils.undistort_image(K, new_K, dist, frame)

            frames[cam_name] = frame
        
        return frames

    def check_timestamp(self):
        
        for cam in os.listdir(self.path):
            if 'imu' not in cam and cam not in self.cams_to_remove and cam not in self.vids[0]:
                closest_file, time_diff = find_closest_video(f"{self.path}/{cam}", self.anchor_timestamp)
                if closest_file:
                    self.vids.append(f"{self.path}/{cam}/{closest_file}")
                else:
                    self.to_delete.append(cam.split('/')[-1].rsplit('_', 0)[0])
            
    def reinit(self):
        """ Reinitialize the reader """
        if self.type == "video":
            self.release_videos()
            self.init_videos()

        self.cur_frame = 0

    def init_videos(self):
        """ Create video captures for each video
                ith: the ith video in each folder will be processed."""
        for vid in self.vids:
            cap = cv2.VideoCapture(vid)
            frame_count = int(ffmpeg.probe(vid, cmd="ffprobe")["streams"][0]["nb_frames"])
            self.frame_count = min(self.frame_count, frame_count)
            cam_name = os.path.basename(vid).split(".")[0]
            self.streams[cam_name] = cap
            
        self.frame_count -= 5 # To account for the last few frames that are corrupted

    def release_videos(self):
        for cap in self.streams.values():
            cap.release()
    
    def __call__(self, frames: Iterable[int]=[]):
        # Sort the frames so that we access them in order
        frames = sorted(frames)
        self.iterator = frames
        
        for frame_idx in frames:
            frame = self._get_next_frame(frame_idx)
            if not frame:
                break
                
            yield frame, self.cur_frame

        # Reinitialize the videos
        self.reinit()
    
    def vis_video_in_dir(self):
        import shutil
        video_vis_dir = f"/users/xcong2/data/users/xcong2/projects/local_visualization/brics_video"
        os.makedirs(video_vis_dir, exist_ok=True)
        for video_path in self.vids:
            shutil.copy(video_path, video_vis_dir)
        
if __name__ == "__main__":
    cam_to_remove = ['brics-odroid-004_cam0', "brics-odroid-008_cam0", "brics-odroid-008_cam1", "brics-odroid-009_cam0",
                     "brics-odroid-013_cam0", "brics-odroid-013_cam1", "brics-odroid-014_cam0", "brics-odroid-018_cam0",
                     "brics-odroid-018_cam1", "brics-odroid-019_cam0", "brics-odroid-026_cam0", "brics-odroid-026_cam1",
                     "brics-odroid-027_cam0", "brics-odroid-027_cam1", "brics-odroid-028_cam0", "brics-odroid-029_cam0",
                     "brics-odroid-029_cam1", "brics-odroid-030_cam0", "brics-odroid-030_cam1"]
    reader = Reader("video", "/users/xcong2/data/brics/non-pii/brics-mini/2024-09-11-action-yixin-instrument-varun-files", cams_to_remove=cam_to_remove, ith=28)
    reader.vis_video_in_dir()

    exit()
    for i in range(len(reader)):
        frames, frame_num = reader.get_frames()
        print(frame_num)
# Extended Colmap-format dataset loader

from .util import Rays, Intrin, similarity_from_cameras, select_or_shuffle_rays
import sys
import torch
import torch.nn.functional as F
from typing import NamedTuple, Optional, Union, List
from os import path
import os
import cv2
import imageio
from tqdm import tqdm
import json
import numpy as np
from warnings import warn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vendor import read_write_model

def get_c2w(image, pose_rescale_factor):
    R = read_write_model.qvec2rotmat(image.qvec)
    t = image.tvec.reshape([3, 1])

    t_world = -R.T @ t
    t_world = t_world * pose_rescale_factor

    # get camera to world transfromation
    bottom = np.array([0, 0, 0, 1.0]).reshape([1, 4])
    c2w = np.concatenate([np.concatenate([R.T, t_world], 1), bottom], 0)
    return c2w


class ColmapDatasetBase:
    split: str
    permutation: bool
    epoch_size: Optional[int]
    n_images: int
    h_full: int
    w_full: int
    intrins_full: Intrin
    c2w: torch.Tensor  # C2W OpenCV poses
    gt: Union[torch.Tensor, List[torch.Tensor]]   # RGB images
    device : Union[str, torch.device]

    def __init__(self):
        self.ndc_coeffs = (-1, -1)
        self.use_sphere_bound = False
        self.should_use_background = True # a hint
        self.use_sphere_bound = True
        self.scene_center = [0.0, 0.0, 0.0]
        self.scene_radius = [1.0, 1.0, 1.0]
        self.permutation = False

    def create_train_set(self, images):
        image_ids = []
        for image_id in images:
            if images[image_id].camera_id != 1:
                continue
            if image_id % 8 != 0:  # take 1/5 to be test data
                image_ids.append(image_id)
        return image_ids


    def create_test_set(self, images):
        image_ids = []
        for image_id in images:
            if images[image_id].camera_id != 1:
                continue
            if image_id % 8 == 0:  # take 1/8 to be test data
                image_ids.append(image_id)
        return image_ids


    def create_val_set(self, images):
        image_ids = []
        for image_id in images:
            if images[image_id].camera_id != 1:
                continue
            if image_id % 10 == 0:  # take 1/10 to be val data
                image_ids.append(image_id)
        return image_ids


    def shuffle_rays(self):
        """
        Shuffle all rays
        """
        if self.split == "train":
            del self.rays
            self.rays = select_or_shuffle_rays(self.rays_init, self.permutation,
                                               self.epoch_size, self.device)


    def gen_rays(self, factor=1):
        print(" Generating rays, scaling factor", factor)
        # Generate rays
        self.factor = factor
        self.h = self.h_full // factor
        self.w = self.w_full // factor
        true_factor = self.h_full / self.h
        self.intrins = self.intrins_full.scale(1.0 / true_factor)
        yy, xx = torch.meshgrid(
            torch.arange(self.h, dtype=torch.float32) + 0.5,
            torch.arange(self.w, dtype=torch.float32) + 0.5,
        )
        xx = (xx - self.intrins.cx) / self.intrins.fx
        yy = (yy - self.intrins.cy) / self.intrins.fy
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, yy, zz), dim=-1)  # OpenCV convention
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        dirs = dirs.reshape(1, -1, 3, 1)
        del xx, yy, zz
        dirs = (self.c2w[:, None, :3, :3] @ dirs)[..., 0]

        if factor != 1:
            gt = F.interpolate(
                self.gt.permute([0, 3, 1, 2]), size=(self.h, self.w), mode="area"
            ).permute([0, 2, 3, 1])
            gt = gt.reshape(self.n_images, -1, 3)
        else:
            gt = self.gt.reshape(self.n_images, -1, 3)
        origins = self.c2w[:, None, :3, 3].expand(-1, self.h * self.w, -1).contiguous()
        if self.split == "train":
            origins = origins.view(-1, 3)
            dirs = dirs.view(-1, 3)
            gt = gt.reshape(-1, 3)

        self.rays_init = Rays(origins=origins, dirs=dirs, gt=gt)
        self.rays = self.rays_init

    def get_image_size(self, i : int):
        # H, W
        if hasattr(self, 'image_size'):
            return tuple(self.image_size[i])
        else:
            return self.h, self.w


class ColmapDataset(ColmapDatasetBase):
    """
    Extended Colmap dataset loader
    """

    focal: float
    c2w: torch.Tensor  # (n_images, 4, 4)
    gt: torch.Tensor  # (n_images, h, w, 3)
    h: int
    w: int
    n_images: int
    rays: Optional[Rays]
    split: str

    def __init__(
        self,
        root,
        split,
        epoch_size : Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        scene_scale: Optional[float] = None,  # Scene scaling
        factor: int = 1,                      # Image scaling (on ray gen; use gen_rays(factor) to dynamically change scale)
        scale : Optional[float] = 1.0,                    # Image scaling (on load)
        permutation: bool = True,
        white_bkgd: bool = True,
        cam_scale_factor : float = 0.95,
        normalize_by_camera: bool = True,
        **kwargs
    ):
        super().__init__()
        assert path.isdir(root), f"'{root}' is not a directory"

        if scene_scale is None:
            scene_scale = 1.0
        if scale is None:
            scale = 1.0

        self.device = device
        self.permutation = permutation
        self.epoch_size = epoch_size

        all_gt = []
        self.pose_rescale_factor = 0.01

        print("LOAD COLMAP DATA", root, ', split:', split, ', scale:', scale)

        self.split = split
        cameras, images, points3D = read_write_model.read_model(path.join(root, "sparse"))

        all_img_poses = {}
        # read all the image poses
        for image_id in images:
            all_img_poses[image_id] = get_c2w(images[image_id], self.pose_rescale_factor)

        # Select subset of files
        if self.split == "train" or self.split == "test_train":
            self.image_ids = self.create_train_set(images)
        elif self.split == "val":
            self.image_ids = self.create_val_set(images)
        elif self.split == "test":
            self.image_ids = self.create_test_set(images)
        else:
            self.image_ids = self.create_train_set(images)

        assert len(self.image_ids) > 0, "No images in directory: " + path.join(root, "images")
        print("  - find", len(self.image_ids), 'images')

        all_c2w = []
        for image_id in tqdm(self.image_ids):
            img_path = path.join(root, "images", images[image_id].name)
            image = imageio.imread(img_path)
            all_c2w.append(torch.from_numpy(all_img_poses[image_id]))
            full_size = list(image.shape[:2])
            assert full_size[0] > 0 and full_size[1] > 0, "Empty images"
            rsz_h, rsz_w = [round(hw * scale) for hw in full_size]

            if scale < 1:  # dynamic resize
                image = cv2.resize(image, (rsz_w, rsz_h), interpolation=cv2.INTER_AREA)
            all_gt.append(torch.from_numpy(image))

        self.c2w_f64 = torch.stack(all_c2w)

        if normalize_by_camera:
            norm_poses = np.stack([all_img_poses[image_id] for image_id in all_img_poses], axis=0)
            T, sscale = similarity_from_cameras(norm_poses)  # Select subset of files

            self.c2w_f64 = torch.from_numpy(T) @ self.c2w_f64
            scene_scale = cam_scale_factor * sscale

        print('  - scene_scale', scene_scale)
        self.c2w_f64[:, :3, 3] *= scene_scale
        self.c2w = self.c2w_f64.float()

        self.gt = torch.stack(all_gt).double() / 255.0
        if self.gt.size(-1) == 4:
            if white_bkgd: # Apply alpha channel
                self.gt = self.gt[..., :3] * self.gt[..., 3:] + (1.0 - self.gt[..., 3:])
            else:
                self.gt = self.gt[..., :3]
        self.gt = self.gt.float()
        self.n_images, self.h_full, self.w_full, _ = self.gt.shape

        # read intrinsics
        self.camera_matrices = {}
        for camera_id in cameras:
            self.camera_matrices[camera_id] = scale * read_write_model.get_intrinsics_matrix(cameras[camera_id])

        # pick camera 1 to be intrins_full
        picked_cam = self.camera_matrices[1]
        self.intrins_full : Intrin = Intrin(picked_cam[0, 0], picked_cam[1, 1], picked_cam[0, 2], picked_cam[1, 2])
        print(' intrinsics (loaded reso)', self.intrins_full)

        self.scene_scale = scene_scale
        if self.split == "train":
            self.gen_rays(factor=factor)
        else:
            # Rays are not needed for testing
            self.h, self.w = self.h_full, self.w_full
            self.intrins : Intrin = self.intrins_full

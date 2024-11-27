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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.brics_utils.sh_utils import eval_sh

def calculate_colors_from_sh(
    posed_means, cano_features, cano_means, camera, sh_degree, tf
):
    shs_view = cano_features.transpose(1, 2).view(-1, 3, (sh_degree + 1) ** 2)
    camera_center = camera.camera_center.repeat(cano_features.shape[0], 1)

    if tf is not None:
        cam_inv = torch.einsum(
            "nij, nj->ni", torch.linalg.inv(tf), homo(camera_center)
        )[..., :3]
        dir_pp_inv = cano_means - cam_inv
        dir_pp_normalized = dir_pp_inv / dir_pp_inv.norm(dim=1, keepdim=True)
    else:
        dir_pp = posed_means - camera_center
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)

    sh2rgb = eval_sh(sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    return colors_precomp

def render(
    posed_means,
    posed_cov,
    cano_means,
    cano_features,
    cano_opacity,
    camera,
    bg_color,
    sh_degree, 
    colors_precomp=None,
    tf=None,
    device=torch.device("cuda"),
):
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            posed_means, dtype=posed_means.dtype, requires_grad=True, device=device
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(camera.fovx * 0.5)
    tanfovy = math.tan(camera.fovy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(camera.height),
        image_width=int(camera.width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1,
        viewmatrix=camera.world_view_transform.to(device),
        projmatrix=camera.full_proj_transform.to(device),
        sh_degree=sh_degree,
        campos=camera.camera_center.to(device),
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    opacity = cano_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.

    if colors_precomp is None:
        colors_precomp = calculate_colors_from_sh(
            posed_means, cano_features, cano_means, camera, sh_degree, tf
        )

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=posed_means,
        means2D=means2D,
        shs=None,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=None,
        rotations=None,
        cov3D_precomp=posed_cov,
    )

    rendered_image = torch.permute(rendered_image, (1, 2, 0))

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image.clamp(0, 1),
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }


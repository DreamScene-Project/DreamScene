import os
from typing import NamedTuple

import numpy as np
import open3d as o3d
import torch
from loguru import logger
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from utils.pointe_utils import init_from_pointe
from utils.sh_utils import RGB2SH, SH2RGB
from utils.system_utils import hash_prompt


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def get_expon_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000):

    def helper(step):
        if lr_init == lr_final:
            # constant lr, ignore other params
            return lr_init
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def gaussian_3d_coeff(xyzs, covs):
    # xyzs: [N, 3]
    # covs: [N, 6]
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
    a, b, c, d, e, f = (
        covs[:, 0],
        covs[:, 1],
        covs[:, 2],
        covs[:, 3],
        covs[:, 4],
        covs[:, 5],
    )

    # eps must be small enough !!!
    inv_det = 1 / (a * d * f + 2 * e * c * b - e**2 * a - c**2 * d - b**2 * f + 1e-24)
    inv_a = (d * f - e**2) * inv_det
    inv_b = (e * c - b * f) * inv_det
    inv_c = (e * b - c * d) * inv_det
    inv_d = (a * f - c**2) * inv_det
    inv_e = (b * c - e * a) * inv_det
    inv_f = (a * d - b**2) * inv_det

    power = -0.5 * (x**2 * inv_a + y**2 * inv_d + z**2 * inv_f) - x * y * inv_b - x * z * inv_c - y * z * inv_e

    power[power > 0] = -1e10  # abnormal values... make weights 0

    return torch.exp(power)


def build_rotation(r):
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, initialize_param: dict, gaussian_class: str, exp_path: str = None):
        self.active_sh_degree = 0
        self.max_sh_degree = initialize_param["sh_degree"]
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._background = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        if gaussian_class == "object":
            if exp_path is None:
                raise ValueError("Error exp_path")
            self.init_pcd(initialize_param, exp_path)
        elif gaussian_class == "scene":
            pass
        elif gaussian_class == "tmp":
            self.load_ply(initialize_param["path"], False)
        elif gaussian_class == "restore":
            self.load_ply(initialize_param["path"], True)
        elif gaussian_class == "env":
            self.init_env_pcd(initialize_param)
        elif gaussian_class == "floor":
            self.init_floor_pcd(initialize_param)
        else:
            raise ValueError("Error Gaussian Class")

    def init_env_pcd(self, initialize_param: dict):
        guided = initialize_param["cam_pose_method"]
        if guided == "indoor":
            num_pts = 400000
            scene_box = initialize_param["scene_box"]
            boxs = np.ones((num_pts, 6)) * scene_box  # + np.random.random((num_pts,6))/50.0-0.01
            boxs[:, :3] -= np.random.random((num_pts, 3)) / 50.0
            boxs[:, 3:] += np.random.random((num_pts, 3)) / 50.0

            xs = np.random.random((num_pts,)) * (scene_box[3] - scene_box[0]) + scene_box[0]
            ys = np.random.random((num_pts,)) * (scene_box[4] - scene_box[1]) + scene_box[1]
            zs = np.random.random((num_pts,)) * (scene_box[5] - scene_box[2]) + scene_box[2]
            xyz = np.zeros((num_pts * 5, 3))
            xyz[:num_pts, :] = np.stack((boxs[:, 0], ys, zs), axis=1)
            xyz[num_pts : num_pts * 2, :] = np.stack((boxs[:, 3], ys, zs), axis=1)
            xyz[num_pts * 2 : num_pts * 3, :] = np.stack((xs, boxs[:, 1], zs), axis=1)
            xyz[num_pts * 3 : num_pts * 4, :] = np.stack((xs, boxs[:, 4], zs), axis=1)
            xyz[num_pts * 4 :, :] = np.stack((xs, ys, boxs[:, 5]), axis=1)

            colors = np.concatenate(
                (
                    0.5 * np.ones((num_pts, 3)),
                    0.5 * np.ones((num_pts, 3)),
                    0.7 * np.ones((num_pts, 3)),
                    0.7 * np.ones((num_pts, 3)),
                    0.9 * np.ones((num_pts, 3)),
                ),
                axis=0,
            )
            pcd = BasicPointCloud(points=xyz, colors=colors, normals=np.zeros((num_pts * 5, 3)))
            self.create_from_pcd(pcd, True, 1)
        elif guided == "outdoor":
            scene_box = np.absolute(initialize_param["scene_box"])
            zero_ground = initialize_param["zero_ground"]

            radius_base = np.sqrt(np.sum(np.max([scene_box[:3], scene_box[3:]], axis=0) ** 2))
            num_pts = np.ceil(radius_base * 50000).astype(int)
            phis = np.random.random((num_pts,)) * 2 * np.pi
            if zero_ground:
                costheta = np.random.random((num_pts,))
            else:
                costheta = np.random.random((num_pts,)) * 2 - 1
            thetas = np.arccos(costheta)
            mu = np.random.random((num_pts,)) / 10 + 0.95
            radius = radius_base * np.cbrt(mu)
            x = radius * np.sin(thetas) * np.cos(phis)
            y = radius * np.sin(thetas) * np.sin(phis)
            z = radius * np.cos(thetas)

            xyz = np.stack((x, y, z), axis=1)

            colors = np.ones((num_pts, 3))
            colors[:, 0] = min(initialize_param["env_init_color"][0] / 255.0, 1.0)
            colors[:, 1] = min(initialize_param["env_init_color"][1] / 255.0, 1.0)
            colors[:, 2] = min(initialize_param["env_init_color"][2] / 255.0, 1.0)
            pcd = BasicPointCloud(points=xyz, colors=colors, normals=np.zeros((num_pts, 3)))

            self.create_from_pcd(pcd, True, 1)
        else:
            pass

    def init_floor_pcd(self, initialize_param: dict):
        guided = initialize_param["cam_pose_method"]
        if guided == "indoor":
            num_pts = 300000
            scene_box = initialize_param["scene_box"]
            boxs = (
                np.ones((num_pts, 6)) * scene_box + np.random.random((num_pts, 6)) / 50.0 - 0.01
            )  # + np.random.random((num_pts,6))/10.0-0.05
            xs = np.random.random((num_pts,)) * (scene_box[3] - scene_box[0]) + scene_box[0]
            ys = np.random.random((num_pts,)) * (scene_box[4] - scene_box[1]) + scene_box[1]
            xyz = np.zeros((num_pts, 3))
            xyz[:num_pts, :] = np.stack((xs, ys, boxs[:, 2]), axis=1)
            colors = np.ones((num_pts, 3))
            colors[:, 0] = min(initialize_param["floor_init_color"][0] / 255.0, 1.0)
            colors[:, 1] = min(initialize_param["floor_init_color"][1] / 255.0, 1.0)
            colors[:, 2] = min(initialize_param["floor_init_color"][2] / 255.0, 1.0)
            pcd = BasicPointCloud(points=xyz, colors=colors, normals=np.zeros((num_pts, 3)))
            self.create_from_pcd(pcd, False, 1)
        elif guided == "outdoor":
            scene_box = np.absolute(initialize_param["scene_box"])
            zero_ground = initialize_param["zero_ground"]

            radius_base = np.sqrt(np.sum(np.max([scene_box[:3], scene_box[3:]], axis=0) ** 2))
            if zero_ground:
                ground_pts = np.ceil(radius_base * 20000).astype(int)
                ground_mu = np.random.random((ground_pts,))
                ground_radius = radius_base * np.sqrt(ground_mu)
                ground_phis = np.random.random((ground_pts,)) * 2 * np.pi
                ground_x = ground_radius * np.cos(ground_phis)
                ground_y = ground_radius * np.sin(ground_phis)

                ground_z = np.random.random((ground_pts,)) / 10.0 - 0.1 + initialize_param["scene_box"][2]
                ground_xyz = np.stack((ground_x, ground_y, ground_z), axis=1)

            if zero_ground:
                colors = np.ones((ground_pts, 3))
                colors[:, 0] = min(initialize_param["floor_init_color"][0] / 255.0, 1.0)
                colors[:, 1] = min(initialize_param["floor_init_color"][1] / 255.0, 1.0)
                colors[:, 2] = min(initialize_param["floor_init_color"][2] / 255.0, 1.0)
                pcd = BasicPointCloud(points=ground_xyz, colors=colors, normals=np.zeros((ground_pts, 3)))
            self.create_from_pcd(pcd, False, 1)
        else:
            pass

    def init_pcd(self, initialize_param: dict, exp_path: str):
        guided = initialize_param["init_guided"]
        init_ply_name = hash_prompt(guided, initialize_param["init_prompt"]) + "_init_points3d.ply"
        # load checkpoint
        ply_path = os.path.join(exp_path, init_ply_name)
        if guided == "shapes":
            if not os.path.exists(ply_path):
                num_pts = 50000
                mesh = o3d.io.read_triangle_mesh(initialize_param["init_prompt"])
                point_cloud = mesh.sample_points_uniformly(number_of_points=num_pts)
                coords = np.array(point_cloud.points)
                shs = np.random.random((num_pts, 3)) / 255.0
                rgb = SH2RGB(shs)
                adjusment = np.zeros_like(coords)
                adjusment[:, 1] = coords[:, 2]
                adjusment[:, 0] = coords[:, 0]
                adjusment[:, 2] = coords[:, 1]
                current_center = np.mean(adjusment, axis=0)
                center_offset = -current_center
                adjusment += center_offset
                adjusment /= 80.0
                pcd = BasicPointCloud(points=adjusment, colors=rgb, normals=np.zeros((num_pts, 3)))
                storePly(ply_path, adjusment, SH2RGB(shs) * 255)
            else:
                try:
                    pcd = fetchPly(ply_path)
                except Exception:
                    pcd = None
            self.create_from_pcd(pcd, True, 1)

        elif guided == "default":
            if not os.path.exists(ply_path):
                num_pts = initialize_param["num_pts"]
                radius = initialize_param["radius"]
                # init from random point cloud

                phis = np.random.random((num_pts,)) * 2 * np.pi
                costheta = np.random.random((num_pts,)) * 2 - 1
                thetas = np.arccos(costheta)
                mu = np.random.random((num_pts,))
                radius = radius * np.cbrt(mu)
                x = radius * np.sin(thetas) * np.cos(phis)
                y = radius * np.sin(thetas) * np.sin(phis)
                z = radius * np.cos(thetas)
                xyz = np.stack((x, y, z), axis=1)
                shs = np.random.random((num_pts, 3)) / 255.0
                pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
                storePly(ply_path, xyz, SH2RGB(shs) * 255)
            else:
                try:
                    pcd = fetchPly(ply_path)
                except Exception:
                    pcd = None
            self.create_from_pcd(pcd, True, 10)

        elif guided == "pointe" or guided == "pointe_330k" or guided == "pointe_825k":
            if not os.path.exists(ply_path):
                num_pts = 100000
                num_pts = int(num_pts / 5000)
                xyz, rgb = init_from_pointe(initialize_param["init_prompt"], guided)
                xyz[:, 1] = -xyz[:, 1]
                xyz[:, 2] = xyz[:, 2] + 0.15
                thetas = np.random.rand(num_pts) * np.pi
                phis = np.random.rand(num_pts) * 2 * np.pi
                radius = np.random.rand(num_pts) * 0.05
                xyz_ball = np.stack(
                    [
                        radius * np.sin(thetas) * np.sin(phis),
                        radius * np.sin(thetas) * np.cos(phis),
                        radius * np.cos(thetas),
                    ],
                    axis=-1,
                )  # [B, 3]expend_dims
                rgb_ball = np.random.random((4096, num_pts, 3)) * 0.0001
                rgb = (np.expand_dims(rgb, axis=1) + rgb_ball).reshape(-1, 3)
                xyz = (np.expand_dims(xyz, axis=1) + np.expand_dims(xyz_ball, axis=0)).reshape(-1, 3)
                xyz = xyz * 1.0
                num_pts = xyz.shape[0]
                shs = np.random.random((num_pts, 3)) / 255.0
                pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
                if initialize_param["use_pointe_rgb"]:
                    pcd = BasicPointCloud(points=xyz, colors=rgb, normals=np.zeros((num_pts, 3)))
                    storePly(ply_path, xyz, rgb * 255)
                else:
                    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
                    storePly(ply_path, xyz, SH2RGB(shs) * 255)
            else:
                try:
                    pcd = fetchPly(ply_path)
                except Exception:
                    pcd = None
            self.create_from_pcd(pcd, True, 1)

        elif guided == "obj":
            pass
            self.create_from_pcd(pcd, True, 1)

        elif isinstance(guided, BasicPointCloud):
            # load from a provided pcd
            self.create_from_pcd(guided, True, 1)

        else:
            # load from saved ply
            self.load_ply(guided)

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_background(self):
        return torch.sigmoid(self._background)

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @torch.no_grad()
    def extract_fields(self, resolution=128, num_blocks=16, relax_ratio=1.5):
        # resolution: resolution of field

        block_size = 2 / num_blocks

        assert resolution % block_size == 0
        split_size = resolution // num_blocks

        opacities = self.get_opacity

        # pre-filter low opacity gaussians to save computation
        mask = (opacities > 0.005).squeeze(1)

        opacities = opacities[mask]
        xyzs = self.get_xyz[mask]
        stds = self.get_scaling[mask]

        # normalize to ~ [-1, 1]
        mn, mx = xyzs.amin(0), xyzs.amax(0)
        self.center = (mn + mx) / 2
        self.scale = 1.8 / (mx - mn).amax().item()

        xyzs = (xyzs - self.center) * self.scale
        stds = stds * self.scale

        covs = self.covariance_activation(stds, 1, self._rotation[mask])

        # tile
        device = opacities.device
        occ = torch.zeros([resolution] * 3, dtype=torch.float32, device=device)

        X = torch.linspace(-1, 1, resolution).split(split_size)
        Y = torch.linspace(-1, 1, resolution).split(split_size)
        Z = torch.linspace(-1, 1, resolution).split(split_size)

        # loop blocks (assume max size of gaussian is small than relax_ratio * block_size !!!)
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    # sample points [M, 3]
                    pts = torch.cat(
                        [xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
                        dim=-1,
                    ).to(device)
                    # in-tile gaussians mask
                    vmin, vmax = pts.amin(0), pts.amax(0)
                    vmin -= block_size * relax_ratio
                    vmax += block_size * relax_ratio
                    mask = (xyzs < vmax).all(-1) & (xyzs > vmin).all(-1)
                    # if hit no gaussian, continue to next block
                    if not mask.any():
                        continue
                    mask_xyzs = xyzs[mask]  # [L, 3]
                    mask_covs = covs[mask]  # [L, 6]
                    mask_opas = opacities[mask].view(1, -1)  # [L, 1] --> [1, L]

                    # query per point-gaussian pair.
                    g_pts = pts.unsqueeze(1).repeat(1, mask_covs.shape[0], 1) - mask_xyzs.unsqueeze(0)  # [M, L, 3]
                    g_covs = mask_covs.unsqueeze(0).repeat(pts.shape[0], 1, 1)  # [M, L, 6]

                    # batch on gaussian to avoid OOM
                    batch_g = 1024
                    val = 0
                    for start in range(0, g_covs.shape[1], batch_g):
                        end = min(start + batch_g, g_covs.shape[1])
                        w = gaussian_3d_coeff(
                            g_pts[:, start:end].reshape(-1, 3),
                            g_covs[:, start:end].reshape(-1, 6),
                        ).reshape(
                            pts.shape[0], -1
                        )  # [M, l]
                        val += (mask_opas[:, start:end] * w).sum(-1)

                    # kiui.lo(val, mask_opas, w)

                    occ[
                        xi * split_size : xi * split_size + len(xs),
                        yi * split_size : yi * split_size + len(ys),
                        zi * split_size : zi * split_size + len(zs),
                    ] = val.reshape(len(xs), len(ys), len(zs))

        return occ

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, require_grad=True, spatial_lr_scale: float = 1):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(require_grad))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(require_grad))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(require_grad))

        self._scaling = nn.Parameter(scales.requires_grad_(require_grad))
        self._rotation = nn.Parameter(rots.requires_grad_(require_grad))
        self._opacity = nn.Parameter(opacities.requires_grad_(require_grad))
        self._background = nn.Parameter(torch.zeros((3, 1, 1), device="cuda").requires_grad_(require_grad))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
            {
                "params": [self._background],
                "lr": training_args.feature_lr,
                "name": "background",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.iterations,
        )
        self.rotation_scheduler_args = get_expon_lr_func(
            lr_init=training_args.rotation_lr,
            lr_final=training_args.rotation_lr_final,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.iterations,
        )

        self.scaling_scheduler_args = get_expon_lr_func(
            lr_init=training_args.scaling_lr,
            lr_final=training_args.scaling_lr_final,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.iterations,
        )

        self.feature_scheduler_args = get_expon_lr_func(
            lr_init=training_args.feature_lr,
            lr_final=training_args.feature_lr_final,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.iterations,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def update_feature_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "f_dc":
                lr = self.feature_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def update_rotation_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "rotation":
                lr = self.rotation_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def update_scaling_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "scaling":
                lr = self.scaling_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def active_grad(self):
        self._xyz.requires_grad = True
        self._features_dc.requires_grad = True
        self._features_rest.requires_grad = True
        self._opacity.requires_grad = True
        self._scaling.requires_grad = True
        self._rotation.requires_grad = True
        self._background.requires_grad = True

    def deactive_grad(self):
        self._xyz.requires_grad = False
        self._features_dc.requires_grad = False
        self._features_rest.requires_grad = False
        self._opacity.requires_grad = False
        self._scaling.requires_grad = False
        self._rotation.requires_grad = False
        self._background.requires_grad = False

    def load_ply(self, path, require_grad=True):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        logger.debug("Loaded {} points.".format(xyz.shape[0]))

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        if len(extra_f_names) != 3 * (self.max_sh_degree + 1) ** 2 - 3:
            logger.debug(
                "Update max_sh_degree, {} => {}.".format(len(extra_f_names), 3 * (self.max_sh_degree + 1) ** 2 - 3)
            )
        features_extra = np.zeros((xyz.shape[0], 3 * (self.max_sh_degree + 1) ** 2 - 3))
        for idx, attr_name in enumerate(extra_f_names):
            if idx == 3 * (self.max_sh_degree + 1) ** 2 - 3:
                break
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        if require_grad:
            self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
            self._features_dc = nn.Parameter(
                torch.tensor(features_dc, dtype=torch.float, device="cuda")
                .transpose(1, 2)
                .contiguous()
                .requires_grad_(True)
            )
            self._features_rest = nn.Parameter(
                torch.tensor(features_extra, dtype=torch.float, device="cuda")
                .transpose(1, 2)
                .contiguous()
                .requires_grad_(True)
            )
            self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
            self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
            self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
            self._background = nn.Parameter(torch.zeros((3, 1, 1), device="cuda").requires_grad_(True))
        else:
            self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(False))
            self._features_dc = nn.Parameter(
                torch.tensor(features_dc, dtype=torch.float, device="cuda")
                .transpose(1, 2)
                .contiguous()
                .requires_grad_(False)
            )
            self._features_rest = nn.Parameter(
                torch.tensor(features_extra, dtype=torch.float, device="cuda")
                .transpose(1, 2)
                .contiguous()
                .requires_grad_(False)
            )
            self._opacity = nn.Parameter(
                torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False)
            )
            self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(False))
            self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))
            self._background = nn.Parameter(torch.zeros((3, 1, 1), device="cuda").requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] not in ["background"]:
                if group["name"] == name:
                    stored_state = self.optimizer.state.get(group["params"][0], None)
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if group["name"] not in ["background"]:
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] not in ["background"]:
                assert len(group["params"]) == 1
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group["params"][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat(
                        (stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                        dim=0,
                    )
                    stored_state["exp_avg_sq"] = torch.cat(
                        (
                            stored_state["exp_avg_sq"],
                            torch.zeros_like(extension_tensor),
                        ),
                        dim=0,
                    )

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                    )
                    self.optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                    )
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def prune(self, min_opacity, extent, max_screen_size):

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def add_densification_stats_div(self, viewspace_point_tensor, update_filter, cut_start, cut_end=None):
        if cut_end:
            self.xyz_gradient_accum[update_filter] += torch.norm(
                viewspace_point_tensor.grad[cut_start:cut_end][update_filter, :2],
                dim=-1,
                keepdim=True,
            )
        else:
            self.xyz_gradient_accum[update_filter] += torch.norm(
                viewspace_point_tensor.grad[cut_start:][update_filter, :2],
                dim=-1,
                keepdim=True,
            )
        self.denom[update_filter] += 1

    def prune_gaussians(self, percent: float, important_score: list) -> None:
        sorted_tensor, _ = torch.sort(important_score, dim=0)
        index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
        value_nth_percentile = sorted_tensor[index_nth_percentile]
        prune_mask = (important_score <= value_nth_percentile).squeeze()
        self.prune_points(prune_mask)

    def free_memory(
        self,
    ):
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._background = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        return

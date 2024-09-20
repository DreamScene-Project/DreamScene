import random
from typing import List, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from config import GenerateCamParams
from utils.graphics_utils import (focal2fov, fov2focal, get_rays_torch,
                                  getProjectionMatrix, getWorld2View2)
from utils.system_utils import safe_normalize


def dot(x, y):
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)

def length(x, eps=1e-20):
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))

def look_at(campos, target, opengl=True):
    # campos: [N, 3], camera/eye position
    # target: [N, 3], object to look at
    # return: [N, 3, 3], rotation matrix
    if not opengl:
        # camera forward aligns with -z
        forward_vector = safe_normalize(target - campos)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(forward_vector, up_vector))
        up_vector = safe_normalize(np.cross(right_vector, forward_vector))
    else:
        # camera forward aligns with +z
        forward_vector = safe_normalize(campos - target)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(up_vector, forward_vector))
        up_vector = safe_normalize(np.cross(forward_vector, right_vector))
    R = np.stack([right_vector, up_vector, forward_vector], axis=1)
    return R

# ['front', 'side', 'back', 'overhead', 'bottom', 'zoom in']
def _get_dir_ind(
    thetas,
    phis,
    radius,
    overhead_threshold=30,
    front_threshold=75,
    ZOOM_IN_THRESH=1.1,
    distinguish_lr=False,
):
    #                   phis [B,];          thetas: [B,]
    # front = 0         [0, front/2)
    # side (left) = 1   [front/2, 180-front/2)
    # back = 2          [180-front/2, 180+front/2)
    # side (right) = 3  [180+front/2, 360-front/2)
    # front = 0         [360-front/2, 360]
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    # zoom in = 6                               radius < 1.1

    if distinguish_lr:
        res = np.zeros(1, dtype=np.int64)
        if (phis >= -(front_threshold / 2)) and (phis < (front_threshold / 2)):
            # front: [-front_threshold/2, front_threshold/2]
            res[0] = 0
        if (phis < -(front_threshold / 2)) and (phis >= -180 + (front_threshold / 2)):
            # left:[-180+front_threshold/2,-front_threshold/2]
            res[0] = 1
        if (phis < -180 + (front_threshold / 2)) or (
            phis >= 180 - (front_threshold / 2)
        ):
            # back:[-180, -180+front_threshold/2] U [180-front_threshold/2, 180]
            res[0] = 2
        if (phis >= (front_threshold / 2)) and (phis < 180 - (front_threshold / 2)):
            # right:[front_threshold/2, 180-front_threshold/2]
            res[0] = 3

        if thetas < (-90 + overhead_threshold):
            # top:[-90,-90+overhead_threshold]
            res[0] = 4
        if thetas >= (90 - overhead_threshold):
            # bottom:[90-overhead_threshold,90]
            res[0] = 5
        # ZOOM IN
        # res[radius <= ZOOM_IN_THRESH] = 6
        ref_l = ["front", "side", "back", "side", "overhead", "bottom", "zoom in"]
    else:
        # thetas = elevation[-90, 90] + 90 -> [0, 180]
        # phis = azimuth[-180,180] + 180 -> [0,360]
        thetas = thetas + 90
        phis = phis + 180
        thetas = np.deg2rad(thetas)
        phis = np.deg2rad(phis)
        overhead_threshold = np.deg2rad(overhead_threshold)
        front_threshold = np.deg2rad(front_threshold)

        res = np.zeros(
            1, dtype=np.int64
        )  # res = torch.zeros(thetas.shape[0], dtype=torch.long)
        # first determine by phis

        # res[(phis < front)] = 0
        res[
            (phis >= (2 * np.pi - front_threshold / 2)) & (phis < front_threshold / 2)
        ] = 0  #'front'

        # res[(phis >= front) & (phis < np.pi)] = 1
        res[(phis >= front_threshold / 2) & (phis < (np.pi - front_threshold / 2))] = (
            1  #'side'
        )

        # res[(phis >= np.pi) & (phis < (np.pi + front))] = 2
        res[
            (phis >= (np.pi - front_threshold / 2))
            & (phis < (np.pi + front_threshold / 2))
        ] = 2  #'back'

        # res[(phis >= (np.pi + front))] = 3
        res[
            (phis >= (np.pi + front_threshold / 2))
            & (phis < (2 * np.pi - front_threshold / 2))
        ] = 3  #'side'
        # override by thetas
        res[thetas <= overhead_threshold] = 4  #'top'
        res[thetas >= (np.pi - overhead_threshold)] = 5  #'bottom'
        # override by radius
        res[radius <= ZOOM_IN_THRESH] = 6  #'zoom in'
        ref_l = ["front", "side", "back", "side", "overhead", "bottom", "zoom in"]
    return ref_l[res[0]]

class RandCameraInfo(NamedTuple):
    # uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    width: int
    height: int
    delta_polar: np.array
    delta_azimuth: np.array
    delta_radius: np.array

class RCamera(nn.Module):
    def __init__(
        self,
        R,
        T,
        FoVx,
        FoVy,
        delta_polar,
        delta_azimuth,
        delta_radius,
        opt,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda",
        SSAA=False,
    ):
        super(RCamera, self).__init__()

        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.delta_polar = delta_polar
        self.delta_azimuth = delta_azimuth
        self.delta_radius = delta_radius
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(
                f"[Warning] Custom device {data_device} failed, fallback to default cuda device"
            )
            self.data_device = torch.device("cuda")

        self.zfar = 100.0
        self.znear = 0.01

        if SSAA:
            ssaa = opt.SSAA
        else:
            ssaa = 1

        self.image_width = opt.image_w * ssaa
        self.image_height = opt.image_h * ssaa

        self.trans = trans
        self.scale = scale

        RT = torch.tensor(getWorld2View2(R, T, trans, scale))
        self.world_view_transform = RT.transpose(0, 1).cuda()
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        # self.rays = get_rays_torch(fov2focal(FoVx, 64), RT).cuda()
        self.rays = get_rays_torch(
            fov2focal(FoVx, self.image_width // 8),
            RT,
            H=self.image_height // 8,
            W=self.image_width // 8,
        ).cuda()

def sample_jit(phi, jit_size, range_max, range_size, islist=False):
    if islist:
        for i in range(len(phi)):
            phi[i] = sample_jit(phi[i], jit_size, range_max, range_size)
        return phi
    phi += jit_size * random.random()
    if phi > range_max:
        phi -= range_size
    return phi

def gen_random_pos(param_range, gamma=1):
    lower, higher = param_range[0], param_range[1]

    mid = lower + (higher - lower) * 0.5
    radius = (higher - lower) * 0.5

    rand_ = torch.rand(1)  # 0, 1
    sign = torch.where(torch.rand(1) > 0.5, torch.ones(1) * -1.0, torch.ones(1))
    rand_ = sign * (rand_**gamma)

    return (rand_ * radius) + mid

def calc_radius(bbox, dim=2, sqrt=False):
    if dim == 2:
        if sqrt:
            # 2D circumscribed radius
            return np.sqrt(np.sum(np.max([bbox[:2], bbox[3:5]], axis=0) ** 2))
        else:
            # 2D inscribed radius
            return torch.min(torch.abs(torch.cat([bbox[0:2], bbox[3:5]])))
    elif dim == 3:
        if sqrt:
            # 3D circumscribed radius
            return np.sqrt(np.sum(np.max([bbox[:3], bbox[3:]], axis=0) ** 2))
        else:
            raise KeyError

def distance_point_to_aabb(point, min_point, max_point):
    if isinstance(point, torch.Tensor):
        while point.dim() > 1:
            point = point[0]
        tmp_t = torch.min(
            torch.stack((max_point[:2] - point[:2], point[:2] - min_point[:2]), dim=0),
            dim=0,
        ).values
    else:
        tmp_t = torch.min(
            torch.stack(
                (
                    max_point[:2] - torch.from_numpy(point[:2]),
                    torch.from_numpy(point[:2]) - min_point[:2],
                ),
                dim=0,
            ),
            dim=0,
        ).values
    return torch.min(tmp_t).item()

def circle_poses(
    radius=torch.tensor([3.2]),
    theta=torch.tensor([60]),
    phi=torch.tensor([0]),
    angle_overhead=30,
    angle_front=60,
):

    theta = theta / 180 * np.pi
    phi = phi / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    centers = torch.stack(
        [
            radius * torch.sin(theta) * torch.sin(phi),
            radius * torch.sin(theta) * torch.cos(phi),
            radius * torch.cos(theta),
        ],
        dim=-1,
    )  # [B, 3]

    # lookat
    forward_vector = safe_normalize(centers)
    up_vector = torch.FloatTensor([0, 0, 1]).unsqueeze(0).repeat(len(centers), 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float).unsqueeze(0).repeat(len(centers), 1, 1)
    poses[:, :3, :3] = torch.stack((-right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses.numpy()

def viewpoint_in_scene(center, scene_box, objects_args, object_collision=False):
    if torch.all(torch.gt(center, scene_box[:3])) and torch.all(
        torch.gt(scene_box[3:], center)
    ):
        if object_collision:
            for object_args in objects_args:
                obox = object_args.bbox.clone()
                if torch.all(torch.gt(center, obox[:3])) and torch.all(
                    torch.gt(obox[3:], center)
                ):
                    return 2
        return 1
    else:
        return 0

def gen_random_delta(
    trans,
    scale,
    theta_range,
    phi_range,
    radius_range,
    scene_box,
    uniform_sphere_rate,
    rand_cam_gamma,
    objects_args,
    cam_pose_method,
    get_cam_outview=False,
    colli=True,
    radius_trans_max=3,
):
    radius = gen_random_pos(radius_range)
    if random.random() < uniform_sphere_rate:
        unit_centers = F.normalize(
            torch.stack(
                [
                    torch.randn(1),
                    torch.abs(torch.randn(1)),
                    torch.randn(1),
                ],
                dim=-1,
            ),
            p=2,
            dim=1,
        )
        thetas = torch.acos(unit_centers[:, 1])
        phis = torch.atan2(unit_centers[:, 0], unit_centers[:, 2])
        phis[phis < 0] += 2 * np.pi
        centers = unit_centers * radius.unsqueeze(-1)
    else:
        thetas = gen_random_pos(theta_range, rand_cam_gamma)
        phis = gen_random_pos(phi_range, rand_cam_gamma)
        phis[phis < 0] += 2 * np.pi

        centers_delta = torch.stack(
            [
                radius * torch.sin(thetas) * torch.sin(phis),
                radius * torch.sin(thetas) * torch.cos(phis),
                radius * torch.cos(thetas),
            ],
            dim=-1,
        )  # [B, 3]

    targets = trans.clone().unsqueeze(0)
    if get_cam_outview:
        centers_delta_outview = centers_delta.clone()
        centers_delta_outview[:2] *= -1
        centers = centers_delta_outview * scale + targets
    else:
        centers = centers_delta * scale + targets

    aabb_check = viewpoint_in_scene(centers, scene_box, objects_args, colli)
    if cam_pose_method == "indoor":
        if aabb_check == 1:
            return centers, targets, centers_delta, phis, thetas, radius, scale
        else:
            if (
                aabb_check == 2
                and distance_point_to_aabb(centers, scene_box[:3], scene_box[3:])
                < radius_trans_max * 0.75
            ):
                factor = 1.02
            else:
                factor = 0.98
            if np.abs(scale) > 3:
                print(
                    aabb_check,
                    phi_range,
                    theta_range,
                    radius_range,
                    radius,
                    centers,
                    centers_delta,
                    scale,
                    targets,
                    scene_box,
                )
                raise Exception
            elif np.abs(scale) < 0.3:
                print(
                    aabb_check,
                    phi_range,
                    theta_range,
                    radius_range,
                    radius,
                    centers,
                    centers_delta,
                    scale,
                    targets,
                    scene_box,
                )
                raise Exception
            return gen_random_delta(
                trans,
                scale * factor,
                theta_range,
                phi_range,
                radius_range,
                scene_box,
                uniform_sphere_rate,
                rand_cam_gamma,
                objects_args,
                cam_pose_method,
                get_cam_outview,
                colli,
                radius_trans_max,
            )
    elif cam_pose_method == "outdoor":
        if aabb_check == 1:
            return centers, targets, centers_delta, phis, thetas, radius, scale
        else:
            if (
                aabb_check == 2
                and distance_point_to_aabb(centers, scene_box[:3], scene_box[3:])
                < radius_trans_max * 0.75
            ):
                factor = 1.02
            else:
                factor = 0.98
            if np.abs(scale) > 3:
                print(
                    aabb_check,
                    phi_range,
                    theta_range,
                    radius_range,
                    radius,
                    centers,
                    centers_delta,
                    scale,
                    targets,
                    scene_box,
                )
                raise Exception
            elif np.abs(scale) < 0.3:
                print(
                    aabb_check,
                    phi_range,
                    theta_range,
                    radius_range,
                    radius,
                    centers,
                    centers_delta,
                    scale,
                    targets,
                    scene_box,
                )
                raise Exception
            return gen_random_delta(
                trans,
                scale * 0.98,
                theta_range,
                phi_range,
                radius_range,
                scene_box,
                uniform_sphere_rate,
                rand_cam_gamma,
                objects_args,
                cam_pose_method,
                get_cam_outview,
                colli,
                radius_trans_max,
            )
    else:
        return centers, targets, centers_delta, phis, thetas, radius, scale

def scene_poses(
    opt,
    trans,
    scale,
    scene_box,
    objects_args,
    cam_pose_method,
    radius_range=[1, 1.5],
    theta_range=[0, 120],
    phi_range=[0, 360],
    angle_overhead=30,
    angle_front=60,
    uniform_sphere_rate=0.5,
    rand_cam_gamma=1,
    get_cam_outview_ratio=0.0,
    colli=True,
):
    theta_range = np.array(theta_range) / 180 * np.pi
    phi_range = np.array(phi_range) / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi
    get_cam_outview = False
    if random.random() < get_cam_outview_ratio:
        get_cam_outview = True
    if get_cam_outview:
        factor = 1.3
        radius_range[1] = min(radius_range[1], 3.0)
        radius_range[0] = min(radius_range[1], radius_range[0])
    else:
        factor = 0.8
        radius_range[0] = max(radius_range[0], 3.0)
        radius_range[1] = max(radius_range[0], radius_range[1])
    for i in range(len(radius_range)):
        radius_range[i] *= factor
    radius_trans_max = min(
        np.abs(scene_box[0]), np.abs(-scene_box[1]), scene_box[3], scene_box[4]
    )
    centers, targets, centers_delta, phis, thetas, radius, scale = gen_random_delta(
        trans,
        scale,
        theta_range,
        phi_range,
        radius_range,
        scene_box,
        uniform_sphere_rate,
        rand_cam_gamma,
        objects_args,
        cam_pose_method,
        get_cam_outview,
        colli,
        radius_trans_max,
    )

    # jitters
    if opt.jitter_pose:
        jit_center = opt.jitter_center
        jit_target = opt.jitter_target
        centers_delta += torch.rand_like(centers_delta) * jit_center - jit_center / 2.0
        targets += torch.randn_like(centers_delta) * jit_target

    # lookat
    # centers = centers_delta + targets
    forward_vector = safe_normalize(centers_delta)
    up_vector = torch.FloatTensor([0, 0, 1]).unsqueeze(0)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))

    if opt.jitter_pose:
        up_noise = torch.randn_like(up_vector) * opt.jitter_up
    else:
        up_noise = 0

    up_vector = safe_normalize(
        torch.cross(right_vector, forward_vector, dim=-1) + up_noise
    )  # forward_vector

    poses = torch.eye(4, dtype=torch.float)
    poses[:3, :3] = torch.stack(
        (-right_vector, up_vector, forward_vector), dim=-1
    )  # up_vector
    poses[:3, 3] = centers_delta
    if get_cam_outview:
        poses[:2, 3] *= -1

    # back to degree
    thetas = thetas / np.pi * 180
    phis = phis / np.pi * 180

    return poses.numpy(), thetas.numpy(), phis.numpy(), radius.numpy(), scale

def scene_circle_poses(
    trans,
    scale,
    scene_box,
    objects_args,
    cam_pose_method,
    radius=torch.tensor([3.2]),
    theta=torch.tensor([60]),
    phi=torch.tensor([0]),
    angle_overhead=30,
    angle_front=60,
):

    theta = theta / 180 * np.pi
    phi = phi / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    centers_delta = torch.stack(
        [
            radius * torch.sin(theta) * torch.sin(phi),
            radius * torch.sin(theta) * torch.cos(phi),
            radius * torch.cos(theta),
        ],
        dim=-1,
    )  # [B, 3]

    centers = centers_delta * scale + trans
    if viewpoint_in_scene(centers, scene_box, objects_args, True) != 1:
        return False, None
    # lookat
    forward_vector = safe_normalize(centers_delta)
    up_vector = torch.FloatTensor([0, 0, 1]).unsqueeze(0).repeat(len(centers_delta), 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = (
        torch.eye(4, dtype=torch.float).unsqueeze(0).repeat(len(centers_delta), 1, 1)
    )
    poses[:, :3, :3] = torch.stack((-right_vector, up_vector, forward_vector), dim=-1)

    poses[:, :3, 3] = centers_delta

    return True, poses.numpy()

def rand_poses(
    opt,
    radius_range=[1, 1.5],
    theta_range=[0, 120],
    phi_range=[0, 360],
    angle_overhead=30,
    angle_front=60,
    uniform_sphere_rate=0.5,
    rand_cam_gamma=1,
):
    theta_range = np.array(theta_range) / 180 * np.pi
    phi_range = np.array(phi_range) / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    radius = gen_random_pos(radius_range)

    if random.random() < uniform_sphere_rate:
        unit_centers = F.normalize(
            torch.stack(
                [
                    torch.randn(1),
                    torch.abs(torch.randn(1)),
                    torch.randn(1),
                ],
                dim=-1,
            ),
            p=2,
            dim=1,
        )
        thetas = torch.acos(unit_centers[:, 1])
        phis = torch.atan2(unit_centers[:, 0], unit_centers[:, 2])
        phis[phis < 0] += 2 * np.pi
        centers = unit_centers * radius.unsqueeze(-1)
    else:
        thetas = gen_random_pos(theta_range, rand_cam_gamma)
        phis = gen_random_pos(phi_range, rand_cam_gamma)
        phis[phis < 0] += 2 * np.pi

        centers = torch.stack(
            [
                radius * torch.sin(thetas) * torch.sin(phis),
                radius * torch.sin(thetas) * torch.cos(phis),
                radius * torch.cos(thetas),
            ],
            dim=-1,
        )  # [B, 3]

    targets = 0

    # jitters
    if opt.jitter_pose:
        jit_center = opt.jitter_center  # 0.015 # was 0.2
        jit_target = opt.jitter_target
        centers += torch.rand_like(centers) * jit_center - jit_center / 2.0
        targets += torch.randn_like(centers) * jit_target

    # lookat
    forward_vector = safe_normalize(centers - targets)
    up_vector = torch.FloatTensor([0, 0, 1]).unsqueeze(0)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))

    if opt.jitter_pose:
        up_noise = torch.randn_like(up_vector) * opt.jitter_up
    else:
        up_noise = 0

    up_vector = safe_normalize(
        torch.cross(right_vector, forward_vector, dim=-1) + up_noise
    )  # forward_vector

    poses = torch.eye(4, dtype=torch.float)
    poses[:3, :3] = torch.stack(
        (-right_vector, up_vector, forward_vector), dim=-1
    )  # up_vector
    poses[:3, 3] = centers

    # back to degree
    thetas = thetas / np.pi * 180
    phis = phis / np.pi * 180

    return poses.numpy(), thetas.numpy(), phis.numpy(), radius.numpy()

def GenerateRandomCamerasAvoidMultiFace(opt, step_ratio, SSAA=True, dirs="random"):
    # random pose on the fly
    radius_range_ = opt.radius_range
    if dirs == "random":
        if step_ratio < 0.1:
            rrc = random.random()
            if rrc > 0.5:
                phi_range_ = [-30, 30]
            elif rrc > 0.75:
                phi_range_ = [-180, -150]
            else:
                phi_range_ = [150, 180]
        else:
            phi_range_ = opt.phi_range  # [-180, 180]
    elif dirs == "front":
        phi_range_ = [-32.5, 32.5]
    elif dirs == "side":
        rrc = random.random()
        if rrc > 0.5:
            phi_range_ = [-147.5, -32.5]
        else:
            phi_range_ = [32.5, 147.5]
    elif dirs == "back":
        rrc = random.random()
        if rrc > 0.5:
            phi_range_ = [-180, -147.5]
        else:
            phi_range_ = [147.5, 180]
    theta_range_ = opt.theta_range
    poses, thetas, phis, radius = rand_poses(
        opt,
        radius_range=radius_range_,
        theta_range=theta_range_,
        phi_range=phi_range_,
        angle_overhead=opt.angle_overhead,
        angle_front=opt.angle_front,
        uniform_sphere_rate=opt.uniform_sphere_rate,
        rand_cam_gamma=opt.rand_cam_gamma,
    )
    # delta polar/azimuth/radius to default view
    delta_polar = thetas - opt.default_polar
    delta_azimuth = phis - opt.default_azimuth
    delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
    delta_radius = radius - opt.default_radius
    # random focal
    fov = random.random() * (opt.fovy_range[1] - opt.fovy_range[0]) + opt.fovy_range[0]

    if SSAA:
        ssaa = opt.SSAA
    else:
        ssaa = 1

    image_h = opt.image_h * ssaa
    image_w = opt.image_w * ssaa

    # generate specific data structure
    matrix = np.linalg.inv(poses)
    R = -np.transpose(matrix[:3, :3])
    R[:, 0] = -R[:, 0]
    T = -matrix[:3, 3]
    fovy = focal2fov(fov2focal(fov, image_h), image_w)
    FovY = fovy
    FovX = fov

    return RandCameraInfo(
        R=R,
        T=T,
        FovY=FovY,
        FovX=FovX,
        width=image_w,
        height=image_h,
        delta_polar=delta_polar,
        delta_azimuth=delta_azimuth,
        delta_radius=delta_radius,
    )

def GenerateRandomCameras(opt, SSAA=True):
    # random pose on the fly
    radius_range_ = opt.radius_range
    phi_range_ = opt.phi_range
    theta_range_ = opt.theta_range
    poses, thetas, phis, radius = rand_poses(
        opt,
        radius_range=radius_range_,
        theta_range=theta_range_,
        phi_range=phi_range_,
        angle_overhead=opt.angle_overhead,
        angle_front=opt.angle_front,
        uniform_sphere_rate=opt.uniform_sphere_rate,
        rand_cam_gamma=opt.rand_cam_gamma,
    )
    # delta polar/azimuth/radius to default view
    delta_polar = thetas - opt.default_polar
    delta_azimuth = phis - opt.default_azimuth
    delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
    delta_radius = radius - opt.default_radius
    # random focal
    fov = random.random() * (opt.fovy_range[1] - opt.fovy_range[0]) + opt.fovy_range[0]

    if SSAA:
        ssaa = opt.SSAA
    else:
        ssaa = 1

    image_h = opt.image_h * ssaa
    image_w = opt.image_w * ssaa

    # generate specific data structure
    matrix = np.linalg.inv(poses)
    R = -np.transpose(matrix[:3, :3])
    R[:, 0] = -R[:, 0]
    T = -matrix[:3, 3]
    fovy = focal2fov(fov2focal(fov, image_h), image_w)
    FovY = fovy
    FovX = fov

    return RandCameraInfo(
        R=R,
        T=T,
        FovY=FovY,
        FovX=FovX,
        width=image_w,
        height=image_h,
        delta_polar=delta_polar,
        delta_azimuth=delta_azimuth,
        delta_radius=delta_radius,
    )

def GenerateCamerasSceneOutdoor1(
    opt,
    trans,
    scale,
    scene_box,
    objects_args,
    cam_pose_method,
    phi_range_=None,
    SSAA=True,
):
    if phi_range_ is None:
        phi_range_ = opt.phi_range
    radius_range_ = [0.1, 0.5]
    theta_range_ = [80, 110]  # opt.theta_range
    poses, thetas, phis, radius, scale = scene_poses(
        opt,
        trans,
        scale,
        scene_box,
        objects_args,
        cam_pose_method,
        radius_range=radius_range_,
        theta_range=theta_range_,
        phi_range=phi_range_,
        angle_overhead=opt.angle_overhead,
        angle_front=opt.angle_front,
        uniform_sphere_rate=opt.uniform_sphere_rate,
        rand_cam_gamma=opt.rand_cam_gamma,
        get_cam_outview_ratio=0.5,
        colli=False,
    )

    # delta polar/azimuth/radius to default view
    delta_polar = thetas - opt.default_polar
    delta_azimuth = phis - opt.default_azimuth
    delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
    delta_radius = radius - opt.default_radius
    fov = 0.96

    if SSAA:
        ssaa = opt.SSAA
    else:
        ssaa = 1

    image_h = opt.image_h * ssaa
    image_w = opt.image_w * ssaa

    # generate specific data structure
    matrix = np.linalg.inv(poses)
    R = -np.transpose(matrix[:3, :3])
    R[:, 0] = -R[:, 0]
    T = -matrix[:3, 3]

    fovy = focal2fov(fov2focal(fov, image_h), image_w)
    FovY = fovy
    FovX = fov

    return RandCameraInfo(
        R=R,
        T=T,
        FovY=FovY,
        FovX=FovX,
        width=image_w,
        height=image_h,
        delta_polar=delta_polar,
        delta_azimuth=delta_azimuth,
        delta_radius=delta_radius,
    )

def GenerateCamerasSceneIndoor1(
    opt,
    trans,
    scale,
    scene_box,
    objects_args,
    cam_pose_method,
    phi_range_=None,
    view_floor=False,
    SSAA=True,
):
    radius_trans_max = min(
        np.abs(scene_box[0]), np.abs(-scene_box[1]), scene_box[3], scene_box[4]
    )
    if phi_range_ is None:
        phi_range_ = opt.phi_range
    radius_range_ = [radius_trans_max * 0.75, radius_trans_max * 1.1]
    theta_range_ = [75, 115]
    if view_floor:
        theta_range_ = [45, 90]
    poses, thetas, phis, radius, scale = scene_poses(
        opt,
        trans,
        scale,
        scene_box,
        objects_args,
        cam_pose_method,
        radius_range=radius_range_,
        theta_range=theta_range_,
        phi_range=phi_range_,
        angle_overhead=opt.angle_overhead,
        angle_front=opt.angle_front,
        uniform_sphere_rate=opt.uniform_sphere_rate,
        rand_cam_gamma=opt.rand_cam_gamma,
        get_cam_outview_ratio=0,
    )

    # delta polar/azimuth/radius to default view
    delta_polar = thetas - opt.default_polar
    delta_azimuth = phis - opt.default_azimuth
    delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
    delta_radius = radius - opt.default_radius

    fov = 0.96

    if SSAA:
        ssaa = opt.SSAA
    else:
        ssaa = 1

    image_h = opt.image_h * ssaa
    image_w = opt.image_w * ssaa

    # generate specific data structure
    matrix = np.linalg.inv(poses)
    R = -np.transpose(matrix[:3, :3])
    R[:, 0] = -R[:, 0]
    T = -matrix[:3, 3]
    fovy = focal2fov(fov2focal(fov, image_h), image_w)
    FovY = fovy
    FovX = fov

    return (
        RandCameraInfo(
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            width=image_w,
            height=image_h,
            delta_polar=delta_polar,
            delta_azimuth=delta_azimuth,
            delta_radius=delta_radius,
        ),
        scale,
    )

def GenerateCamerasSceneIndoor2(
    opt,
    trans,
    scale,
    scene_box,
    objects_args,
    cam_pose_method,
    phi_range_=None,
    max_radius=None,
    get_cam_outview_ratio=0,
    SSAA=True,
):
    if phi_range_ is None:
        phi_range_ = opt.phi_range

    radius_range_ = [0.1, 1.0]
    if max_radius is not None:
        radius_range_ = [3.0, max(max_radius, 3.0)]

    theta_range_ = [60, 110]

    poses, thetas, phis, radius, scale = scene_poses(
        opt,
        trans,
        scale,
        scene_box,
        objects_args,
        cam_pose_method,
        radius_range=radius_range_,
        theta_range=theta_range_,
        phi_range=phi_range_,
        angle_overhead=opt.angle_overhead,
        angle_front=opt.angle_front,
        uniform_sphere_rate=opt.uniform_sphere_rate,
        rand_cam_gamma=opt.rand_cam_gamma,
        get_cam_outview_ratio=get_cam_outview_ratio,
    )

    # delta polar/azimuth/radius to default view
    delta_polar = thetas - opt.default_polar
    delta_azimuth = phis - opt.default_azimuth
    delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
    delta_radius = radius - opt.default_radius
    fov = 0.96

    if SSAA:
        ssaa = opt.SSAA
    else:
        ssaa = 1

    image_h = opt.image_h * ssaa
    image_w = opt.image_w * ssaa

    # generate specific data structure
    matrix = np.linalg.inv(poses)
    R = -np.transpose(matrix[:3, :3])
    R[:, 0] = -R[:, 0]
    T = -matrix[:3, 3]
    fovy = focal2fov(fov2focal(fov, image_h), image_w)
    FovY = fovy
    FovX = fov

    return (
        RandCameraInfo(
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            width=image_w,
            height=image_h,
            delta_polar=delta_polar,
            delta_azimuth=delta_azimuth,
            delta_radius=delta_radius,
        ),
        scale,
    )

def GenerateCamerasSceneOutdoor2(
    opt,
    trans,
    scale,
    scene_box,
    objects_args,
    cam_pose_method,
    phi_range=None,
    get_cam_outview_ratio=0,
    SSAA=True,
):
    if cam_pose_method == "outdoor":
        phi_range_ = phi_range
        theta_range_ = [70, 100]  # opt.theta_range
        radius_range_ = [0.1, 1.1]
    elif cam_pose_method == "indoor":
        phi_range_ = opt.phi_range
        theta_range_ = opt.theta_range
        radius_range_ = [0.1, 2.1]

    poses, thetas, phis, radius, scale = scene_poses(
        opt,
        trans,
        scale,
        scene_box,
        objects_args,
        cam_pose_method,
        radius_range=radius_range_,
        theta_range=theta_range_,
        phi_range=phi_range_,
        angle_overhead=opt.angle_overhead,
        angle_front=opt.angle_front,
        uniform_sphere_rate=opt.uniform_sphere_rate,
        rand_cam_gamma=opt.rand_cam_gamma,
        get_cam_outview_ratio=get_cam_outview_ratio,
        colli=False,
    )

    # delta polar/azimuth/radius to default view
    delta_polar = thetas - opt.default_polar
    delta_azimuth = phis - opt.default_azimuth
    delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
    delta_radius = radius - opt.default_radius
    # random focal
    if cam_pose_method == "outdoor":
        fov = 0.96
    else:
        fov = (
            random.random() * (opt.fovy_range[1] - opt.fovy_range[0])
            + opt.fovy_range[0]
        )

    if SSAA:
        ssaa = opt.SSAA
    else:
        ssaa = 1

    image_h = opt.image_h * ssaa
    image_w = opt.image_w * ssaa

    # generate specific data structure
    matrix = np.linalg.inv(poses)
    R = -np.transpose(matrix[:3, :3])
    R[:, 0] = -R[:, 0]
    T = -matrix[:3, 3]
    fovy = focal2fov(fov2focal(fov, image_h), image_w)
    FovY = fovy
    FovX = fov

    return (
        RandCameraInfo(
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            width=image_w,
            height=image_h,
            delta_polar=delta_polar,
            delta_azimuth=delta_azimuth,
            delta_radius=delta_radius,
        ),
        scale,
    )

def GenerateCamerasSceneOutdoor3(
    opt,
    trans,
    scale,
    scene_box,
    objects_args,
    cam_pose_method,
    phi_range=None,
    get_cam_outview_ratio=0,
    SSAA=True,
):
    if cam_pose_method == "outdoor":
        phi_range_ = phi_range
        theta_range_ = [85, 95]
        radius_range_ = [0.1, 0.3]
        if scale > 0:
            theta_range_ = [90, 90]

    poses, thetas, phis, radius, scale = scene_poses(
        opt,
        trans,
        scale,
        scene_box,
        objects_args,
        cam_pose_method,
        radius_range=radius_range_,
        theta_range=theta_range_,
        phi_range=phi_range_,
        angle_overhead=opt.angle_overhead,
        angle_front=opt.angle_front,
        uniform_sphere_rate=opt.uniform_sphere_rate,
        rand_cam_gamma=opt.rand_cam_gamma,
        get_cam_outview_ratio=get_cam_outview_ratio,
        colli=False,
    )

    # delta polar/azimuth/radius to default view
    delta_polar = thetas - opt.default_polar
    delta_azimuth = phis - opt.default_azimuth
    delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
    delta_radius = radius - opt.default_radius
    # random focal
    if cam_pose_method == "outdoor":
        fov = random.random() * 0.48 + 0.96
    else:
        fov = (
            random.random() * (opt.fovy_range[1] - opt.fovy_range[0])
            + opt.fovy_range[0]
        )

    if SSAA:
        ssaa = opt.SSAA
    else:
        ssaa = 1

    image_h = opt.image_h * ssaa
    image_w = opt.image_w * ssaa

    # generate specific data structure
    matrix = np.linalg.inv(poses)
    R = -np.transpose(matrix[:3, :3])
    R[:, 0] = -R[:, 0]
    T = -matrix[:3, 3]
    fovy = focal2fov(fov2focal(fov, image_h), image_w)
    FovY = fovy
    FovX = fov

    return (
        RandCameraInfo(
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            width=image_w,
            height=image_h,
            delta_polar=delta_polar,
            delta_azimuth=delta_azimuth,
            delta_radius=delta_radius,
        ),
        scale,
    )

def GenerateCamerasSceneOuddoor4(
    opt,
    trans,
    scale,
    scene_box,
    objects_args,
    cam_pose_method,
    phi_range=None,
    opti_target="env",
    get_cam_outview_ratio=0,
    SSAA=True,
):
    if cam_pose_method == "outdoor":
        phi_range_ = phi_range
        if opti_target == "env":
            theta_range_ = [95, 95]  # opt.theta_range
            radius_range_ = [0.5, 0.5]
        elif opti_target == "env2":
            theta_range_ = [110, 110]  # opt.theta_range
            radius_range_ = [0.5, 0.5]
        elif opti_target == "floor":
            theta_range_ = [70, 70]
            radius_range_ = [0.5, 0.5]
        else:
            theta_range_ = [55, 55]
            radius_range_ = [0.5, 0.5]

    poses, thetas, phis, radius, scale = scene_poses(
        opt,
        trans,
        scale,
        scene_box,
        objects_args,
        cam_pose_method,
        radius_range=radius_range_,
        theta_range=theta_range_,
        phi_range=phi_range_,
        angle_overhead=opt.angle_overhead,
        angle_front=opt.angle_front,
        uniform_sphere_rate=opt.uniform_sphere_rate,
        rand_cam_gamma=opt.rand_cam_gamma,
        get_cam_outview_ratio=get_cam_outview_ratio,
        colli=False,
    )

    # delta polar/azimuth/radius to default view
    delta_polar = thetas - opt.default_polar
    delta_azimuth = phis - opt.default_azimuth
    delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
    delta_radius = radius - opt.default_radius
    # random focal
    if cam_pose_method == "outdoor":
        fov = 0.96
        if "floor" in opti_target:
            fov = 1.2
    else:
        fov = (
            random.random() * (opt.fovy_range[1] - opt.fovy_range[0])
            + opt.fovy_range[0]
        )

    if SSAA:
        ssaa = opt.SSAA
    else:
        ssaa = 1

    image_h = opt.image_h * ssaa
    image_w = opt.image_w * ssaa

    # generate specific data structure
    matrix = np.linalg.inv(poses)
    R = -np.transpose(matrix[:3, :3])
    R[:, 0] = -R[:, 0]
    T = -matrix[:3, 3]
    fovy = focal2fov(fov2focal(fov, image_h), image_w)
    FovY = fovy
    FovX = fov

    return (
        RandCameraInfo(
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            width=image_w,
            height=image_h,
            delta_polar=delta_polar,
            delta_azimuth=delta_azimuth,
            delta_radius=delta_radius,
        ),
        scale,
    )

def sphere_poses(size, radius=torch.tensor([3.2])):
    centers = torch.randn(size, 3)
    centers /= torch.norm(centers, dim=1).unsqueeze(1).repeat(1, 3)
    centers *= radius
    # lookat
    forward_vector = safe_normalize(centers)
    up_vector = torch.FloatTensor([0, 0, 1]).unsqueeze(0).repeat(len(centers), 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float).unsqueeze(0).repeat(len(centers), 1, 1)
    poses[:, :3, :3] = torch.stack((-right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses.numpy()

def GenerateSphereCameras(opt, size=8):
    # random focal
    fov = opt.default_fovy
    cam_infos = []
    radius = torch.FloatTensor([opt.default_radius])

    poses_sphere = sphere_poses(size, radius)
    for poses in poses_sphere:
        matrix = np.linalg.inv(poses)
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]
        fovy = focal2fov(fov2focal(fov, opt.image_h), opt.image_w)
        FovY = fovy
        FovX = fov
        cam_infos.append(
            RandCameraInfo(
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                width=opt.image_w,
                height=opt.image_h,
                delta_polar=0,
                delta_azimuth=0,
                delta_radius=0,
            )
        )
    return cam_infos

def GenerateRecoCameras(opt, circle_size, thetas, scale=1.0):
    fov = opt.default_fovy
    cam_infos = []
    radius = torch.FloatTensor([opt.default_radius]) * scale
    for i in range(len(thetas)):
        theta = torch.FloatTensor([thetas[i]])
        for idx in range(circle_size[i]):
            phis = torch.FloatTensor([(idx / circle_size[i]) * 360])
            poses = circle_poses(
                radius=radius,
                theta=theta,
                phi=phis,
                angle_overhead=opt.angle_overhead,
                angle_front=opt.angle_front,
            )
            matrix = np.linalg.inv(poses[0])
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]
            fovy = focal2fov(fov2focal(fov, opt.image_h), opt.image_w)
            FovY = fovy
            FovX = fov
            # delta polar/azimuth/radius to default view
            delta_polar = theta - opt.default_polar
            delta_azimuth = phis - opt.default_azimuth
            delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
            delta_radius = radius - opt.default_radius
            cam_infos.append(
                RandCameraInfo(
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    width=opt.image_w,
                    height=opt.image_h,
                    delta_polar=delta_polar,
                    delta_azimuth=delta_azimuth,
                    delta_radius=delta_radius,
                )
            )
    return cam_infos

def GenerateClipCameras(opt, clip_radius=4, clip_angle=90, size=120):
    fov = opt.default_fovy
    cam_infos = []
    # generate specific data structure
    for idx in range(size):
        thetas = torch.FloatTensor([clip_angle])
        phis = torch.FloatTensor([(idx / size) * 360])
        radius = torch.FloatTensor([clip_radius])
        # random pose on the fly
        poses = circle_poses(
            radius=radius,
            theta=thetas,
            phi=phis,
            angle_overhead=opt.angle_overhead,
            angle_front=opt.angle_front,
        )
        matrix = np.linalg.inv(poses[0])
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]
        fovy = focal2fov(fov2focal(fov, opt.image_h), opt.image_w)
        FovY = fovy
        FovX = fov

        # delta polar/azimuth/radius to default view
        delta_polar = thetas - opt.default_polar
        delta_azimuth = phis - opt.default_azimuth
        delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
        delta_radius = radius - opt.default_radius
        cam_infos.append(
            RandCameraInfo(
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                width=opt.image_w,
                height=opt.image_h,
                delta_polar=delta_polar,
                delta_azimuth=delta_azimuth,
                delta_radius=delta_radius,
            )
        )
    return cam_infos

def GenerateCircleCameras(opt, size=8, render45=False):
    fov = opt.default_fovy
    cam_infos = []
    # generate specific data structure
    for idx in range(size):
        thetas = torch.FloatTensor([opt.default_polar])
        phis = torch.FloatTensor([(idx / size) * 360])
        radius = torch.FloatTensor([opt.default_radius])
        # random pose on the fly
        poses = circle_poses(
            radius=radius,
            theta=thetas,
            phi=phis,
            angle_overhead=opt.angle_overhead,
            angle_front=opt.angle_front,
        )
        matrix = np.linalg.inv(poses[0])
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]
        fovy = focal2fov(fov2focal(fov, opt.image_h), opt.image_w)
        FovY = fovy
        FovX = fov

        # delta polar/azimuth/radius to default view
        delta_polar = thetas - opt.default_polar
        delta_azimuth = phis - opt.default_azimuth
        delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
        delta_radius = radius - opt.default_radius
        cam_infos.append(
            RandCameraInfo(
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                width=opt.image_w,
                height=opt.image_h,
                delta_polar=delta_polar,
                delta_azimuth=delta_azimuth,
                delta_radius=delta_radius,
            )
        )
    if render45:
        for idx in range(size):
            thetas = torch.FloatTensor([opt.default_polar * 2 // 3])
            phis = torch.FloatTensor([(idx / size) * 360])
            radius = torch.FloatTensor([opt.default_radius])
            # random pose on the fly
            poses = circle_poses(
                radius=radius,
                theta=thetas,
                phi=phis,
                angle_overhead=opt.angle_overhead,
                angle_front=opt.angle_front,
            )
            matrix = np.linalg.inv(poses[0])
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]
            fovy = focal2fov(fov2focal(fov, opt.image_h), opt.image_w)
            FovY = fovy
            FovX = fov

            # delta polar/azimuth/radius to default view
            delta_polar = thetas - opt.default_polar
            delta_azimuth = phis - opt.default_azimuth
            delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
            delta_radius = radius - opt.default_radius
            cam_infos.append(
                RandCameraInfo(
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    width=opt.image_w,
                    height=opt.image_h,
                    delta_polar=delta_polar,
                    delta_azimuth=delta_azimuth,
                    delta_radius=delta_radius,
                )
            )
    return cam_infos

def GenerateCircleCamerasInScene(
    opt,
    trans,
    trans_45,
    scale,
    scene_box,
    objects_args,
    cam_pose_method,
    size=8,
    render45=False,
    is_object=False,
    start_phi=0.0,
    end_phi=None,
    mode="default",
):
    if mode=="default":
        fov = opt.default_fovy
        if is_object:
            radius = torch.FloatTensor([opt.default_radius])
        else:
            radius = calc_radius(scene_box) - 0.01
    elif mode=="nearby":
        fov = 0.96  # opt.default_fovy
        radius = 0.1  # 1.0
        if end_phi is not None:
            if end_phi < start_phi:
                end_phi += 360

    cam_infos = []
    # generate specific data structure
    for idx in range(size):
        thetas = torch.FloatTensor([opt.default_polar])
        if mode=="nearby" and end_phi is not None:
            if ((idx / size) * 360 + start_phi) > end_phi:
                break
        phis = torch.FloatTensor([((idx / size) * 360 + start_phi) % 360])
        # random pose on the fly
        res, poses = scene_circle_poses(
            trans,
            scale,
            scene_box,
            objects_args,
            cam_pose_method,
            radius=radius,
            theta=thetas,
            phi=phis,
            angle_overhead=opt.angle_overhead,
            angle_front=opt.angle_front,
        )
        if not res:
            continue
        matrix = np.linalg.inv(poses[0])
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]
        fovy = focal2fov(fov2focal(fov, opt.image_h), opt.image_w)
        FovY = fovy
        FovX = fov

        # delta polar/azimuth/radius to default view
        delta_polar = thetas - opt.default_polar
        delta_azimuth = phis - opt.default_azimuth
        delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
        delta_radius = radius - opt.default_radius

        cam_infos.append(
            RandCameraInfo(
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                width=opt.image_w,
                height=opt.image_h,
                delta_polar=delta_polar,
                delta_azimuth=delta_azimuth,
                delta_radius=delta_radius,
            )
        )
    if render45:
        thetas = torch.FloatTensor([opt.default_polar * 2 // 3])
        radius = radius / torch.sin(thetas / 180 * np.pi)
        for idx in range(size):
            phis = torch.FloatTensor([((idx / size) * 360 + start_phi) % 360])
            # random pose on the fly
            res, poses = scene_circle_poses(
                trans_45,
                scale,
                scene_box,
                objects_args,
                cam_pose_method,
                radius=radius,
                theta=thetas,
                phi=phis,
                angle_overhead=opt.angle_overhead,
                angle_front=opt.angle_front,
            )
            if not res:
                continue
            matrix = np.linalg.inv(poses[0])
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]
            fovy = focal2fov(fov2focal(fov, opt.image_h), opt.image_w)
            FovY = fovy
            FovX = fov

            # delta polar/azimuth/radius to default view
            delta_polar = thetas - opt.default_polar
            delta_azimuth = phis - opt.default_azimuth
            delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
            delta_radius = radius - opt.default_radius

            cam_infos.append(
                RandCameraInfo(
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    width=opt.image_w,
                    height=opt.image_h,
                    delta_polar=delta_polar,
                    delta_azimuth=delta_azimuth,
                    delta_radius=delta_radius,
                )
            )
    return cam_infos

def GenerateCircleCamerasInSceneFaraway(
    opt,
    trans,
    trans_45,
    scale,
    scene_box,
    objects_args,
    cam_pose_method,
    size=8,
    is_object=False,
    start_phi=0.0,
):
    fov = opt.default_fovy
    if is_object:
        radius = torch.FloatTensor([opt.default_radius])
    else:
        radius = calc_radius(scene_box) - 0.01

    cam_infos = []
    # generate specific data structure
    thetas = torch.FloatTensor([opt.default_polar * 2 // 3])
    radius = radius / torch.sin(thetas / 180 * np.pi)
    for idx in range(size):
        phis = torch.FloatTensor([((idx / size) * 360 + start_phi) % 360])
        # random pose on the fly
        res, poses = scene_circle_poses(
            trans_45,
            scale,
            scene_box,
            objects_args,
            cam_pose_method,
            radius=radius,
            theta=thetas,
            phi=phis,
            angle_overhead=opt.angle_overhead,
            angle_front=opt.angle_front,
        )
        if not res:
            continue
        matrix = np.linalg.inv(poses[0])
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]
        fovy = focal2fov(fov2focal(fov, opt.image_h), opt.image_w)
        FovY = fovy
        FovX = fov

        # delta polar/azimuth/radius to default view
        delta_polar = thetas - opt.default_polar
        delta_azimuth = phis - opt.default_azimuth
        delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
        delta_radius = radius - opt.default_radius

        cam_infos.append(
            RandCameraInfo(
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                width=opt.image_w,
                height=opt.image_h,
                delta_polar=delta_polar,
                delta_azimuth=delta_azimuth,
                delta_radius=delta_radius,
            )
        )
    return cam_infos

def loadRandomCam(opt: GenerateCamParams, SSAA: bool = False) -> RCamera:
    cam_info = GenerateRandomCameras(opt, SSAA=True)
    return RCamera(
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        delta_polar=cam_info.delta_polar,
        delta_azimuth=cam_info.delta_azimuth,
        delta_radius=cam_info.delta_radius,
        opt=opt,
        data_device=opt.device,
        SSAA=SSAA,
    )

def loadRandomCamAvoidMultiFace_4p(
    opt: GenerateCamParams, step_ratio: float, SSAA: bool = False, size: int = 4
) -> List[RCamera]:

    cam_infos = []
    transs = []
    scales = []
    dirs = "random"
    rcc = random.random()
    if step_ratio < 0.1:
        if rcc < 0.7:
            dirs = "front"
        else:
            dirs = "back"
    elif step_ratio < 0.7:
        if rcc < 0.3:
            dirs = "front"
        elif rcc < 0.6:
            dirs = "back"
        else:
            dirs = "side"
    else:
        dirs = "random"
    for i in range(size):
        cam_info = GenerateRandomCamerasAvoidMultiFace(
            opt, step_ratio, SSAA=True, dirs=dirs
        )
        if step_ratio > 0.7:
            trans = np.array([0, 0, random.random() * 0.5 - 0.2])
        else:
            trans = np.array([0, 0, 0])
        cam_infos.append(cam_info)
        transs.append(trans)
        scales.append(1.0)

    camera_list = []
    for trans, scale, cam_info in zip(transs, scales, cam_infos):
        camera_list.append(
            RCamera(
                R=cam_info.R,
                T=cam_info.T,
                FoVx=cam_info.FovX,
                FoVy=cam_info.FovY,
                delta_polar=cam_info.delta_polar,
                delta_azimuth=cam_info.delta_azimuth,
                delta_radius=cam_info.delta_radius,
                opt=opt,
                trans=trans,
                scale=scale,
                data_device=opt.device,
                SSAA=SSAA,
            )
        )

    return camera_list

def loadClipCam(opt, angles=[75, 90]):
    SSAA = False
    cam_infos = GenerateClipCameras(opt, clip_angle=75) + GenerateClipCameras(
        opt, clip_angle=90
    )
    camera_list = []
    for id, cam_info in enumerate(cam_infos):
        camera_list.append(
            RCamera(
                R=cam_info.R,
                T=cam_info.T,
                FoVx=cam_info.FovX,
                FoVy=cam_info.FovY,
                delta_polar=cam_info.delta_polar,
                delta_azimuth=cam_info.delta_azimuth,
                delta_radius=cam_info.delta_radius,
                opt=opt,
                data_device=opt.device,
                SSAA=SSAA,
            )
        )
    return camera_list

def loadCircleCam(opt, circle_size=120, render45=True):
    SSAA = False
    cam_infos = GenerateCircleCameras(opt, circle_size, render45)
    camera_list = []
    for id, cam_info in enumerate(cam_infos):
        camera_list.append(
            RCamera(
                R=cam_info.R,
                T=cam_info.T,
                FoVx=cam_info.FovX,
                FoVy=cam_info.FovY,
                delta_polar=cam_info.delta_polar,
                delta_azimuth=cam_info.delta_azimuth,
                delta_radius=cam_info.delta_radius,
                opt=opt,
                data_device=opt.device,
                SSAA=SSAA,
            )
        )
    return camera_list

def loadSphereCam(opt, circle_size=48):
    SSAA = False
    cam_infos = GenerateSphereCameras(opt, circle_size)
    camera_list = []
    for id, cam_info in enumerate(cam_infos):
        camera_list.append(
            RCamera(
                R=cam_info.R,
                T=cam_info.T,
                FoVx=cam_info.FovX,
                FoVy=cam_info.FovY,
                delta_polar=cam_info.delta_polar,
                delta_azimuth=cam_info.delta_azimuth,
                delta_radius=cam_info.delta_radius,
                opt=opt,
                data_device=opt.device,
                SSAA=SSAA,
            )
        )
    return camera_list

def loadRecoCam(
    opt: GenerateCamParams,
    circle_size: list = [4],
    thetas: list = [90],
    scale: float = 1.0,
):
    SSAA = False
    cam_infos = GenerateRecoCameras(opt, circle_size, thetas, scale)
    camera_list = []
    for id, cam_info in enumerate(cam_infos):
        camera_list.append(
            RCamera(
                R=cam_info.R,
                T=cam_info.T,
                FoVx=cam_info.FovX,
                FoVy=cam_info.FovY,
                delta_polar=cam_info.delta_polar,
                delta_azimuth=cam_info.delta_azimuth,
                delta_radius=cam_info.delta_radius,
                opt=opt,
                data_device=opt.device,
                SSAA=SSAA,
            )
        )
    return camera_list

def GenSingleCam(opt, fovx, radius, phi, theta, img_h, img_w):
    thetas = torch.FloatTensor([theta])
    phis = torch.FloatTensor([phi])
    radius = torch.FloatTensor([radius])
    # random pose on the fly
    poses = circle_poses(
        radius=radius,
        theta=thetas,
        phi=phis,
        angle_overhead=opt.angle_overhead,
        angle_front=opt.angle_front,
    )
    matrix = np.linalg.inv(poses[0])
    R = -np.transpose(matrix[:3, :3])
    R[:, 0] = -R[:, 0]
    T = -matrix[:3, 3]
    fovy = focal2fov(fov2focal(fovx, img_h), img_w)
    FovY = fovy
    FovX = fovx

    # delta polar/azimuth/radius to default view
    delta_polar = thetas - opt.default_polar
    delta_azimuth = phis - opt.default_azimuth
    delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
    delta_radius = radius - opt.default_radius
    return True, RandCameraInfo(
        R=R,
        T=T,
        FovY=FovY,
        FovX=FovX,
        width=img_w,
        height=img_h,
        delta_polar=delta_polar,
        delta_azimuth=delta_azimuth,
        delta_radius=delta_radius,
    )

def loadSingleCam(
    opt,
    camera_center=[0, 0, 0],
    object_center=[1, 0, 0],
    theta=90,
    radius=3.5,
    fov=0.96,
    img_w=1920,
    img_h=1080,
):
    opt.image_w = img_w
    opt.image_h = img_h
    SSAA = False
    object_center_ = torch.Tensor(object_center)
    camera_center_ = torch.Tensor(camera_center)
    trans = camera_center_.numpy()
    phi = (
        torch.arctan2(
            object_center_[0] - camera_center_[0], object_center_[1] - camera_center_[1]
        )
        * 180
        / torch.pi
        + 180
    )  # 0
    res, cam_info = GenSingleCam(opt, fov, radius, phi, theta, img_h, img_w)
    scale = 1.0
    return RCamera(
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        delta_polar=cam_info.delta_polar,
        delta_azimuth=cam_info.delta_azimuth,
        delta_radius=cam_info.delta_radius,
        opt=opt,
        trans=trans,
        scale=scale,
        data_device=opt.device,
        SSAA=SSAA,
    )

class SceneCameraLoader:
    def __init__(self, opt: GenerateCamParams, scene_box, objects_args, cam_pose_method):
        self.opt = opt
        self.opt_infer = opt
        self.s_box = scene_box
        self.o_args = objects_args
        self.c_method = cam_pose_method
        
    def Stage1_Outdoor(
        self,
        SSAA=True,
    ):
        trans = np.array(
            [0, 0, (self.s_box[5] + self.s_box[2]) / 2.0 + random.random() - 0.5]
        )
        scale = 1.0
        cam_infos = []
        size = 12
        for idx in range(size):
            phi_range_ = [idx / size * 360, idx / size * 360]
            phi_range_ = sample_jit(phi_range_, 360 / size, 360, 360, True)
            cam_info = GenerateCamerasSceneOutdoor1(
                self.opt,
                torch.FloatTensor(trans),
                scale,
                self.s_box,
                self.o_args,
                self.c_method,
                phi_range_=phi_range_,
                SSAA=True,
            )
            cam_infos.append(cam_info)
        camera_list = []
        for id, cam_info in enumerate(cam_infos):
            camera_list.append(
                RCamera(
                    R=cam_info.R,
                    T=cam_info.T,
                    FoVx=cam_info.FovX,
                    FoVy=cam_info.FovY,
                    delta_polar=cam_info.delta_polar,
                    delta_azimuth=cam_info.delta_azimuth,
                    delta_radius=cam_info.delta_radius,
                    opt=self.opt,
                    trans=trans,
                    scale=scale,
                    data_device=self.opt.device,
                    SSAA=SSAA,
                )
            )
        return camera_list
    
    def Stage1_Outdoor2(
        self,
        SSAA=True,
    ):
        cam_infos = []
        transs = []
        scales = []

        trans_phis_d = random.random() * 360 - 180  # 2 * np.pi
        trans_phis = trans_phis_d / 180 * np.pi
        if trans_phis < 0:
            trans_phis += 2 * np.pi
        radius_trans_max = min(
            np.abs(self.s_box[0]), np.abs(self.s_box[1]), self.s_box[3], self.s_box[4]
        )
        for i in range(4):
            if i == 0:
                radius_trans = (
                    -radius_trans_max / 2.0
                    + random.random() * radius_trans_max / 10.0
                    - radius_trans_max / 20.0
                )
            elif i == 1:
                radius_trans = (
                    -radius_trans_max / 4.0
                    + random.random() * radius_trans_max / 10.0
                    - radius_trans_max / 20.0
                )
            elif i == 2:
                radius_trans = (
                    radius_trans_max / 4.0
                    + random.random() * radius_trans_max / 10.0
                    - radius_trans_max / 20.0
                )
            else:
                radius_trans = (
                    radius_trans_max / 2.0
                    + random.random() * radius_trans_max / 10.0
                    - radius_trans_max / 20.0
                )

            trans = np.array(
                [
                    radius_trans * np.sin(trans_phis),
                    radius_trans * np.cos(trans_phis),
                    (self.s_box[5] + self.s_box[2]) / 2.0 + random.random() - 0.5,
                ]
            )
            scale = 1.0
            if i <= 1:
                scale *= -1
            cam_info, scale = GenerateCamerasSceneOutdoor2(
                self.opt,
                torch.FloatTensor(trans),
                scale,
                self.s_box,
                self.o_args,
                self.c_method,
                [trans_phis_d, trans_phis_d],
                SSAA=True,
            )
            cam_infos.append(cam_info)
            scales.append(scale)
            transs.append(trans)
        camera_list = []
        for cam_info, trans, scale in zip(cam_infos, transs, scales):
            camera_list.append(
                RCamera(
                    R=cam_info.R,
                    T=cam_info.T,
                    FoVx=cam_info.FovX,
                    FoVy=cam_info.FovY,
                    delta_polar=cam_info.delta_polar,
                    delta_azimuth=cam_info.delta_azimuth,
                    delta_radius=cam_info.delta_radius,
                    opt=self.opt,
                    trans=trans,
                    scale=scale,
                    data_device=self.opt.device,
                    SSAA=SSAA,
                )
            )
        return camera_list

    def Stage2_Outdoor(
        self,
        SSAA=True,
    ):
        cam_infos = []
        transs = []
        scales = []

        trans_phis_d = random.random() * 360 - 180  # 2 * np.pi
        trans_phis = trans_phis_d / 180 * np.pi
        if trans_phis < 0:
            trans_phis += 2 * np.pi
        radius_trans_max = min(
            np.abs(self.s_box[0]), np.abs(self.s_box[1]), self.s_box[3], self.s_box[4]
        )
        for i in range(4):
            if i == 0:
                radius_trans = (
                    -radius_trans_max / 2.0
                    + random.random() * radius_trans_max / 10.0
                    - radius_trans_max / 20.0
                )
            elif i == 1:
                radius_trans = (
                    -radius_trans_max / 4.0
                    + random.random() * radius_trans_max / 10.0
                    - radius_trans_max / 20.0
                )
            elif i == 2:
                radius_trans = (
                    radius_trans_max / 4.0
                    + random.random() * radius_trans_max / 10.0
                    - radius_trans_max / 20.0
                )
            else:
                radius_trans = (
                    radius_trans_max / 2.0
                    + random.random() * radius_trans_max / 10.0
                    - radius_trans_max / 20.0
                )

            trans = np.array(
                [
                    radius_trans * np.sin(trans_phis),
                    radius_trans * np.cos(trans_phis),
                    (self.s_box[5] + self.s_box[2]) * 2.0 / 3.0,
                ]
            )  # + random.random() - 0.5])
            scale = 1.0
            if i <= 1:
                scale *= -1
            cam_info, scale = GenerateCamerasSceneOutdoor3(
                self.opt,
                torch.FloatTensor(trans),
                scale,
                self.s_box,
                self.o_args,
                self.c_method,
                [trans_phis_d, trans_phis_d],
                SSAA=True,
            )
            cam_infos.append(cam_info)
            scales.append(scale)
            transs.append(trans)
        camera_list = []
        for cam_info, trans, scale in zip(cam_infos, transs, scales):
            camera_list.append(
                RCamera(
                    R=cam_info.R,
                    T=cam_info.T,
                    FoVx=cam_info.FovX,
                    FoVy=cam_info.FovY,
                    delta_polar=cam_info.delta_polar,
                    delta_azimuth=cam_info.delta_azimuth,
                    delta_radius=cam_info.delta_radius,
                    opt=self.opt,
                    trans=trans,
                    scale=scale,
                    data_device=self.opt.device,
                    SSAA=SSAA,
                )
            )
        return camera_list

    def Stage3_Outdoor(
        self,
        opti_target="env",
        SSAA=True
    ):
        cam_infos = []
        transs = []
        scales = []

        size = 16
        radius_trans_max = min(
            np.abs(self.s_box[0]), np.abs(self.s_box[1]), self.s_box[3], self.s_box[4]
        )
        for idx in range(size):
            trans_phis_d = idx / size * 360 - 180  # 2 * np.pi
            trans_phis = trans_phis_d / 180 * np.pi
            if trans_phis < 0:
                trans_phis += 2 * np.pi
            if opti_target == "env":
                radius_trans = -radius_trans_max / 4.0
                trans = np.array(
                    [
                        radius_trans * np.sin(trans_phis),
                        radius_trans * np.cos(trans_phis),
                        (self.s_box[5] + self.s_box[2]) / 2.0,
                    ]
                )  # + random.random() - 0.5])
            else:
                radius_trans = -radius_trans_max * 2.0 / 3.0
                trans = np.array(
                    [
                        radius_trans * np.sin(trans_phis),
                        radius_trans * np.cos(trans_phis),
                        (self.s_box[5]),
                    ]
                )  # + random.random() - 0.5]) ### +self.s_box[2]
            scale = -1.0
            cam_info, scale = GenerateCamerasSceneOuddoor4(
                self.opt,
                torch.FloatTensor(trans),
                scale,
                self.s_box,
                self.o_args,
                self.c_method,
                [trans_phis_d, trans_phis_d],
                opti_target,
                SSAA=True,
            )
            cam_infos.append(cam_info)
            scales.append(scale)
            transs.append(trans)
            scale = -1.0
            cam_info, scale = GenerateCamerasSceneOuddoor4(
                self.opt,
                torch.FloatTensor(trans),
                scale,
                self.s_box,
                self.o_args,
                self.c_method,
                [trans_phis_d, trans_phis_d],
                opti_target + "2",
                SSAA=True,
            )
            cam_infos.append(cam_info)
            scales.append(scale)
            transs.append(trans)
        camera_list = []
        for cam_info, trans, scale in zip(cam_infos, transs, scales):
            camera_list.append(
                RCamera(
                    R=cam_info.R,
                    T=cam_info.T,
                    FoVx=cam_info.FovX,
                    FoVy=cam_info.FovY,
                    delta_polar=cam_info.delta_polar,
                    delta_azimuth=cam_info.delta_azimuth,
                    delta_radius=cam_info.delta_radius,
                    opt=self.opt,
                    trans=trans,
                    scale=scale,
                    data_device=self.opt.device,
                    SSAA=SSAA,
                )
            )
        return camera_list

    def Stage1_Indoor(
        self,
        size=8,
        view_floor=False,
        SSAA=True,
    ):
        trans = np.array(
            [0, 0, (self.s_box[5] + self.s_box[2]) / 2.0 + random.random() * 1 - 0.5]
        )
        scale = 1.0
        cam_infos = []
        scales = []
        for idx in range(size):
            try:
                phi_range_ = [idx / size * 360, idx / size * 360]
                phi_range_ = sample_jit(phi_range_, 360 / size, 360, 360, True)
                cam_info, scale = GenerateCamerasSceneIndoor1(
                    self.opt,
                    torch.FloatTensor(trans),
                    scale,
                    self.s_box,
                    self.o_args,
                    self.c_method,
                    phi_range_=phi_range_,
                    view_floor=view_floor,
                    SSAA=True,
                )
                cam_infos.append(cam_info)
                scales.append(scale)
            except Exception:
                print("camera sampling failure [single angle]")
        camera_list = []
        for cam_info, scale in zip(cam_infos, scales):
            camera_list.append(
                RCamera(
                    R=cam_info.R,
                    T=cam_info.T,
                    FoVx=cam_info.FovX,
                    FoVy=cam_info.FovY,
                    delta_polar=cam_info.delta_polar,
                    delta_azimuth=cam_info.delta_azimuth,
                    delta_radius=cam_info.delta_radius,
                    opt=self.opt,
                    trans=trans,
                    scale=scale,
                    data_device=self.opt.device,
                    SSAA=SSAA,
                )
            )
        return camera_list

    def Stage2_Indoor(
        self,
        affine_params=None,
        idx=0,
        size=8,
        SSAA=True,
    ):
        cam_infos = []
        transs = []
        scales = []

        radius_trans_max = min(
            np.abs(self.s_box[0]), np.abs(-self.s_box[1]), self.s_box[3], self.s_box[4]
        )
        if affine_params is not None:
            if len(affine_params["S"] == 3):
                diff_z = affine_params["S"][2].numpy() / 2.0 + random.random() - 0.5
            else:
                diff_z = affine_params["S"][0].numpy() / 2.0 + random.random() - 0.5
            trans = np.array(affine_params["T"]) + np.array([0, 0, diff_z])
            scale = np.clip(affine_params["S"][0].numpy(), 0.75, 1.5)
            trans_phis_d = np.arctan2(trans[0], trans[1]) * 180 / np.pi

            obj_distance_to_edge = distance_point_to_aabb(
                trans, self.s_box[:3], self.s_box[3:]
            )
            for i in range(8):
                cam_info, scale = GenerateCamerasSceneIndoor2(
                    self.opt,
                    torch.FloatTensor(trans),
                    scale,
                    self.s_box,
                    self.o_args,
                    self.c_method,
                    max_radius=obj_distance_to_edge,
                    SSAA=True,
                )
                cam_infos.append(cam_info)
                scales.append(scale)
                transs.append(trans)
        else:
            trans_phis_d = idx / size * 360 - 180  # 2 * np.pi
            trans_phis_d = sample_jit(trans_phis_d, 360 / size, 180, 360)
            trans_phis = trans_phis_d / 180 * np.pi
            if trans_phis < 0:
                trans_phis += 2 * np.pi
            radius_trans = radius_trans_max / 3.0
            trans = np.array(
                [
                    radius_trans * np.sin(trans_phis),
                    radius_trans * np.cos(trans_phis),
                    (self.s_box[5] + self.s_box[2]) / 2.0 + random.random() * 2 - 1,
                ]
            )
            scale = 1.0
            phis_range = [trans_phis_d + 180 - 60, trans_phis_d + 180 + 60]
            for i in range(8):
                cam_info, scale = GenerateCamerasSceneIndoor2(
                    self.opt,
                    torch.FloatTensor(trans),
                    scale,
                    self.s_box,
                    self.o_args,
                    self.c_method,
                    phi_range_=phis_range,
                    SSAA=True,
                )
                cam_infos.append(cam_info)
                scales.append(scale)
                transs.append(trans)
        camera_list = []
        for cam_info, trans, scale in zip(cam_infos, transs, scales):
            camera_list.append(
                RCamera(
                    R=cam_info.R,
                    T=cam_info.T,
                    FoVx=cam_info.FovX,
                    FoVy=cam_info.FovY,
                    delta_polar=cam_info.delta_polar,
                    delta_azimuth=cam_info.delta_azimuth,
                    delta_radius=cam_info.delta_radius,
                    opt=self.opt,
                    trans=trans,
                    scale=scale,
                    data_device=self.opt.device,
                    SSAA=SSAA,
                )
            )
        return camera_list

    def Line(
        self,
        start=[0, 0, 0],
        stop=[1, 1, 1],
        step_size=0.1,
        radius=3.5,
        fov=0.96,
        img_w=1920,
        img_h=1080,
    ):
        opt = self.opt_infer
        opt.image_w = img_w
        opt.image_h = img_h
        SSAA = False
        p_start = torch.Tensor(start)
        p_stop = torch.Tensor(stop)
        num_steps = int(torch.dist(p_start, p_stop).item() / step_size)
        xs = torch.linspace(p_start[0], p_stop[0], num_steps)
        ys = torch.linspace(p_start[1], p_stop[1], num_steps)
        zs = torch.linspace(p_start[2], p_stop[2], num_steps)
        trans_list = []
        for i in range(num_steps):
            trans_list.append([xs[i], ys[i], zs[i]])
        transs = []
        scales = []
        cam_infos = []
        phi = (
            torch.arctan2(p_stop[0] - p_start[0], p_stop[1] - p_start[1]) * 180 / torch.pi
            + 180
        )
        theta = 90
        for trans in trans_list:
            res, cam_info = GenSingleCam(opt, 0.96, 1.0, phi, theta, img_h, img_w)
            scale = 1.0
            if res:
                transs.append(trans)
                scales.append(scale)
                cam_infos.append(cam_info)
            pass
        camera_list = []
        for trans, scale, cam_info in zip(transs, scales, cam_infos):
            camera_list.append(
                RCamera(
                    R=cam_info.R,
                    T=cam_info.T,
                    FoVx=cam_info.FovX,
                    FoVy=cam_info.FovY,
                    delta_polar=cam_info.delta_polar,
                    delta_azimuth=cam_info.delta_azimuth,
                    delta_radius=cam_info.delta_radius,
                    opt=opt,
                    trans=trans,
                    scale=scale,
                    data_device=opt.device,
                    SSAA=SSAA,
                )
            )
        return camera_list

    def Circle(
        self,
        affine_params=None,
        circle_size=120,
        render45=True,
    ):
        SSAA = False
        if affine_params is None:
            trans = np.array([0, 0, (self.s_box[5] + self.s_box[2]) / 2.0])
            trans_45 = np.array([0, 0, self.s_box[2]])
            scale = 1.0
            is_object = False
        else:
            if len(affine_params["S"]) == 3:
                diff_z = affine_params["S"][2].numpy() / 2.0
            else:
                diff_z = affine_params["S"][0].numpy() / 2.0
            trans_45 = np.array(affine_params["T"])
            trans = np.array(affine_params["T"]) + np.array([0, 0, diff_z])
            scale = np.clip(affine_params["S"][0].numpy(), 0.75, 1.5)
            is_object = True
        cam_infos = []
        while len(cam_infos) < circle_size // 2:
            scale *= 0.98
            cam_infos = GenerateCircleCamerasInScene(
                self.opt,
                torch.FloatTensor(trans),
                torch.FloatTensor(trans_45),
                scale,
                self.s_box,
                self.o_args,
                self.c_method,
                circle_size,
                render45,
                is_object,
            )

        camera_list = []

        for id, cam_info in enumerate(cam_infos):
            camera_list.append(
                RCamera(
                    R=cam_info.R,
                    T=cam_info.T,
                    FoVx=cam_info.FovX,
                    FoVy=cam_info.FovY,
                    delta_polar=cam_info.delta_polar,
                    delta_azimuth=cam_info.delta_azimuth,
                    delta_radius=cam_info.delta_radius,
                    opt=self.opt,
                    trans=trans if id < circle_size else trans_45,
                    scale=scale,
                    data_device=self.opt.device,
                    SSAA=SSAA,
                )
            )
        return camera_list

    def Circle2(
        self,
        start_phi=0.0,
        end_phi=None,
        affine_params=None,
        circle_size=120,
        render45=True,
    ):
        SSAA = False
        if affine_params is None:
            trans = np.array([0, 0, (self.s_box[5] + self.s_box[2]) / 2.0])
            trans_45 = np.array([0, 0, self.s_box[2]])
            scale = 1.0
            is_object = False
        else:
            trans_45 = np.array(affine_params["T"])
            trans = np.array(affine_params["T"])
            scale = np.clip(affine_params["S"][0].numpy(), 0.75, 1.5)
            is_object = True
        cam_infos = []
        cam_infos = GenerateCircleCamerasInScene(
            self.opt,
            torch.FloatTensor(trans),
            torch.FloatTensor(trans_45),
            scale,
            self.s_box,
            self.o_args,
            self.c_method,
            circle_size,
            render45,
            is_object,
            start_phi,
            end_phi,
            mode="nearby",
        )

        camera_list = []

        for id, cam_info in enumerate(cam_infos):
            camera_list.append(
                RCamera(
                    R=cam_info.R,
                    T=cam_info.T,
                    FoVx=cam_info.FovX,
                    FoVy=cam_info.FovY,
                    delta_polar=cam_info.delta_polar,
                    delta_azimuth=cam_info.delta_azimuth,
                    delta_radius=cam_info.delta_radius,
                    opt=self.opt,
                    trans=trans if id < circle_size else trans_45,
                    scale=scale,
                    data_device=self.opt.device,
                    SSAA=SSAA,
                )
            )
        return camera_list

    def Circle3(
        self,
        affine_params=None,
        circle_size=120,
        render45=True,
    ):
        SSAA = False
        if affine_params is None:
            trans = np.array([0, 0, (self.s_box[5] + self.s_box[2]) / 2.0])
            trans_45 = np.array([0, 0, self.s_box[2]])
            if self.c_method == "indoor":
                trans_45 = np.array([0, 0, (self.s_box[5] + self.s_box[2]) / 3.0])
            scale = 1.0
            is_object = False
        else:
            if len(affine_params["S"]) == 3:
                diff_z = affine_params["S"][2].numpy() / 2.0
            else:
                diff_z = affine_params["S"][0].numpy() / 2.0
            trans_45 = np.array(affine_params["T"])
            trans = np.array(affine_params["T"]) + np.array([0, 0, diff_z])
            scale = np.clip(affine_params["S"][0].numpy(), 0.75, 1.5)
            is_object = True
        cam_infos = []
        while len(cam_infos) < circle_size // 2:
            scale *= 0.98
            cam_infos = GenerateCircleCamerasInScene(
                self.opt,
                torch.FloatTensor(trans),
                torch.FloatTensor(trans_45),
                scale,
                self.s_box,
                self.o_args,
                self.c_method,
                circle_size,
                False,
                is_object,
            )

        cam_infos_45 = []

        scale_45 = 1.2
        if render45:
            while len(cam_infos_45) < circle_size // 2:
                scale_45 *= 0.98
                cam_infos_45 = GenerateCircleCamerasInSceneFaraway(
                    self.opt,
                    torch.FloatTensor(trans),
                    torch.FloatTensor(trans_45),
                    scale_45,
                    self.s_box,
                    self.o_args,
                    self.c_method,
                    circle_size,
                    is_object,
                )

        camera_list = []

        for id, cam_info in enumerate(cam_infos):
            camera_list.append(
                RCamera(
                    R=cam_info.R,
                    T=cam_info.T,
                    FoVx=cam_info.FovX,
                    FoVy=cam_info.FovY,
                    delta_polar=cam_info.delta_polar,
                    delta_azimuth=cam_info.delta_azimuth,
                    delta_radius=cam_info.delta_radius,
                    opt=self.opt,
                    trans=trans,
                    scale=scale,
                    data_device=self.opt.device,
                    SSAA=SSAA,
                )
            )

        if render45:
            for id, cam_info in enumerate(cam_infos_45):
                camera_list.append(
                    RCamera(
                        R=cam_info.R,
                        T=cam_info.T,
                        FoVx=cam_info.FovX,
                        FoVy=cam_info.FovY,
                        delta_polar=cam_info.delta_polar,
                        delta_azimuth=cam_info.delta_azimuth,
                        delta_radius=cam_info.delta_radius,
                        opt=self.opt,
                        trans=trans_45,
                        scale=scale_45,
                        data_device=self.opt.device,
                        SSAA=SSAA,
                    )
                )
        return camera_list

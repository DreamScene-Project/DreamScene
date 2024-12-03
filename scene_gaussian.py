import gc
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import omegaconf
import torch
from diff_gaussian_rasterization import (GaussianRasterizationSettings,
                                         GaussianRasterizer)
from e3nn import o3
from loguru import logger
from pytorch3d.transforms import (euler_angles_to_matrix, matrix_to_quaternion,
                                  quaternion_to_matrix)

from gs_renderer import GaussianModel
from utils.cam_utils import RCamera, loadSphereCam
from utils.quaternion_utils import quaternion_raw_multiply


@dataclass
class ObjectGaussian:
    id: str
    step: int
    model: GaussianModel
    text: dict
    image: dict
    cam_pose_method: str

@dataclass
class ObjectArgs:
    clas: str
    objectId: int
    affine_params: dict  # {"T","R","S"}
    bbox: torch.Tensor  # xmin,xmax,ymin,ymax,zmin,zmax

class SceneGaussian(GaussianModel):
    def __init__(self, cfg, white_background=True) -> None:
        self.white_background = white_background
        self.bg_color = torch.tensor(
            [1, 1, 1] if white_background else [0, 0, 0],
            dtype=torch.float32,
            device="cuda",
        )
        self.cfg = cfg
        self.cameras_extent = self.cfg.generateCamParams.default_radius
        self.objects_args = []
        self.objects_count = 0
        self.stage_n = 0

    def ckpt_checker(self, exp_path, id, objectParams):
        save_dir = exp_path / "checkpoints"
        ckpts_list = os.listdir(save_dir)
        iters_restore = 0
        for ckpt_name in ckpts_list:
            if id == "_".join(ckpt_name.split("_")[:-2]):
                ckpt_iter = ckpt_name.split("_")[-2]
                if ckpt_iter == "final":
                    iters_restore = "final"
                    break
                if iters_restore < int(ckpt_iter):
                    iters_restore = int(ckpt_iter)
        if iters_restore != 0:
            path = os.path.join(save_dir, f"{id}_{iters_restore}_model.ply")
            object_gaussian = ObjectGaussian(
                id,
                model=GaussianModel(
                    {"sh_degree": objectParams.sh_degree, "path": path},
                    "restore",
                ),
                step=iters_restore,
                text={},
                image={},
                cam_pose_method=objectParams.cam_pose_method,
            )
            logger.debug(f"Restored ckpt from {path}")
            return object_gaussian
        return 0

    def ply_checker(self, id, objectParams):
        path = Path(str(objectParams.init_prompt))
        iters_restore = int(os.path.basename(path).split("_")[-2])
        object_gaussian = ObjectGaussian(
            id,
            model=GaussianModel(
                {"sh_degree": objectParams.sh_degree, "path": path}, "restore"
            ),
            step=iters_restore,
            text={},
            image={},
            cam_pose_method=objectParams.cam_pose_method,
        )
        logger.debug(f"Loaded Object from {path}")
        return object_gaussian

    def init_object_gaussian(self, id, objectParams):
        object_gaussians_dict = {}
        logger.debug(f"Object Init Start: object_name: {id}")
        if objectParams.init_guided == "load":
            object_gaussian = self.ply_checker(id, objectParams)
        else:
            exp_path = Path("experiments/") / self.cfg.log.exp_name
            load_ckpt = True

            object_gaussian = 0
            if load_ckpt:
                object_gaussian = self.ckpt_checker(exp_path, id, objectParams)
            if object_gaussian == 0:
                object_gaussian = ObjectGaussian(
                    id,
                    model=GaussianModel(objectParams, "object", exp_path),
                    step=0,
                    text={},
                    image={},
                    cam_pose_method=objectParams.cam_pose_method,
                )
                logger.debug(f"Create New Object {id}")

        object_gaussian.model.training_setup(self.cfg.optimizationParams)
        object_gaussian.text = {
            "text": objectParams.text,
            "negative_text": objectParams.negative_text,
        }
        object_gaussians_dict[id] = object_gaussian

        logger.debug("Objects Initialization Completed")
        self.object_gaussians_dict = object_gaussians_dict

    def init_gaussians(self):
        object_gaussians_dict = {}
        if self.cfg.scene_configs.objects is not None:
            num_object_gaussians = len(self.cfg.scene_configs.objects)
        else:
            num_object_gaussians = 0
        for i in range(num_object_gaussians):
            object_in_scene = self.cfg.scene_configs.objects[i]
            logger.debug(f"Object Init Start: object_name: {object_in_scene.id}")
            if object_in_scene.init_guided in ["indoor", "outdoor"]:
                object_in_scene.cam_pose_method = object_in_scene.init_guided
            if object_in_scene.init_guided == "load":
                path = Path(str(object_in_scene.init_prompt))
                iters_restore = int(os.path.basename(path).split("_")[-2])
                object_gaussian = ObjectGaussian(
                    object_in_scene.id,
                    model=GaussianModel(
                        {"sh_degree": object_in_scene.sh_degree, "path": path},
                        "restore",
                    ),
                    step=iters_restore,
                    text={},
                    image={},
                    cam_pose_method=object_in_scene.cam_pose_method,
                )
            else:
                exp_path = Path("experiments/") / self.cfg.log.exp_name
                load_ckpt = True

                if load_ckpt:
                    object_gaussian = self.ckpt_checker(
                        exp_path, object_in_scene.id, object_in_scene
                    )
                if object_gaussian == 0:
                    object_gaussian = ObjectGaussian(
                        object_in_scene.id,
                        model=GaussianModel(object_in_scene, "object", exp_path),
                        step=0,
                        text={},
                        image={},
                        cam_pose_method=object_in_scene.cam_pose_method,
                    )
                    logger.debug(f"Create New Object {object_in_scene.id}")
            object_gaussian.model.training_setup(self.cfg.optimizationParams)
            object_gaussians_dict[object_in_scene.id] = object_gaussian

        logger.debug("Objects Initialization Completed")
        self.object_gaussians_dict = object_gaussians_dict

    def load_ckpt(self, path):
        env_args, floor_args, self.stage_n = torch.load(path)
        logger.debug("Load ckpt From {} at stage {}".format(path, self.stage_n))
        self.restore(env_args, floor_args, self.cfg.sceneOptimizationParams)

    def init_gaussian_scene(self):
        scene_cfg = self.cfg.scene_configs.scene
        logger.debug(f"Start Init: scene_name: {scene_cfg.scene_name}")
        exp_path = Path("experiments/") / self.cfg.log.exp_name

        load_ckpt = True
        logger.debug(f"Create New Scene {scene_cfg.scene_name}")
        self.scene_box = torch.zeros(6).cuda()

        self.add_objects_to_scene(scene_cfg, exp_path)
        self.env_gaussian.training_setup(self.cfg.sceneOptimizationParams)
        self.floor_gaussian.training_setup(self.cfg.sceneOptimizationParams)
        if load_ckpt:
            scene_ckpt_path = exp_path / "scene_checkpoints"
            ckpt_list = os.listdir(scene_ckpt_path)
            stage_restore = 0
            for ckpts in ckpt_list:
                if "scene" in ckpts and ".ckpt" in ckpts:
                    ckpt_iter = ckpts.split("_")[-2]
                    if ckpt_iter == "final":
                        continue
                    if stage_restore < int(ckpt_iter):
                        stage_restore = int(ckpt_iter)
            if stage_restore:
                self.load_ckpt(scene_ckpt_path / f"scene_{stage_restore}_stage.ckpt")

    def capture(self):
        return (
            self.env_gaussian.capture(),
            self.floor_gaussian.capture(),
            self.stage_n,
        )

    def restore(self, env_args, floor_args, training_args):
        self.env_gaussian.restore(env_args, training_args)
        self.floor_gaussian.restore(floor_args, training_args)

    def compress_objects(self, scene_cfg: dict, exp_path):
        for obj in scene_cfg.scene_composition:
            save_dir = exp_path / "checkpoints"
            path = os.path.join(save_dir, f"{obj.id}_final_model.ply")
            tmp_ogs = GaussianModel(
                {
                    "sh_degree": scene_cfg.sh_degree,
                    "init_guided": path,
                    "init_prompt": "",
                },
                "object",
                exp_path,
            )
            tmp_ogs.training_setup(self.cfg.optimizationParams)
            gaussian_filtering(tmp_ogs, self, self.cfg.generateCamParams)
            tmp_ogs.save_ply(path)
            logger.debug("Saved " + path)

    def init_sh_transform_matrices(self):
        self.v_to_sh_transform = torch.tensor(
            [[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=torch.float32
        ).cuda()
        self.sh_to_v_transform = (
            self.v_to_sh_transform.transpose(0, 1).unsqueeze(0).cuda()
        )
        self.v_to_sh_transform = self.v_to_sh_transform.unsqueeze(0)

    def export_layout(self, scene_params, exp_path):
        import cv2

        layout_width, layout_height = (
            scene_params["scene_box"][3:5] - scene_params["scene_box"][:2]
        )
        max_side_length = 1024
        layout_scale_factor = max_side_length / max(layout_height, layout_width)

        layout = np.zeros(
            (
                int(layout_scale_factor * layout_height),
                int(layout_scale_factor * layout_width),
                3,
            )
        )
        for object_args in self.objects_args:
            object_lb = (
                int(
                    layout_scale_factor
                    * (object_args.bbox[0].numpy() - scene_params["scene_box"][0])
                ),
                int(
                    layout_scale_factor
                    * (scene_params["scene_box"][4] - object_args.bbox[1].numpy())
                ),
            )
            object_rt = (
                int(
                    layout_scale_factor
                    * (object_args.bbox[3].numpy() - scene_params["scene_box"][0])
                ),
                int(
                    layout_scale_factor
                    * (scene_params["scene_box"][4] - object_args.bbox[4].numpy())
                ),
            )
            object_color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            cv2.rectangle(layout, object_lb, object_rt, color=object_color, thickness=2)
            cv2.putText(
                layout,
                f"{object_args.objectId}_{object_args.clas}",
                object_lb,
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 255, 255),
                1,
            )
        cv2.imwrite(str(exp_path) + "/layout.jpg", layout)

    def transform_SHs(self, shs, source_cameras_to_world):
        # https://github.com/szymanowiczs/splatter-image/blob/main/scene/gaussian_predictor.py
        # shs: N x SH_num x 3
        # source_cameras_to_world: 3 x 3
        assert shs.shape[2] == 3, "Can only process shs order 1"
        transforms = torch.bmm(
            self.sh_to_v_transform,
            source_cameras_to_world.unsqueeze(0),
        )
        transforms = torch.bmm(transforms, self.v_to_sh_transform)

        shs_transformed = torch.bmm(shs, transforms.expand(shs.shape[0], 3, 3))

        return shs_transformed

    def add_objects_to_scene(self, scene_cfg: dict, exp_path):
        self.init_sh_transform_matrices()
        self.gaussians_collection = {}
        if scene_cfg.scene_composition is not None:
            for obj in scene_cfg.scene_composition:
                logger.debug(f"Loading object from {obj.id}")
                save_dir = exp_path / "checkpoints"
                path = os.path.join(save_dir, obj.id + "_final_model.ply")
                tmp_ogs = GaussianModel(
                    {"sh_degree": scene_cfg.sh_degree, "path": path}, "tmp"
                )
                for transform_param in obj.params:
                    t_center = transform_param.center
                    origin_points = tmp_ogs._xyz.detach()
                    points_num = origin_points.shape[0]
                    transform_matrix_R, transform_matrix_S = (
                        self.create_transform_matrix_RS(
                            np.deg2rad(transform_param.rotation), transform_param.scale
                        )
                    )

                    transformed_xyz = (
                        transform_matrix_R
                        @ transform_matrix_S
                        @ origin_points.permute(1, 0)
                    )
                    z_min = transformed_xyz.min(dim=1)[0].detach()[2]
                    t_center[2] -= z_min.item()
                    transform_matrix_T = self.create_transform_matrix_T(t_center)
                    transformed_xyz = (
                        transformed_xyz
                        + transform_matrix_T.unsqueeze(1).repeat(1, points_num)
                    ).permute(1, 0)

                    transformed_scaling = tmp_ogs._scaling.detach() + torch.log(
                        torch.Tensor(transform_param.scale).float().cuda()
                    )
                    quaternion_R = matrix_to_quaternion(transform_matrix_R)

                    features_rest = tmp_ogs._features_rest.detach()
                    features_rest[:, :3, :] = self.transform_SHs(
                        features_rest[:, :3, :], transform_matrix_R
                    )

                    # calculate Wigner D matrix
                    if features_rest.shape[1] > 3:
                        rot_angles = o3._rotation.matrix_to_angles(
                            transform_matrix_R.cpu()
                        )
                        D_2 = o3.wigner_D(
                            2, rot_angles[0], rot_angles[1], rot_angles[2]
                        ).cuda()  # order2
                        features_rest[:, 3:8, :] = D_2 @ features_rest[:, 3:8, :]
                    if features_rest.shape[1] > 8:
                        D_3 = o3.wigner_D(
                            3, rot_angles[0], rot_angles[1], rot_angles[2]
                        ).cuda()  # order3
                        features_rest[:, 8:, :] = D_3 @ features_rest[:, 8:, :]

                    transformed_rotation = quaternion_raw_multiply(
                        a=quaternion_R.unsqueeze(0).expand(tmp_ogs._rotation.shape),
                        b=tmp_ogs._rotation.detach(),
                    )

                    xyz_min = transformed_xyz.min(dim=0)[0].detach()
                    xyz_max = transformed_xyz.max(dim=0)[0].detach()
                    self.scene_box[:3] = torch.min(self.scene_box[:3], xyz_min)
                    self.scene_box[3:] = torch.max(self.scene_box[3:], xyz_max)
                    object_args = ObjectArgs(
                        obj.id,
                        self.objects_count,
                        {
                            "T": torch.Tensor(t_center).float(),
                            "R": torch.Tensor(transform_param.rotation).float(),
                            "S": torch.Tensor(transform_param.scale).float(),
                        },
                        torch.concat([xyz_min, xyz_max]).cpu(),
                    )

                    self.gaussians_collection[f"{self.objects_count}_{obj.id}"] = (
                        ObjectGaussian(
                            f"{self.objects_count}_{obj.id}",
                            model=GaussianModel(scene_cfg, "scene", exp_path),
                            step=0,
                            text={
                                "text": scene_cfg.scene_text,
                                "negative_text": scene_cfg.negative_text,
                            },
                            image={},
                            cam_pose_method=scene_cfg.cam_pose_method,
                        )
                    )
                    gs_obj = self.gaussians_collection[f"{self.objects_count}_{obj.id}"]
                    gs_obj.model._xyz = transformed_xyz.cuda()
                    gs_obj.model._features_dc = tmp_ogs._features_dc.detach().cuda()
                    gs_obj.model._features_rest = features_rest.cuda()
                    gs_obj.model._scaling = transformed_scaling.cuda()
                    gs_obj.model._rotation = transformed_rotation.cuda()
                    gs_obj.model._opacity = tmp_ogs._opacity.detach().cuda()
                    gs_obj.model.max_radii2D = tmp_ogs.max_radii2D.detach().cuda()
                    gs_obj.model.xyz_gradient_accum = (
                        tmp_ogs.xyz_gradient_accum.detach().cuda()
                    )
                    gs_obj.model.denom = tmp_ogs.denom.detach().cuda()

                    self.objects_args.append(object_args)
                    self.objects_count += 1

                tmp_ogs.free_memory()
                tmp_ogs = None

        cfg_box = torch.zeros(6).cuda()
        cfg_box[3:] = torch.asarray(scene_cfg.radius)

        if scene_cfg.zero_ground:
            cfg_box[:2] = -cfg_box[3:5]
        else:
            cfg_box[:3] = -cfg_box[3:]
        if scene_cfg.cam_pose_method == "indoor":
            self.scene_box[:3] = torch.min(self.scene_box[:3], cfg_box[:3])
            self.scene_box[3:] = torch.max(self.scene_box[3:], cfg_box[3:])
        elif scene_cfg.cam_pose_method == "outdoor":
            self.scene_box[:3] = torch.min(self.scene_box[:3], cfg_box[:3])
            self.scene_box[3:] = torch.max(self.scene_box[3:], cfg_box[3:])
        scene_params = {
            "cam_pose_method": scene_cfg.cam_pose_method,
            "text": scene_cfg.scene_text,
            "negative_text": scene_cfg.negative_text,
            "scene_box": self.scene_box.cpu().numpy(),
            "sh_degree": scene_cfg.sh_degree,
            "zero_ground": scene_cfg.zero_ground,
            "env_init_color": scene_cfg.env_init_color,
            "floor_init_color": scene_cfg.floor_init_color,
        }

        self.env_gaussian = GaussianModel(scene_params, "env")
        self.floor_gaussian = GaussianModel(scene_params, "floor")

        self.gaussians_collection["env"] = ObjectGaussian(
            scene_cfg.scene_name + "env",
            model=self.env_gaussian,
            step=0,
            text={
                "text": scene_cfg.scene_text,
                "negative_text": scene_cfg.negative_text,
            },
            image={},
            cam_pose_method=scene_cfg.cam_pose_method,
        )
        self.gaussians_collection["floor"] = ObjectGaussian(
            scene_cfg.scene_name + "floor",
            model=self.floor_gaussian,
            step=0,
            text={
                "text": scene_cfg.scene_text,
                "negative_text": scene_cfg.negative_text,
            },
            image={},
            cam_pose_method=scene_cfg.cam_pose_method,
        )
        self.export_layout(scene_params, exp_path)

    def create_transform_matrix_RS(self, rotation, scale):
        if len(rotation) == 3:
            rx, ry, rz = rotation
            rotation_matrix = (
                euler_angles_to_matrix(torch.Tensor([rx, ry, rz]), "XYZ").float().cuda()
            )
        elif len(rotation) == 4:
            qw, qx, qy, qz = rotation
            rotation_matrix = (
                quaternion_to_matrix(torch.Tensor([qw, qx, qy, qz])).float().cuda()
            )

        if len(scale) == 3:
            scale_matrix = (
                torch.tensor(
                    [[scale[0], 0.0, 0.0], [0.0, scale[1], 0.0], [0.0, 0.0, scale[2]]]
                )
                .float()
                .cuda()
            )
        else:
            scale_matrix = (
                torch.tensor(
                    [
                        [scale[0], 0.0, 0.0],
                        [0.0, scale[0], 0.0],
                        [0.0, 0.0, scale[0]],
                    ]
                )
                .float()
                .cuda()
            )

        return rotation_matrix, scale_matrix

    def create_transform_matrix_T(self, translation):
        translation_matrix = torch.Tensor(translation).float().cuda()
        return translation_matrix

    def final_combine_all(self):
        # Combine all objects and scene into a GaussianModel
        tmp = {"_xyz":[], "_features_dc":[], "_features_rest":[], "_scaling":[], "_rotation":[], "_opacity":[], "max_radii2D":[], "xyz_gradient_accum":[], "denom":[]}
        max_sh_degree = 0
        for gs_obj in self.gaussians_collection.keys():
            tmp["_xyz"].append(self.gaussians_collection[gs_obj].model._xyz.detach())
            tmp["_features_dc"].append(self.gaussians_collection[gs_obj].model._features_dc.detach())
            tmp["_features_rest"].append(self.gaussians_collection[gs_obj].model._features_rest.detach())
            tmp["_scaling"].append(self.gaussians_collection[gs_obj].model._scaling.detach())
            tmp["_rotation"].append(self.gaussians_collection[gs_obj].model._rotation.detach())
            tmp["_opacity"].append(self.gaussians_collection[gs_obj].model._opacity.detach())
            tmp["max_radii2D"].append(self.gaussians_collection[gs_obj].model.max_radii2D.detach())
            tmp["xyz_gradient_accum"].append(self.gaussians_collection[gs_obj].model.xyz_gradient_accum.detach())
            tmp["denom"].append(self.gaussians_collection[gs_obj].model.denom.detach())
            max_sh_degree = max(max_sh_degree, self.gaussians_collection[gs_obj].model.max_sh_degree)
        final_gs = GaussianModel({"sh_degree":max_sh_degree}, "scene")
        final_gs._xyz = torch.cat(tmp["_xyz"])
        final_gs._features_dc = torch.cat(tmp["_features_dc"])
        final_gs._features_rest = torch.cat(tmp["_features_rest"])
        final_gs._scaling = torch.cat(tmp["_scaling"])
        final_gs._rotation = torch.cat(tmp["_rotation"])
        final_gs._opacity = torch.cat(tmp["_opacity"])
        final_gs.max_radii2D = torch.cat(tmp["max_radii2D"])
        final_gs.xyz_gradient_accum = torch.cat(tmp["xyz_gradient_accum"])
        final_gs.denom = torch.cat(tmp["denom"])
        return final_gs

    def score_render(
        self,
        object_gs: GaussianModel,
        viewpoint_camera: RCamera,
        bg_color: torch.Tensor,
        scaling_modifier: float =1.0,
        black_video: bool =False,
        override_color: torch.Tensor =None,
        sh_deg_aug_ratio: float =0.1,
        bg_aug_ratio: float =0.3,
        shs_aug_ratio: float =1.0,
        scale_aug_ratio: float =1.0,
        test: bool =True,
        compute_cov3D_python: bool = False,
        convert_SHs_python: bool = False,
    ):
        # Background tensor (bg_color) must be on GPU!
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                object_gs.get_xyz,
                dtype=object_gs.get_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except Exception as e:
            logger.error(e)

        if black_video:
            bg_color = torch.zeros_like(bg_color)
        # Aug
        act_SH = object_gs.active_sh_degree
        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=act_SH,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            score_flag=True,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = object_gs.get_xyz
        means2D = screenspace_points
        opacity = object_gs.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = object_gs.get_covariance(scaling_modifier)
        else:
            scales = object_gs.get_scaling
            rotations = object_gs.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if convert_SHs_python:
                raw_rgb = (
                    object_gs.get_features.transpose(1, 2)
                    .view(-1, 3, (object_gs.max_sh_degree + 1) ** 2)
                    .squeeze()[:, :3]
                )
                rgb = torch.sigmoid(raw_rgb)
                colors_precomp = rgb
            else:
                shs = object_gs.get_features
        else:
            colors_precomp = override_color
        # Rasterize visible Gaussians to image, obtain their radii (on screen).

        important_score, rendered_image, radii, depth_alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        depth, alpha = torch.chunk(depth_alpha, 2)

        focal = 1 / (2 * math.tan(viewpoint_camera.FoVx / 2))
        disp = focal / (depth + (alpha * 10) + 1e-5)

        try:
            min_d = disp[alpha <= 0.1].min()
        except Exception:
            min_d = disp.min()

        disp = torch.clamp((disp - min_d) / (disp.max() - min_d), 0.0, 1.0)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "image": rendered_image,
            "depth": disp,
            "alpha": alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "scales": scales,
            "important_score": important_score,
        }

    def scene_render(
        self,
        visible_gaussians: list,
        viewpoint_camera: RCamera,
        bg_color: torch.Tensor,
        scaling_modifier: float = 1.0,
        black_video: bool = False,
        override_color: torch.Tensor = None,
        sh_deg_aug_ratio: float = 0.1,
        bg_aug_ratio: float = 0.3,
        shs_aug_ratio: float = 1.0,
        scale_aug_ratio: float = 1.0,
        test: bool =False,
        compute_cov3D_python: bool =False,
        convert_SHs_python: bool =False,
        no_grad: bool =False,
    ):
        """
        Render the scene.
        Background tensor (bg_color) must be on GPU!
        """
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                torch.cat(
                    list(
                        map(
                            lambda visible_gaussian: self.gaussians_collection[
                                visible_gaussian
                            ].model.get_xyz,
                            visible_gaussians,
                        )
                    )
                ),
                dtype=self.env_gaussian.get_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )

        if not no_grad:
            try:
                screenspace_points.retain_grad()
            except Exception as e:
                logger.error(e)

        if black_video:
            bg_color = torch.zeros_like(bg_color)
        # Aug
        if random.random() < sh_deg_aug_ratio and not test:
            act_SH = 0
        else:
            act_SH = self.env_gaussian.active_sh_degree

        if random.random() < bg_aug_ratio and not test:
            if random.random() < 0.5:
                bg_color = torch.rand_like(bg_color)
            else:
                bg_color = torch.zeros_like(bg_color)

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=act_SH,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            score_flag=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        means3D = torch.cat(
            list(
                map(
                    lambda visible_gaussian: self.gaussians_collection[
                        visible_gaussian
                    ].model.get_xyz,
                    visible_gaussians,
                )
            )
        )
        means2D = screenspace_points
        opacity = torch.cat(
            list(
                map(
                    lambda visible_gaussian: self.gaussians_collection[
                        visible_gaussian
                    ].model.get_opacity,
                    visible_gaussians,
                )
            )
        )

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = torch.cat(
                list(
                    map(
                        lambda visible_gaussian: self.gaussians_collection[
                            visible_gaussian
                        ].model.get_covariance(scaling_modifier),
                        visible_gaussians,
                    )
                )
            )
        else:
            scales = torch.cat(
                list(
                    map(
                        lambda visible_gaussian: self.gaussians_collection[
                            visible_gaussian
                        ].model.get_scaling,
                        visible_gaussians,
                    )
                )
            )
            rotations = torch.cat(
                list(
                    map(
                        lambda visible_gaussian: self.gaussians_collection[
                            visible_gaussian
                        ].model.get_rotation,
                        visible_gaussians,
                    )
                )
            )

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if convert_SHs_python:
                raw_rgb = (
                    torch.cat(
                        list(
                            map(
                                lambda visible_gaussian: self.gaussians_collection[
                                    visible_gaussian
                                ].model.get_features,
                                visible_gaussians,
                            )
                        )
                    )
                    .transpose(1, 2)
                    .view(-1, 3, (self.env_gaussian.max_sh_degree + 1) ** 2)
                    .squeeze()[:, :3]
                )
                rgb = torch.sigmoid(raw_rgb)
                colors_precomp = rgb
            else:
                shs = torch.cat(
                    list(
                        map(
                            lambda visible_gaussian: self.gaussians_collection[
                                visible_gaussian
                            ].model.get_features,
                            visible_gaussians,
                        )
                    )
                )
        else:
            colors_precomp = override_color

        if random.random() < shs_aug_ratio and not test:
            variance = (0.2**0.5) * shs
            shs = shs + (torch.randn_like(shs) * variance)

        # add noise to scales
        if random.random() < scale_aug_ratio and not test:
            variance = (0.2**0.5) * scales / 4
            scales = torch.clamp(scales + (torch.randn_like(scales) * variance), 0.0)

        # Rasterize visible Gaussians to image, obtain their radii (on screen).

        rendered_image, radii, depth_alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
        depth, alpha = torch.chunk(depth_alpha, 2)

        focal = 1 / (2 * math.tan(viewpoint_camera.FoVx / 2))
        disp = focal / (depth + (alpha * 10) + 1e-5)

        try:
            min_d = disp[alpha <= 0.1].min()
        except Exception:
            min_d = disp.min()

        disp = torch.clamp((disp - min_d) / (disp.max() - min_d), 0.0, 1.0)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "image": rendered_image,
            "depth": disp,
            "alpha": alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "scales": scales,
        }

    def object_render(
        self,
        object_gs: GaussianModel,
        viewpoint_camera: RCamera,
        bg_color: torch.Tensor,
        scaling_modifier: float = 1.0,
        black_video: bool = False,
        override_color: torch.Tensor = None,
        sh_deg_aug_ratio: float = 0.1,
        bg_aug_ratio: float = 0.3,
        shs_aug_ratio: float = 1.0,
        scale_aug_ratio: float = 1.0,
        test: bool = False,
        compute_cov3D_python: bool = False,
        convert_SHs_python: bool = False,
        no_grad: bool = False,
    ):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                object_gs.get_xyz,
                dtype=object_gs.get_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )

        if not no_grad:
            try:
                screenspace_points.retain_grad()
            except Exception as e:
                logger.error(e)

        if black_video:
            bg_color = torch.zeros_like(bg_color)

        if random.random() < sh_deg_aug_ratio and not test:
            act_SH = 0
        else:
            act_SH = object_gs.active_sh_degree

        if random.random() < bg_aug_ratio and not test:
            if random.random() < 0.5:
                bg_color = torch.rand_like(bg_color)
            else:
                bg_color = torch.zeros_like(bg_color)

        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=act_SH,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            score_flag=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = object_gs.get_xyz
        means2D = screenspace_points
        opacity = object_gs.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = object_gs.get_covariance(scaling_modifier)
        else:
            scales = object_gs.get_scaling
            rotations = object_gs.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if convert_SHs_python:
                raw_rgb = (
                    object_gs.get_features.transpose(1, 2)
                    .view(-1, 3, (object_gs.max_sh_degree + 1) ** 2)
                    .squeeze()[:, :3]
                )
                rgb = torch.sigmoid(raw_rgb)
                colors_precomp = rgb
            else:
                shs = object_gs.get_features
        else:
            colors_precomp = override_color

        if random.random() < shs_aug_ratio and not test:
            variance = (0.2**0.5) * shs
            shs = shs + (torch.randn_like(shs) * variance)

        # add noise to scales
        if random.random() < scale_aug_ratio and not test:
            variance = (0.2**0.5) * scales / 4
            scales = torch.clamp(scales + (torch.randn_like(scales) * variance), 0.0)

        # Rasterize visible Gaussians to image, obtain their radii (on screen).

        rendered_image, radii, depth_alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        depth, alpha = torch.chunk(depth_alpha, 2)
        focal = 1 / (2 * math.tan(viewpoint_camera.FoVx / 2))
        disp = focal / (depth + (alpha * 10) + 1e-5)

        try:
            min_d = disp[alpha <= 0.1].min()
        except Exception:
            min_d = disp.min()

        disp = torch.clamp((disp - min_d) / (disp.max() - min_d), 0.0, 1.0)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "image": rendered_image,
            "depth": disp,
            "alpha": alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "scales": scales,
        }

def calculate_v_imp_score(
    gaussians: GaussianModel, imp_list: torch.Tensor, v_pow: float
) -> torch.Tensor:
    """
    :param gaussians: A data structure containing Gaussian components with a get_scaling method.
    :param imp_list: The importance scores for each Gaussian component.
    :param v_pow: The power to which the volume ratios are raised.
    :return: A list of adjusted values (v_list) used for pruning.
    """
    volume = torch.prod(gaussians.get_scaling, dim=1)
    index = int(len(volume) * 0.9)
    sorted_volume, _ = torch.sort(volume, descending=True)
    kth_percent_largest = sorted_volume[index]
    v_list = torch.pow(volume / kth_percent_largest, v_pow)
    v_list = v_list * imp_list
    return v_list

def prune_list(
    object_gs: GaussianModel,
    renderer: SceneGaussian,
    pose_args: omegaconf.dictconfig.DictConfig,
    bg_color: torch.Tensor,
) -> torch.Tensor:

    imp_list = None
    viewpoint_cams = loadSphereCam(pose_args)
    for viewpoint_cam in viewpoint_cams:
        render_pkg = renderer.score_render(object_gs, viewpoint_cam, bg_color)
        if imp_list is None:
            imp_list = render_pkg["important_score"]
        else:
            imp_list += render_pkg["important_score"].detach()
            gc.collect()
    return imp_list

def gaussian_filtering(
    object_gs: GaussianModel,
    renderer: SceneGaussian,
    pose_args: omegaconf.dictconfig.DictConfig,
    bg_color: torch.Tensor,
    v_pow: float,
    prune_decay: float,
    prune_percent: float,
) -> None:
    """3D Gaussian Filtering"""

    pcn_0 = object_gs.get_xyz.shape[0]
    imp_list = prune_list(object_gs, renderer, pose_args, bg_color)
    prune_decay_i = 1
    v_list = calculate_v_imp_score(object_gs, imp_list, v_pow)
    object_gs.prune_gaussians((prune_decay**prune_decay_i) * prune_percent, v_list)
    pcn_1 = object_gs.get_xyz.shape[0]
    logger.debug(
        "Point Number Changed From {} to {} After {}",
        pcn_0,
        pcn_1,
        "3D Gaussian Filtering",
    )

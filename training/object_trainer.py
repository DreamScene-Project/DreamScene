import gc
import os
import random
from pathlib import Path
from typing import Tuple

import imageio
import numpy as np
import torch
import tqdm
from loguru import logger
from torchvision.utils import save_image

from scene_gaussian import ObjectGaussian, SceneGaussian, gaussian_filtering
from utils.cam_utils import (_get_dir_ind, loadClipCam, loadRandomCam,
                             loadRandomCamAvoidMultiFace_4p, loadRecoCam)
from utils.system_utils import l2_loss, make_path, tv_loss

class ObjectTrainer:
    def __init__(self, cfg, renderer=None, id=None, gs_obj=None, device=None) -> None:
        self.pose_args = cfg.generateCamParams
        self.pose_args_recon = cfg.generateCamParams
        self.guidance_opt = cfg.guidanceParams
        self.dataset_args = cfg.modelParams
        self.optimizationParams = cfg.optimizationParams
        self.reconOptimizationParams = cfg.reconOptimizationParams
        self.mode_args = cfg.mode_args

        # avoid multi face
        if "avoid_multi_face" in self.mode_args.keys():
            self.avoid_multi_face = self.mode_args.avoid_multi_face
        else:
            self.avoid_multi_face = None

        exp_root = Path("experiments/")
        exp_path = make_path(Path(exp_root / cfg.log.exp_name))
        self.ckpt_path = make_path(exp_path / "checkpoints")
        self.train_renders_path = make_path(exp_path / "vis" / "train")
        self.eval_renders_path = make_path(exp_path / "vis" / "eval")
        self.train_steps = 1

        if renderer:
            # Inherit renderer from SceneTrainer
            self.id = id
            self.gs_obj = gs_obj
            self.renderer = renderer
            self.device = device
            self.train_steps = 1
        else:
            self.device = "cuda"
            self.id = cfg.objectParams.id
            # Set renderer for single object
            renderer = SceneGaussian(cfg=cfg)
            renderer.init_object_gaussian(self.id, cfg.objectParams)
            gs_obj = renderer.object_gaussians_dict[self.id]
            self.gs_obj = gs_obj
            self.renderer = renderer

    def seed_everything(self, seed):
        try:
            seed = int(seed)
        except Exception:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    @torch.no_grad()
    def save_model(self, step: int):
        save_dir = make_path(self.ckpt_path)
        path = os.path.join(save_dir, f"{self.id}_{step}_model.ply")
        self.gs_obj.model.save_ply(path)
        logger.debug(f"[INFO] save ply model to {path}.")

    def video_inference(self, step: int, gs_obj: ObjectGaussian, save_folder: str) -> None:
        """generate video"""
        video_cameras = loadClipCam(self.pose_args)
        bg_color = torch.tensor(
            [1, 1, 1] if self.dataset_args._white_background else [0, 0, 0],
            dtype=torch.float32,
            device="cuda",
        )
        img_frames = []
        depth_frames = []
        for viewpoint in video_cameras:
            out = self.renderer.object_render(gs_obj.model, viewpoint, bg_color=bg_color, test=True)
            rgb, depth = out["image"], out["depth"]
            if depth is not None:
                depth_norm = depth / depth.max()
                depths = torch.clamp(depth_norm, 0.0, 1.0)
                depths = depths.detach().cpu().permute(1, 2, 0).numpy()
                depths = (depths * 255).round().astype(np.uint8)
                depth_frames.append(depths)
            image = torch.clamp(rgb, 0.0, 1.0)
            image = image.detach().cpu().permute(1, 2, 0).numpy()
            image = (image * 255).round().astype(np.uint8)
            img_frames.append(image)
            # Img to Numpy
        imageio.mimwrite(
            os.path.join(save_folder, f"video_rgb_{gs_obj.id}_{step}.mp4"),
            img_frames,
            fps=30,
            quality=8,
        )
        if len(depth_frames) > 0:
            imageio.mimwrite(
                os.path.join(save_folder, f"video_depth_{gs_obj.id}_{step}.mp4"),
                depth_frames,
                fps=30,
                quality=8,
            )
        logger.debug("[ITER {}] Video Save Done!".format(step))

    def save_recon_img(self, path, pred_rgb, gt_image):
        save_image([pred_rgb, gt_image], path)

    def prepare_train(self) -> None:
        """load text-to-image to guide 3D representation generation"""
        from guidance.multitime_sd_utils import StableDiffusion

        logger.debug("[INFO] loading SD GUIDANCE...")
        self.guidance = StableDiffusion(
            self.device,
            self.guidance_opt.fp16,
            self.guidance_opt.vram_O,
            self.guidance_opt.t_range,
            self.guidance_opt.max_t_range,
            num_train_timesteps=self.guidance_opt.num_train_timesteps,
            textual_inversion_path=self.guidance_opt.textual_inversion_path,
            LoRA_path=self.guidance_opt.LoRA_path,
            guidance_opt=self.guidance_opt,
        )
        if self.guidance is not None:
            for p in self.guidance.parameters():
                p.requires_grad = False
        logger.debug("[INFO] loaded SD GUIDANCE!")
        self.set_embeds()

    @torch.no_grad()
    def set_embeds(self):
        """Caculate embeddings of positive text and negativae_text"""

        gs_obj = self.gs_obj
        gs_obj.text["text_embeddings"] = self.calc_text_embeddings(gs_obj.text["text"], gs_obj.text["negative_text"])

    def calc_text_embeddings(self, ref_text: str, negative_text: str = ""):
        """we caculate text embeddings following CSD(Classifier Score Distillation)"""
        embeddings = {}

        style_prompt = self.optimizationParams.style_prompt
        style_negative_prompt = self.optimizationParams.style_negative_prompt
        embeddings["default"] = self.guidance.get_text_embeds([ref_text + ", " + style_prompt])
        embeddings["uncond"] = self.guidance.get_text_embeds([negative_text + ", " + style_negative_prompt])

        embeddings["inverse_text"] = self.guidance.get_text_embeds(self.guidance_opt.inverse_text)

        embeddings["text_embeddings_vd"] = {}
        embeddings["uncond_text_embeddings_vd"] = {}

        for d in ["front", "side", "back", "overhead", "bottom"]:
            embeddings["text_embeddings_vd"][d] = self.guidance.get_text_embeds([f"{ref_text}, {d} view, {style_prompt}"])

        for d, d_neg in zip(
            ["front", "side", "back", "overhead", "bottom"],
            [
                "side view, back view, overhead view",
                "front view, back view, overhead view",
                "front view, side view, overhead view",
                "front view, back view, side view",
                "front view, back view, side view, overhead view",
            ],
        ):
            embeddings["uncond_text_embeddings_vd"][d] = self.guidance.get_text_embeds([f"{negative_text}, {d_neg}, {style_negative_prompt}"])

        return embeddings

    def get_text_embeddings(
        self,
        obj_text_embeddings: dict,
        elevation: torch.Tensor,
        azimuth: torch.Tensor,
        camera_distances: torch.Tensor,
        view_dependent_prompting: bool = True,
        return_null_text_embeddings: bool = False,
    ) -> Tuple[torch.Tensor, list]:
        """Get text emdedding in different directions"""

        assert view_dependent_prompting, "Perp-Neg only works with view-dependent prompting"

        batch_size = elevation.shape[0]

        pos_text_embeddings = []
        uncond_text_embeddings = []

        idxs = []
        for ele, azi, dis in zip(elevation, azimuth, camera_distances):
            idx = _get_dir_ind(ele, azi, dis, distinguish_lr=True)
            idxs.append(idx)
            uncond_text_embeddings.append(obj_text_embeddings["uncond_text_embeddings_vd"][idx].squeeze(0))
            pos_text_embeddings.append(obj_text_embeddings["text_embeddings_vd"][idx].squeeze(0))
        if return_null_text_embeddings:
            text_embeddings = torch.cat(
                [
                    torch.stack(pos_text_embeddings, dim=0),
                    torch.stack(uncond_text_embeddings, dim=0),
                    obj_text_embeddings["inverse_text"].expand(batch_size, -1, -1),
                ],
                dim=0,
            )
        else:
            text_embeddings = torch.cat(
                [
                    torch.stack(pos_text_embeddings, dim=0),
                    torch.stack(uncond_text_embeddings, dim=0),
                ],
                dim=0,
            )

        return text_embeddings, idxs

    def train_step(self) -> None:
        """train one step by setting lr, camera poses, caculating FPS loss and compressing gaussian points"""

        gs_obj = self.gs_obj
        self.optimizer = gs_obj.model.optimizer
        optimParams = self.optimizationParams
        iters = optimParams.iterations

        for _ in range(self.train_steps):
            self.step += 1
            # update lr
            gs_obj.model.update_learning_rate(self.step)
            gs_obj.model.update_feature_learning_rate(self.step)
            gs_obj.model.update_rotation_learning_rate(self.step)
            gs_obj.model.update_scaling_learning_rate(self.step)

            if self.step % 500 == 0:
                gs_obj.model.oneupSHdegree()

            if not optimParams.use_progressive:
                if self.step >= optimParams.progressive_view_iter and self.step % optimParams.scale_up_cameras_iter == 0:
                    self.pose_args.fovy_range[0] = max(
                        self.pose_args.max_fovy_range[0],
                        self.pose_args.fovy_range[0] * optimParams.fovy_scale_up_factor[0],
                    )
                    self.pose_args.fovy_range[1] = min(
                        self.pose_args.max_fovy_range[1],
                        self.pose_args.fovy_range[1] * optimParams.fovy_scale_up_factor[1],
                    )

                    self.pose_args.radius_range[1] = max(
                        self.pose_args.max_radius_range[1],
                        self.pose_args.radius_range[1] * optimParams.scale_up_factor,
                    )
                    self.pose_args.radius_range[0] = max(
                        self.pose_args.max_radius_range[0],
                        self.pose_args.radius_range[0] * optimParams.scale_up_factor,
                    )

                    self.pose_args.theta_range[1] = min(
                        self.pose_args.max_theta_range[1],
                        self.pose_args.theta_range[1] * optimParams.phi_scale_up_factor,
                    )
                    self.pose_args.theta_range[0] = max(
                        self.pose_args.max_theta_range[0],
                        self.pose_args.theta_range[0] * 1 / optimParams.phi_scale_up_factor,
                    )
                    self.pose_args.phi_range[0] = max(
                        self.pose_args.max_phi_range[0],
                        self.pose_args.phi_range[0] * optimParams.phi_scale_up_factor,
                    )
                    self.pose_args.phi_range[1] = min(
                        self.pose_args.max_phi_range[1],
                        self.pose_args.phi_range[1] * optimParams.phi_scale_up_factor,
                    )

                    logger.debug(f"scale up theta_range to: {self.pose_args.theta_range}")
                    logger.debug(f"scale up radius_range to: {self.pose_args.radius_range}")
                    logger.debug(f"scale up phi_range to: {self.pose_args.phi_range}")
                    logger.debug(f"scale up fovy_range to: {self.pose_args.fovy_range}")

            loss = 0
            C_batch_size = self.guidance_opt.C_batch_size
            images = []
            depths = []
            alphas = []
            scales = []

            elevation = torch.zeros(C_batch_size, dtype=torch.float32, device=self.device)
            azimuth = torch.zeros(C_batch_size, dtype=torch.float32, device=self.device)
            camera_distances = torch.zeros(C_batch_size, dtype=torch.float32, device=self.device)

            if self.avoid_multi_face:
                viewpoint_cams = loadRandomCamAvoidMultiFace_4p(self.pose_args, float(self.step) / iters, SSAA=True, size=4)

            for i in range(C_batch_size):
                if self.avoid_multi_face:
                    viewpoint_cam = viewpoint_cams[i]
                else:
                    viewpoint_cam = loadRandomCam(self.pose_args, SSAA=True)

                elevation[i] = viewpoint_cam.delta_polar.item()
                azimuth[i] = viewpoint_cam.delta_azimuth.item()  # [-180, 180]
                camera_distances[i] = viewpoint_cam.delta_radius.item()  # [] - 3.5
                bg_color = torch.tensor(
                    [1, 1, 1] if self.dataset_args._white_background else [0, 0, 0],
                    dtype=torch.float32,
                    device="cuda",
                )
                out = self.renderer.object_render(
                    gs_obj.model,
                    viewpoint_cam,
                    bg_color=bg_color,
                    sh_deg_aug_ratio=self.dataset_args.sh_deg_aug_ratio,
                    bg_aug_ratio=self.dataset_args.bg_aug_ratio,
                    shs_aug_ratio=self.dataset_args.shs_aug_ratio,
                    scale_aug_ratio=self.dataset_args.scale_aug_ratio,
                )
                image, viewspace_point_tensor, visibility_filter, radii = (
                    out["image"],
                    out["viewspace_points"],
                    out["visibility_filter"],
                    out["radii"],
                )
                depth, alpha = out["depth"], out["alpha"]
                scales.append(out["scales"])
                images.append(image)
                depths.append(depth)
                alphas.append(alpha)

            images = torch.stack(images, dim=0)
            depths = torch.stack(depths, dim=0)
            alphas = torch.stack(alphas, dim=0)

            # Loss
            _aslatent = False
            use_control_net = False

            if self.step < optimParams.geo_iter or random.random() < optimParams.as_latent_ratio:
                _aslatent = True
            if self.step > optimParams.use_control_net_iter and (random.random() < self.guidance_opt.controlnet_ratio):
                use_control_net = True

            (text_embeddings, vds) = self.get_text_embeddings(
                gs_obj.text["text_embeddings"],
                elevation,
                azimuth,
                camera_distances,
                True,
                return_null_text_embeddings=True,
            )

            stage_step_rate = min(self.step / iters, 1.0)

            loss = self.guidance.train_step(
                text_embeddings,
                images,
                pred_depth=depths,
                pred_alpha=alphas,
                grad_scale=self.guidance_opt.lambda_guidance,
                use_control_net=use_control_net,
                save_folder=self.train_renders_path,
                iteration=self.step,
                stage_step_rate=stage_step_rate,
                resolution=(self.pose_args.image_h, self.pose_args.image_w),
                guidance_opt=self.guidance_opt,
                as_latent=_aslatent,
                vds=vds,
                obj_id=gs_obj.id,
            )

            scales = torch.stack(scales, dim=0)
            loss_scale = torch.mean(scales, dim=-1).mean()
            loss_tv = tv_loss(images) + tv_loss(depths)
            loss = loss + optimParams.lambda_tv * loss_tv + optimParams.lambda_scale * loss_scale
            loss.backward()

            # densify and prune
            if self.step < optimParams.densify_until_iter:
                gs_obj.model.max_radii2D[visibility_filter] = torch.max(
                    gs_obj.model.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                gs_obj.model.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.step >= optimParams.densify_from_iter and self.step % optimParams.densification_interval == 0:
                    pcn_0 = gs_obj.model.get_xyz.shape[0]
                    size_threshold = 20 if self.step > optimParams.opacity_reset_interval else None

                    if pcn_0 < optimParams.max_point_number:
                        gs_obj.model.densify_and_prune(
                            optimParams.densify_grad_threshold,
                            0.005,
                            self.renderer.cameras_extent,
                            size_threshold,
                        )
                        pcn_1 = gs_obj.model.get_xyz.shape[0]
                        logger.debug(
                            "Point Number Changed From {} to {} After {}",
                            pcn_0,
                            pcn_1,
                            "densify_and_prune",
                        )
                    else:
                        gs_obj.model.densify_and_prune(
                            optimParams.densify_grad_threshold,
                            0.005,
                            self.renderer.cameras_extent,
                            size_threshold,
                        )
                        pcn_1 = gs_obj.model.get_xyz.shape[0]
                        logger.debug(
                            "Point Number Changed From {} to {} After {}",
                            pcn_0,
                            pcn_1,
                            "densify_and_prune",
                        )

                        if self.step < 1500:
                            bg_color = torch.tensor(
                                ([1, 1, 1] if self.dataset_args._white_background else [0, 0, 0]),
                                dtype=torch.float32,
                                device="cuda",
                            )
                            gaussian_filtering(
                                gs_obj.model,
                                self.renderer,
                                self.pose_args,
                                bg_color,
                                self.mode_args.v_pow,
                                self.mode_args.prune_decay,
                                self.mode_args.prune_percent,
                            )

                if self.step % optimParams.opacity_reset_interval == 0:
                    gs_obj.model.reset_opacity()

            if self.step in [1500]:
                bg_color = torch.tensor(
                    [1, 1, 1] if self.dataset_args._white_background else [0, 0, 0],
                    dtype=torch.float32,
                    device="cuda",
                )
                gaussian_filtering(
                    gs_obj.model,
                    self.renderer,
                    self.pose_args,
                    bg_color,
                    self.mode_args.v_pow,
                    self.mode_args.prune_decay,
                    0.3,
                    # self.mode_args.prune_percent,
                )

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

    def refine_step(self) -> None:
        """Reconstructive Generation"""

        gs_obj = self.gs_obj
        self.optimizer = gs_obj.model.optimizer
        optimParams = self.reconOptimizationParams
        iters = optimParams.iterations
        stage1_max_step = self.optimizationParams.iterations

        for _ in range(self.train_steps):
            self.step += 1
            if self.gt_images is None:
                self.viewpoint_cams = loadRecoCam(self.pose_args, [4, 12, 14, 6], [100, 85, 75, 55], scale=0.9)
                self.gt_size = len(self.viewpoint_cams)

            self.bg_color = torch.tensor(
                [1, 1, 1] if self.dataset_args._white_background else [0, 0, 0],
                dtype=torch.float32,
                device="cuda",
            )

            # update lr
            gs_obj.model.update_learning_rate(self.step)
            gs_obj.model.update_feature_learning_rate(self.step)
            gs_obj.model.update_rotation_learning_rate(self.step)
            gs_obj.model.update_scaling_learning_rate(self.step)

            if self.step % 300 == 0:
                gs_obj.model.oneupSHdegree()

            if not optimParams.use_progressive:
                if self.step >= optimParams.progressive_view_iter and self.step % optimParams.scale_up_cameras_iter == 0:
                    self.pose_args.fovy_range[0] = max(
                        self.pose_args.max_fovy_range[0],
                        self.pose_args.fovy_range[0] * optimParams.fovy_scale_up_factor[0],
                    )
                    self.pose_args.fovy_range[1] = min(
                        self.pose_args.max_fovy_range[1],
                        self.pose_args.fovy_range[1] * optimParams.fovy_scale_up_factor[1],
                    )

                    self.pose_args.radius_range[1] = max(
                        self.pose_args.max_radius_range[1],
                        self.pose_args.radius_range[1] * optimParams.scale_up_factor,
                    )
                    self.pose_args.radius_range[0] = max(
                        self.pose_args.max_radius_range[0],
                        self.pose_args.radius_range[0] * optimParams.scale_up_factor,
                    )

                    self.pose_args.theta_range[1] = min(
                        self.pose_args.max_theta_range[1],
                        self.pose_args.theta_range[1] * optimParams.phi_scale_up_factor,
                    )
                    self.pose_args.theta_range[0] = max(
                        self.pose_args.max_theta_range[0],
                        self.pose_args.theta_range[0] * 1 / optimParams.phi_scale_up_factor,
                    )

                    # opt.reset_resnet_iter = max(500, opt.reset_resnet_iter // 1.25)
                    self.pose_args.phi_range[0] = max(
                        self.pose_args.max_phi_range[0],
                        self.pose_args.phi_range[0] * optimParams.phi_scale_up_factor,
                    )
                    self.pose_args.phi_range[1] = min(
                        self.pose_args.max_phi_range[1],
                        self.pose_args.phi_range[1] * optimParams.phi_scale_up_factor,
                    )

                    logger.debug(f"scale up theta_range to: {self.pose_args.theta_range}")
                    logger.debug(f"scale up radius_range to: {self.pose_args.radius_range}")
                    logger.debug(f"scale up phi_range to: {self.pose_args.phi_range}")
                    logger.debug(f"scale up fovy_range to: {self.pose_args.fovy_range}")

            loss = 0

            images = []
            depths = []
            alphas = []
            scales = []

            REC_batch_size = self.gt_size // 2
            optimParams.densify_until_iter = int(iters * REC_batch_size * 0.8)
            step_size = 4
            stage_step_rate = min((self.step) / (iters), 1.0)
            elevation = torch.zeros(self.gt_size, dtype=torch.float32, device=self.device)
            azimuth = torch.zeros(self.gt_size, dtype=torch.float32, device=self.device)
            camera_distances = torch.zeros(self.gt_size, dtype=torch.float32, device=self.device)
            if self.gt_images is None:
                for i in range(self.gt_size):
                    viewpoint_cam = self.viewpoint_cams[i]
                    # viewpoint_cam
                    elevation[i] = viewpoint_cam.delta_polar.item()
                    azimuth[i] = viewpoint_cam.delta_azimuth.item()  # [-180, 180]
                    camera_distances[i] = viewpoint_cam.delta_radius.item()  # [] - 3.5
                    with torch.no_grad():
                        out = self.renderer.object_render(
                            gs_obj.model,
                            viewpoint_cam,
                            bg_color=self.bg_color,
                            sh_deg_aug_ratio=self.dataset_args.sh_deg_aug_ratio,
                            bg_aug_ratio=self.dataset_args.bg_aug_ratio,
                            shs_aug_ratio=self.dataset_args.shs_aug_ratio,
                            scale_aug_ratio=self.dataset_args.scale_aug_ratio,
                            test=True,
                            no_grad=True,
                        )
                        image, viewspace_point_tensor, visibility_filter, radii = (
                            out["image"],
                            out["viewspace_points"],
                            out["visibility_filter"],
                            out["radii"],
                        )
                        depth, alpha = out["depth"], out["alpha"]
                        scales.append(out["scales"])
                        images.append(image)
                        depths.append(depth)
                        alphas.append(alpha)

                images = torch.stack(images, dim=0)
                depths = torch.stack(depths, dim=0)
                alphas = torch.stack(alphas, dim=0)
                # Loss
                _aslatent = False
                use_control_net = False

                (text_embeddings, vds) = self.get_text_embeddings(
                    gs_obj.text["text_embeddings"],
                    elevation,
                    azimuth,
                    camera_distances,
                    True,
                    return_null_text_embeddings=True,
                )

                self.gt_images = []
                logger.debug(torch.cuda.memory_allocated())

                for j in range(0, self.gt_size // 4 * 4, step_size):
                    gt_image_ = self.guidance.train_step_gt(
                        text_embeddings[3 * j : 3 * (j + step_size)],
                        images[j : (j + step_size)],
                        pred_depth=depths[j : (j + step_size)],
                        pred_alpha=alphas[j : (j + step_size)],
                        grad_scale=self.guidance_opt.lambda_guidance,
                        use_control_net=use_control_net,
                        save_folder=self.train_renders_path,
                        iteration=self.step + stage1_max_step,
                        stage_step_rate=stage_step_rate,
                        resolution=(
                            self.pose_args_recon.image_h,
                            self.pose_args_recon.image_w,
                        ),
                        guidance_opt=self.guidance_opt,
                        as_latent=_aslatent,
                        vds=vds[j : (j + step_size)],
                        obj_id=gs_obj.id,
                        gid=j,
                    )
                    self.gt_images += gt_image_.clone().detach()
                self.gt_depths = depths.clone().detach()

            for i in range(REC_batch_size):
                out = self.renderer.object_render(
                    gs_obj.model,
                    self.viewpoint_cams[i],
                    bg_color=self.bg_color,
                    # black_video=self.step % 2,
                    sh_deg_aug_ratio=self.dataset_args.sh_deg_aug_ratio,
                    bg_aug_ratio=self.dataset_args.bg_aug_ratio,
                    shs_aug_ratio=self.dataset_args.shs_aug_ratio,
                    scale_aug_ratio=self.dataset_args.scale_aug_ratio,
                    test=True,
                )
                image, viewspace_point_tensor, visibility_filter, radii = (
                    out["image"].to(torch.float16),
                    out["viewspace_points"],
                    out["visibility_filter"],
                    out["radii"],
                )
                depth = out["depth"]

                Ll1 = l2_loss(image, self.gt_images[i])
                lambda_dssim = 0.0
                loss = (1.0 - lambda_dssim) * Ll1
                loss *= 100

                self.rec_count += 1

                loss.backward()
                if self.rec_count % 100 == 0:
                    path = self.eval_renders_path / f"{self.rec_count}.jpg"
                    self.save_recon_img(path, image, self.gt_images[i])
                if self.rec_count < optimParams.densify_until_iter:
                    gs_obj.model.max_radii2D[visibility_filter] = torch.max(
                        gs_obj.model.max_radii2D[visibility_filter],
                        radii[visibility_filter],
                    )
                    gs_obj.model.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    if self.rec_count % optimParams.densification_interval == 0:
                        pcn_0 = gs_obj.model.get_xyz.shape[0]
                        size_threshold = 20 if self.rec_count > optimParams.opacity_reset_interval else None
                        gs_obj.model.densify_and_prune(
                            optimParams.densify_grad_threshold,
                            0.005,
                            self.renderer.cameras_extent,
                            size_threshold,
                        )
                        if gs_obj.model.get_xyz.shape[0] > optimParams.max_point_number and self.step < 25:
                            gaussian_filtering(
                                gs_obj.model,
                                self.renderer,
                                self.pose_args,
                                self.bg_color,
                                self.mode_args.v_pow,
                                self.mode_args.prune_decay,
                                self.mode_args.prune_percent,
                            )
                        pcn_1 = gs_obj.model.get_xyz.shape[0]
                        logger.debug(
                            "Point Number Changed From {} to {} After {}",
                            pcn_0,
                            pcn_1,
                            "densify_and_prune",
                        )
                    if self.rec_count % optimParams.opacity_reset_interval == 0:
                        gs_obj.model.reset_opacity()

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

    def train(self) -> None:
        """traing process"""
        id = self.id
        logger.info(f"Start Training Object: {id}")

        path = os.path.join(self.ckpt_path, f"{id}_final_model.ply")
        if os.path.exists(path):
            return

        self.prepare_train()
        self.step = self.gs_obj.step
        self.rec_count = 0
        object_iters = self.optimizationParams.iterations
        if not self.reconOptimizationParams.only_recon_stage:
            for i in tqdm.trange(object_iters):
                if self.step > i:
                    continue
                self.train_step()
                if self.step % 500 == 0:
                    self.video_inference(self.step, self.gs_obj, self.eval_renders_path)
            self.save_model(self.step)

        self.step = 0
        self.gs_obj.model.training_setup(self.reconOptimizationParams)
        self.gt_images = None
        refine_iters = self.reconOptimizationParams.iterations
        if self.reconOptimizationParams.only_recon_stage:
            self.video_inference("before_recon", self.gs_obj, self.eval_renders_path)
        # TODO:stage_range 和 jump_range，最好是放进config中
        self.guidance.stage_range = [140, 200]
        self.guidance.stage_range_step = self.guidance.stage_range[1] - self.guidance.stage_range[0]
        self.guidance.jump_range = [75, 150]
        for i in tqdm.trange(refine_iters):
            if self.step > (i + object_iters):
                continue
            self.refine_step()
            if (i + 1) % 20 == 0:
                self.video_inference(self.step + object_iters, self.gs_obj, self.eval_renders_path)
        self.video_inference(object_iters + refine_iters, self.gs_obj, self.eval_renders_path)
        self.save_model("final")
        self.gs_obj.model.free_memory()
        del self.guidance
        gc.collect()
        torch.cuda.empty_cache()

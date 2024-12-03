import os
import random
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
import tqdm
from loguru import logger
from omegaconf import OmegaConf
from torchvision.utils import save_image

from scene_gaussian import SceneGaussian
from training.object_trainer import ObjectTrainer
from utils.cam_utils import SceneCameraLoader, _get_dir_ind
from utils.system_utils import l2_loss, make_path, tv_loss


class SceneTrainer:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.pose_args = cfg.generateCamParams
        self.scene_pose_args = cfg.sceneGenerateCamParams
        self.guidance_opt = cfg.guidanceParams

        self.device = torch.device("cuda")
        gpu_nums = torch.cuda.device_count()
        total_memory = torch.cuda.get_device_properties(self.device).total_memory / (1000**3)
        if gpu_nums == 1 and self.guidance_opt.g_device == "cuda:1":
            self.guidance_opt.g_device = "cuda"
        elif gpu_nums >= 1 and self.guidance_opt.g_device == "cuda" and total_memory < 40:
            logger.info("Please set guidanceParams.g_device to 'cuda:1' for scene generation.")

        self.pipe = cfg.pipelineParams
        self.dataset_args = cfg.modelParams
        self.W = cfg.W
        self.H = cfg.H
        self.seed = cfg.seed  # "random"

        self.perp_neg_f_sb = (1, 0.5, -0.606)
        self.perp_neg_f_fsb = (1, 0.5, +0.967)
        self.perp_neg_f_fs = (
            4,
            0.5,
            -2.426,
        )
        self.perp_neg_f_sf = (4, 0.5, -2.426)
        self.seed_everything()

        # training stuff
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop

        # Make dirs
        self.exp_root = Path("experiments/")
        self.exp_path = make_path(Path(self.exp_root / self.cfg.log.exp_name))
        self.ckpt_path = make_path(self.exp_path / "checkpoints")
        self.scene_ckpt_path = make_path(self.exp_path / "scene_checkpoints")
        self.train_renders_path = make_path(self.exp_path / "vis" / "train")
        self.eval_renders_path = make_path(self.exp_path / "vis" / "eval")
        OmegaConf.save(self.cfg, self.exp_path / "config.yaml")

        self.init_logger()

        self.renderer = SceneGaussian(cfg=cfg)

        self.prepare_train()
        logger.debug(f"Successfully initialized {self.cfg.log.exp_name}")

    def seed_everything(self):
        try:
            seed = int(self.seed)
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

        self.last_seed = seed

    def prepare_train(self):
        self.step = 0
        self.renderer.init_gaussians()

        if self.cfg.scene_configs.objects is not None:
            self.num_object_gaussians = len(self.cfg.scene_configs.objects)
        else:
            self.num_object_gaussians = 0
        for i in range(self.num_object_gaussians):
            self.renderer.object_gaussians_dict[self.cfg.scene_configs.objects[i].id].text = {
                "text": self.cfg.scene_configs.objects[i].text,
                "negative_text": self.cfg.scene_configs.objects[i].negative_text,
            }

    def prepare_train_scene(self):
        import gc

        gc.collect()
        torch.cuda.empty_cache()

        self.step = 0
        self.cam_pose_method = self.cfg.scene_configs.scene.cam_pose_method
        self.renderer.init_gaussian_scene()

        logger.debug("[INFO] loading SD GUIDANCE...")
        if self.guidance_opt.guidance == "MTSD":
            from guidance.multitime_sd_utils import StableDiffusion
        self.guidance = StableDiffusion(
            self.guidance_opt.g_device,
            self.guidance_opt.fp16,
            self.guidance_opt.vram_O,
            self.guidance_opt.t_range,
            self.guidance_opt.max_t_range,
            num_train_timesteps=self.guidance_opt.num_train_timesteps,
            textual_inversion_path=self.guidance_opt.textual_inversion_path,
            guidance_opt=self.guidance_opt,
        )
        if self.guidance is not None:
            for p in self.guidance.parameters():
                p.requires_grad = False
        logger.debug("[INFO] loaded SD GUIDANCE!")
        with torch.no_grad():
            env_obj = self.renderer.gaussians_collection["env"]
            env_obj.text["text_embeddings"] = self.calc_text_embeddings(
                env_obj.text["text"], env_obj.text["negative_text"]
            )

    def calc_text_embeddings(self, ref_text, negative_text=""):
        embeddings = {}
        style_prompt = self.cfg.sceneOptimizationParams.style_prompt
        style_negative_prompt = self.cfg.sceneOptimizationParams.style_negative_prompt
        # text embeddings (stable-diffusion) and (IF)
        embeddings["default"] = self.guidance.get_text_embeds([ref_text + ", " + style_prompt])
        embeddings["uncond"] = self.guidance.get_text_embeds([negative_text + ", " + style_negative_prompt])
        embeddings["inverse_text"] = self.guidance.get_text_embeds(self.guidance_opt.inverse_text)
        embeddings["text_embeddings_vd"] = {}
        embeddings["uncond_text_embeddings_vd"] = {}
        for d in ["front", "side", "back", "overhead", "bottom"]:
            if self.renderer.gaussians_collection["env"].cam_pose_method == "outdoor":
                if d == "overhead":
                    embeddings["text_embeddings_vd"][d] = self.guidance.get_text_embeds(
                        [f"ground of {ref_text}, {style_prompt}"]
                    )
                elif d == "bottom":
                    embeddings["text_embeddings_vd"][d] = self.guidance.get_text_embeds(
                        [f"sky of {ref_text}, {style_prompt}"]
                    )
                else:
                    embeddings["text_embeddings_vd"][d] = self.guidance.get_text_embeds(
                        [f"{ref_text}, {d} view, {style_prompt}"]
                    )
            elif self.renderer.gaussians_collection["env"].cam_pose_method == "indoor":
                if d == "overhead":
                    embeddings["text_embeddings_vd"][d] = self.guidance.get_text_embeds(
                        [f"floor of {ref_text}, {style_prompt}"]
                    )
                elif d == "bottom":
                    embeddings["text_embeddings_vd"][d] = self.guidance.get_text_embeds(
                        [f"ceiling of {ref_text}, {style_prompt}"]
                    )
                else:
                    embeddings["text_embeddings_vd"][d] = self.guidance.get_text_embeds(
                        [f"{ref_text}, {d} view, {style_prompt}"]
                    )
                embeddings["text_embeddings_vd"][d] = self.guidance.get_text_embeds(
                    [f"{ref_text}, {d} view, {style_prompt}"]
                )
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
            embeddings["uncond_text_embeddings_vd"][d] = self.guidance.get_text_embeds(
                [f"{negative_text}, {d_neg}, {style_negative_prompt}"]
            )
        return embeddings

    def get_text_embeddings(
        self,
        obj_text_embeddings,
        elevation,
        azimuth,
        camera_distances,
        view_dependent_prompting: bool = True,
        return_null_text_embeddings: bool = False,
    ):
        batch_size = elevation.shape[0]
        if view_dependent_prompting:
            idxs = []
            pos_text_embeddings = []
            uncond_text_embeddings = []
            null_text_embeddings = []
            # Get direction
            for ele, azi, dis in zip(elevation, azimuth, camera_distances):
                if _get_dir_ind(ele, azi, dis, distinguish_lr=True) == "overhead":
                    idx = "overhead"
                    pos_text_embeddings.append(obj_text_embeddings["text_embeddings_vd"][idx])
                    uncond_text_embeddings.append(obj_text_embeddings["uncond_text_embeddings_vd"][idx])
                    null_text_embeddings.append(obj_text_embeddings["inverse_text"])
                else:
                    idx = "None"
                    pos_text_embeddings.append(obj_text_embeddings["default"])
                    uncond_text_embeddings.append(obj_text_embeddings["uncond"])
                    null_text_embeddings.append(obj_text_embeddings["inverse_text"])
                idxs.append(idx)
        else:
            idxs = ["None", "None", "None", "None"]
            pos_text_embeddings = obj_text_embeddings["default"].expand(batch_size, -1, -1)
            uncond_text_embeddings = obj_text_embeddings["uncond"].expand(batch_size, -1, -1)
            null_text_embeddings = obj_text_embeddings["inverse_text"].expand(batch_size, -1, -1)

        # IMPORTANT: we return (cond, uncond), which is in different order than other implementations!
        if return_null_text_embeddings:
            if view_dependent_prompting:
                text_embeddings = torch.cat(
                    [
                        torch.stack(pos_text_embeddings, dim=0),  # 4
                        torch.stack(uncond_text_embeddings, dim=0),  # 4
                        torch.stack(null_text_embeddings, dim=0),  # 4
                    ],
                    dim=0,
                )
            else:
                text_embeddings = torch.cat(
                    [pos_text_embeddings, uncond_text_embeddings, null_text_embeddings],
                    dim=0,
                )
        else:
            if view_dependent_prompting:
                text_embeddings = torch.cat(
                    [
                        torch.stack(pos_text_embeddings, dim=0),  # 4
                        torch.stack(uncond_text_embeddings, dim=0),  # 4
                    ],
                    dim=0,
                )
            else:
                text_embeddings = torch.cat([pos_text_embeddings, uncond_text_embeddings], dim=0)
        return text_embeddings, idxs

    def init_logger(self):
        logger.remove()  # Remove default logger
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
        logger.add(lambda msg: tqdm.tqdm.write(msg, end=""), colorize=True, format=log_format)
        logger.add(self.exp_path / "log.txt", colorize=False, format=log_format, level="DEBUG")
        logger.add(sys.stderr, level="INFO")

    def scene_video_inference(self, step, save_folder, only_env=False):
        img_frames = []
        depth_frames = []
        if only_env:
            visible_gaussians = ["floor", "env"]
        else:
            visible_gaussians = self.visible_gaussians
        for viewpoint in self.scene_cams_inference:
            out = self.renderer.scene_render(visible_gaussians, viewpoint, bg_color=self.bg_color, test=True)
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
            os.path.join(save_folder, "video_rgb_{}_{}.mp4".format("scene", step)),
            img_frames,
            fps=30,
            quality=8,
        )
        if len(depth_frames) > 0:
            imageio.mimwrite(
                os.path.join(save_folder, "video_depth_{}_{}.mp4".format("scene", step)),
                depth_frames,
                fps=30,
                quality=8,
            )
        logger.debug("\n[ITER {}] Video Save Done!".format(step))

    def scene_cams_record(self, step, save_folder, only_env=False, render_size=120):
        # Render cams before training
        # For debug.
        img_frames = []
        depth_frames = []
        if len(self.scene_cams) < render_size:
            render_size = len(self.scene_cams)
        if only_env:
            visible_gaussians = ["floor", "env"]
        else:
            visible_gaussians = self.visible_gaussians
        for viewpoint in self.scene_cams[:render_size]:
            out = self.renderer.scene_render(visible_gaussians, viewpoint, bg_color=self.bg_color, test=True, no_grad=True)
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
            os.path.join(save_folder, "record_video_rgb_{}_{}.mp4".format("scene", step)),
            img_frames,
            fps=30,
            quality=8,
        )
        if len(depth_frames) > 0:
            imageio.mimwrite(
                os.path.join(save_folder, "record_video_depth_{}_{}.mp4".format("scene", step)),
                depth_frames,
                fps=30,
                quality=8,
            )
        logger.debug("\n[ITER {}] Video Save Done!".format(step))

    def object_task(self, id):
        object_trainer = ObjectTrainer(
            self.cfg,
            self.renderer,
            id,
            self.renderer.object_gaussians_dict[id],
            f"cuda",
        )
        object_trainer.train()
        return "Success"

    def save_ckpt(self, path):
        torch.save(self.renderer.capture(), path)

    def load_ckpt(self, path):
        env_args, self.renderer.stage_n = torch.load(path)
        self.renderer.restore(env_args, self.cfg.reconSceneOptimizationParams)

    def scene_only_render(self, step_points=None, start_points=None, stop_points=None):
        # Render for inference
        self.scene_cams = []
        if self.cam_pose_method == "indoor":
            # living_room
            # start_points = [[-1,0,2.2], [0,1,2.2]]
            # stop_points = [[0.0,0,2.2], [0,-0.5,2.2]]
            start_points = [[-3.0, 0, 2.2], [1.5, 0.0, 2.2], [-1.0, 0.0, 2.2]]
            stop_points = [[1.5, 0, 2.2], [-1.0, 0.0, 2.2], [1.0, 1.0, 2.2]]

            # kitchen
            # start_points = [[-3.0, 0, 2.2], [1.5, 0.0, 2.2], [-1.0, 0.0, 2.2]]
            # stop_points  = [[ 1.5, 0, 2.2], [-1.0, 0.0, 2.2], [1.0, 1.0, 2.2]]

        elif self.cam_pose_method == "outdoor":
            # park
            # start_points = [[-3,-2,2.5], [4,-2,2.5], [0,-4,2.5]]
            # stop_points = [[3,-2,2.5], [-4,0,2.5], [0,-2,2.5]]
            # zoo
            start_points = [[-3, -2, 2.5], [4, -2, 2.5], [0, -4, 2.5]]
            stop_points = [[3, -2, 2.5], [-4, 0, 2.5], [0, -2, 2.5]]
        end_point = [0, 0, 0]
        line_n = 0

        for start_point, stop_point in zip(start_points, stop_points):
            self.scene_cams += self.cams_loader.Line(
                start_point,
                stop_point,
                0.1,
                img_h=512,
                img_w=512,
            )
            tmp_aff_params = {
                "T": torch.Tensor(stop_point),
                "R": torch.Tensor([0.0, 0.0, 0.0]),
                "S": torch.Tensor([1.0, 1.0, 1.0]),
            }
            start_phi = np.arctan2(start_point[0] - stop_point[0], start_point[1] - stop_point[1]) * 180 / np.pi
            line_n += 1
            if line_n == len(start_points):
                end_phi = (
                    np.arctan2(
                        stop_points[line_n - 1][0] - end_point[0],
                        stop_points[line_n - 1][1] - end_point[1],
                    )
                    * 180
                    / np.pi
                )
            else:
                end_phi = (
                    np.arctan2(
                        start_points[line_n][0] - stop_points[line_n][0],
                        start_points[line_n][1] - stop_points[line_n][1],
                    )
                    * 180
                    / np.pi
                )
            self.scene_cams += self.cams_loader.Circle2(
                start_phi=start_phi,
                end_phi=end_phi,
                affine_params=tmp_aff_params,
                circle_size=180,
                render45=False,
            )
        self.scene_cams += self.cams_loader.Circle3()
        self.scene_cams_record(
            "render",
            self.eval_renders_path,
            only_env=False,
            render_size=len(self.scene_cams),
        )
        return

    def train(self):
        logger.debug("DreamScene Training Starting.")
        for i in range(self.num_object_gaussians):
            self.object_task(self.cfg.scene_configs.objects[i].id)
        if self.cfg.reconOptimizationParams.only_recon_stage:
            # If you only need to refine through our reconstructive generation process.
            logger.debug("Fast Recon Completed.")
            return
        logger.debug("Scene Training Starting")
        self.prepare_train_scene()
        self.bg_color = torch.tensor(
            [1, 1, 1] if self.dataset_args._white_background else [0, 0, 0],
            dtype=torch.float32,
            device="cuda",
        )
        self.current_prev_n = 0
        objects_args = self.renderer.objects_args
        self.cams_loader = SceneCameraLoader(
            self.scene_pose_args, self.renderer.scene_box.cpu(), self.renderer.objects_args, self.cam_pose_method
        )
        self.scene_cams_inference = self.prepare_scene_cams()

        self.visible_gaussians = []
        for objects_arg in objects_args:
            self.visible_gaussians.append(f"{objects_arg.objectId}_{objects_arg.clas}")
        self.visible_gaussians.append("env")
        self.visible_gaussians.append("floor")

        if self.renderer.stage_n > 0:
            logger.info(f"Load Scene ckpt from Stage: {self.renderer.stage_n}")
        if self.cfg.only_render:
            self.scene_only_render()
            return

        self.n_stage1 = self.cfg.sceneOptimizationParams.iterations
        if self.renderer.stage_n == 0:
            logger.info("Start Stage-1")
            self.step = 0
            self.iters = self.n_stage1
            # Prepare camera poses for stage-1
            self.scene_cams = []
            self.scene_cams_max = self.n_stage1 * self.guidance_opt.C_batch_size
            self.scene_cams_mid = self.scene_cams_max * 0.7
            if self.cam_pose_method == "outdoor":
                while len(self.scene_cams) < self.scene_cams_max:
                    self.scene_cams += self.cams_loader.Stage1_Outdoor()
                    if len(self.scene_cams) > self.scene_cams_mid:
                        self.scene_cams += self.cams_loader.Stage1_Outdoor2()
            elif self.cam_pose_method == "indoor":
                object_count = 0
                while len(self.scene_cams) < self.scene_cams_max:
                    self.scene_cams += self.cams_loader.Stage1_Indoor()
                    if len(self.scene_cams) > self.scene_cams_mid:
                        if random.random() > 0.7:
                            try:
                                object_args = objects_args[object_count % self.renderer.objects_count]
                                self.scene_cams += self.cams_loader.Stage2_Indoor(
                                    affine_params=object_args.affine_params
                                )
                            except Exception:
                                logger.debug(
                                    f"A failure during sampling camera pose around: {object_args.clas}-{object_args.objectId}"
                                )
                            finally:
                                object_count += 1

            for i in tqdm.trange(self.n_stage1):
                if self.step > i:
                    continue
                self.scene_train_step("env", only_env=True if self.cam_pose_method == "outdoor" else False)
                # Visualize
                if (i + 1) % 300 == 0:
                    self.scene_video_inference(
                        self.step + self.current_prev_n,
                        self.eval_renders_path,
                        True if self.cam_pose_method == "outdoor" else False,
                    )
            self.scene_video_inference(self.step + self.current_prev_n, self.eval_renders_path)
            self.renderer.stage_n = 1
            self.save_ckpt(self.scene_ckpt_path / f"scene_{self.renderer.stage_n}_stage.ckpt")
        self.current_prev_n += self.n_stage1

        self.n_stage2 = self.cfg.sceneOptimizationParams.iterations - 300
        if self.renderer.stage_n == 1:
            logger.info("Start Stage-2")
            self.step = 0
            self.iters = self.n_stage2
            # Prepare camera poses for stage-2
            self.scene_cams = []
            self.scene_cams_max = self.n_stage2 * self.guidance_opt.C_batch_size
            object_count = 0
            if self.cam_pose_method == "outdoor":
                self.guidance.stage_range = [350, 800]
                self.guidance.stage_range_step = self.guidance.stage_range[1] - self.guidance.stage_range[0]
                self.guidance.jump_range = [150, 200]
                while len(self.scene_cams) < self.scene_cams_max:
                    self.scene_cams += self.cams_loader.Stage2_Outdoor()
            elif self.cam_pose_method == "indoor":
                while len(self.scene_cams) < self.scene_cams_max:
                    rcc = random.random()
                    if rcc < 0.25:
                        object_args = objects_args[object_count % self.renderer.objects_count]
                        affine_params = object_args.affine_params
                        try:
                            self.scene_cams += self.cams_loader.Stage2_Indoor(affine_params=affine_params)
                        except Exception:
                            logger.debug(
                                f"A failure during sampling camera pose around: {object_args.clas}-{object_args.objectId}"
                            )
                        finally:
                            object_count += 1
                    elif rcc < 0.75:
                        self.scene_cams += self.cams_loader.Stage2_Indoor()
                    else:
                        self.scene_cams += self.cams_loader.Stage1_Indoor(
                            size=8,
                            view_floor=True,
                        )

            self.guidance.stage_range = [350, 750]
            self.guidance.stage_range_step = self.guidance.stage_range[1] - self.guidance.stage_range[0]
            self.guidance.jump_range = [150, 200]
            self.renderer.floor_gaussian.training_setup(self.cfg.sceneOptimizationParams)
            for i in tqdm.trange(self.n_stage2):
                if self.cam_pose_method == "outdoor" and random.random() < 0.5:
                    only_env = True
                self.scene_train_step("floor", only_env=False)
                if (i + 1) % 200 == 0:
                    self.scene_video_inference(
                        self.step + self.current_prev_n,
                        self.eval_renders_path,
                        True if self.cam_pose_method == "outdoor" else False,
                    )
            self.scene_video_inference(self.step + self.current_prev_n, self.eval_renders_path)
            self.renderer.stage_n = 2
            self.save_ckpt(self.scene_ckpt_path / f"scene_{self.renderer.stage_n}_stage.ckpt")
        self.current_prev_n += self.n_stage2

        self.n_stage3 = 25
        if self.renderer.stage_n == 2:
            logger.info("Start Stage-3")
            self.step = 0
            self.iters = self.n_stage3

            # Prepare camera poses for stage-3
            self.scene_cams = []
            self.scene_cams_max = 20 * self.guidance_opt.C_batch_size
            object_count = 0
            if self.cam_pose_method == "outdoor":
                self.scene_cams = self.cams_loader.Stage3_Outdoor("env")
                while len(self.scene_cams) < self.scene_cams_max:
                    self.scene_cams += self.cams_loader.Stage2_Outdoor()
                self.scene_cams_floor = self.scene_cams
                random.shuffle(self.scene_cams_floor)
            elif self.cam_pose_method == "indoor":
                while len(self.scene_cams) < self.scene_cams_max:
                    rcc = random.random()
                    if rcc < 0.5:
                        self.scene_cams += self.cams_loader.Stage1_Indoor(
                            size=12,
                            view_floor=True,
                        )
                    else:
                        self.scene_cams += self.cams_loader.Stage2_Indoor(
                            idx=(i - object_count) % 12,
                            size=12,
                        )

            random.shuffle(self.scene_cams)
            self.gt_size = len(self.scene_cams) // 4 * 4
            self.gt_images = None
            self.gt_images_floor = None
            if self.cam_pose_method == "outdoor":
                scene_optim = False
                only_env = True
            elif self.cam_pose_method == "indoor":
                scene_optim = True
                only_env = False
                for object_key in self.visible_gaussians:
                    if object_key == "env" or object_key == "floor":
                        continue
                    self.renderer.gaussians_collection[object_key].model.training_setup(
                        self.cfg.fineSceneOptimizationParams
                    )
            self.renderer.env_gaussian.training_setup(self.cfg.reconSceneOptimizationParams)
            self.renderer.floor_gaussian.training_setup(self.cfg.reconSceneOptimizationParams)
            self.rec_count = 0
            self.guidance.stage_range = [140, 200]
            self.guidance.stage_range_step = self.guidance.stage_range[1] - self.guidance.stage_range[0]
            self.guidance.jump_range = [75, 150]
            for i in tqdm.trange(self.n_stage3):
                if self.cam_pose_method == "outdoor":
                    self.scene_refine_step_outdoor("floor", only_env=only_env, scene_optim=scene_optim)
                else:
                    self.scene_refine_step("all", only_env=only_env, scene_optim=scene_optim)
                if (i + 1) % 10 == 0:
                    self.scene_video_inference(self.step + self.current_prev_n, self.eval_renders_path)

            self.scene_video_inference(self.step + self.current_prev_n, self.eval_renders_path)
            self.renderer.stage_n = 3
        self.current_prev_n += self.n_stage3

        # Inference
        self.scene_cams = []

        # Customized camera poses for inference rendering
        # if self.cam_pose_method == "indoor":
        #     start_points = [[-3.0, 0, 2.2], [1.5, 0.0, 2.2], [-1.0, 0.0, 2.2]]
        #     stop_points = [[1.5, 0, 2.2], [-1.0, 0.0, 2.2], [1.0, 1.0, 2.2]]
        # elif self.cam_pose_method == "outdoor":
        #     start_points = [[-4, -2, 2.5], [4, -2, 2.5], [0, -4, 2.5]]
        #     stop_points = [[3, -2, 2.5], [-4, 0, 2.5], [0, 0, 2.5]]

        # for start_point, stop_point in zip(start_points, stop_points):
        #     self.scene_cams += self.cams_loader.Line(
        #         start_point,
        #         stop_point,
        #         0.1,
        #         img_h=512,
        #         img_w=512,
        #     )
        #     tmp_aff_params = {
        #         "T": torch.Tensor(stop_point),
        #         "R": torch.Tensor([0.0, 0.0, 0.0]),
        #         "S": torch.Tensor([1.0, 1.0, 1.0]),
        #     }
        #     start_phi = (
        #         np.arctan2(
        #             start_point[0] - stop_point[0], start_point[1] - stop_point[1]
        #         )
        #         * 180
        #         / np.pi
        #     )
        #     self.scene_cams += self.cams_loader.Circle2(
        #         start_phi=start_phi,
        #         affine_params=tmp_aff_params,
        #         render45=False,
        #     )

        # Generic camera poses for inference rendering
        for object_args in objects_args:
            self.scene_cams += self.cams_loader.Circle(
                affine_params=object_args.affine_params,
                render45=False,
            )
        self.scene_cams += self.cams_loader.Circle()
        self.scene_cams_record(
            "render",
            self.eval_renders_path,
            only_env=False,
            render_size=len(self.scene_cams),
        )
        # Combine all objects and scene into a GaussianModel to save it.
        with torch.no_grad():
            final_gs = self.renderer.final_combine_all()
            final_path = self.scene_ckpt_path / f"scene_final_model.ply"
            final_gs.save_ply(final_path)
            logger.debug(f"[INFO] save final ply model to {final_path}.")

    def prepare_scene_cams(self):
        viewpoint_cams = []
        objects_args = self.renderer.objects_args
        # objects
        for object_args in objects_args:
            viewpoint_cams += self.cams_loader.Circle(
                affine_params=object_args.affine_params,
            )
        # scene
        viewpoint_cams += self.cams_loader.Circle()
        return viewpoint_cams

    def scene_train_step(self, key_gs="all", only_env=False, scene_optim=False):
        if scene_optim:
            for object_key in self.visible_gaussians:
                self.renderer.gaussians_collection[object_key].model.active_grad()
        if key_gs == "env":
            env_gs = self.renderer.env_gaussian
            env_gs.active_grad()
            self.optimizer = env_gs.optimizer
            self.renderer.floor_gaussian.deactive_grad()
        elif key_gs == "floor":
            floor_gs = self.renderer.floor_gaussian
            floor_gs.active_grad()
            self.optimizer = floor_gs.optimizer
            self.renderer.env_gaussian.deactive_grad()
        elif key_gs == "all":
            env_gs = self.renderer.env_gaussian
            env_gs.active_grad()
            floor_gs = self.renderer.floor_gaussian
            floor_gs.active_grad()
            self.optimizer = env_gs.optimizer
            self.optimizer_floor = floor_gs.optimizer
        iters = self.iters
        optimParams = self.cfg.sceneOptimizationParams

        for _ in range(self.train_steps):
            self.step += 1
            # update lr
            if key_gs == "env" or key_gs == "all":
                env_gs.update_learning_rate(self.step)
                env_gs.update_feature_learning_rate(self.step)
                env_gs.update_rotation_learning_rate(self.step)
                env_gs.update_scaling_learning_rate(self.step)

                if self.step % 500 == 0:
                    env_gs.oneupSHdegree()

            if key_gs == "floor" or key_gs == "all":
                floor_gs.update_learning_rate(self.step)
                floor_gs.update_feature_learning_rate(self.step)
                floor_gs.update_rotation_learning_rate(self.step)
                floor_gs.update_scaling_learning_rate(self.step)

                if self.step % 500 == 0:
                    floor_gs.oneupSHdegree()

            if not optimParams.use_progressive:
                if (
                    self.step >= optimParams.progressive_view_iter
                    and self.step % optimParams.scale_up_cameras_iter == 0
                ):
                    self.scene_pose_args.fovy_range[0] = max(
                        self.scene_pose_args.max_fovy_range[0],
                        self.scene_pose_args.fovy_range[0] * optimParams.fovy_scale_up_factor[0],
                    )
                    self.scene_pose_args.fovy_range[1] = min(
                        self.scene_pose_args.max_fovy_range[1],
                        self.scene_pose_args.fovy_range[1] * optimParams.fovy_scale_up_factor[1],
                    )

                    self.scene_pose_args.radius_range[1] = max(
                        self.scene_pose_args.max_radius_range[1],
                        self.scene_pose_args.radius_range[1] * optimParams.scale_up_factor,
                    )
                    self.scene_pose_args.radius_range[0] = max(
                        self.scene_pose_args.max_radius_range[0],
                        self.scene_pose_args.radius_range[0] * optimParams.scale_up_factor,
                    )

                    self.scene_pose_args.theta_range[1] = min(
                        self.scene_pose_args.max_theta_range[1],
                        self.scene_pose_args.theta_range[1] * optimParams.phi_scale_up_factor,
                    )
                    self.scene_pose_args.theta_range[0] = max(
                        self.scene_pose_args.max_theta_range[0],
                        self.scene_pose_args.theta_range[0] * 1 / optimParams.phi_scale_up_factor,
                    )

                    self.scene_pose_args.phi_range[0] = max(
                        self.scene_pose_args.max_phi_range[0],
                        self.scene_pose_args.phi_range[0] * optimParams.phi_scale_up_factor,
                    )
                    self.scene_pose_args.phi_range[1] = min(
                        self.scene_pose_args.max_phi_range[1],
                        self.scene_pose_args.phi_range[1] * optimParams.phi_scale_up_factor,
                    )

                    print("scale up theta_range to:", self.scene_pose_args.theta_range)
                    print("scale up radius_range to:", self.scene_pose_args.radius_range)
                    print("scale up phi_range to:", self.scene_pose_args.phi_range)
                    print("scale up fovy_range to:", self.scene_pose_args.fovy_range)

            loss = 0
            stage_step_rate = min(self.step / iters, 1.0)

            C_batch_size = self.guidance_opt.C_batch_size
            images = []
            depths = []
            alphas = []
            scales = []
            elevation = torch.zeros(C_batch_size, dtype=torch.float32, device=self.device)
            azimuth = torch.zeros(C_batch_size, dtype=torch.float32, device=self.device)
            camera_distances = torch.zeros(C_batch_size, dtype=torch.float32, device=self.device)
            for i in range(C_batch_size):
                viewpoint_cam = self.scene_cams[(self.step - 1) * C_batch_size + i]
                elevation[i] = viewpoint_cam.delta_polar.item()
                azimuth[i] = viewpoint_cam.delta_azimuth.item()  # [-180, 180]
                camera_distances[i] = viewpoint_cam.delta_radius.item()
                if only_env:
                    visible_gaussians = ["floor", "env"]
                else:
                    visible_gaussians = self.visible_gaussians
                out = self.renderer.scene_render(
                    visible_gaussians,
                    viewpoint_cam,
                    bg_color=self.bg_color,
                    sh_deg_aug_ratio=self.dataset_args.sh_deg_aug_ratio,
                    bg_aug_ratio=0.5 * stage_step_rate,
                    shs_aug_ratio=self.dataset_args.shs_aug_ratio,
                    scale_aug_ratio=self.dataset_args.scale_aug_ratio,
                )

                image, viewspace_point_tensor, visibility_filter = (
                    out["image"],
                    out["viewspace_points"],
                    out["visibility_filter"],
                )
                depth, alpha = out["depth"], out["alpha"]
                scales.append(out["scales"].to(self.guidance_opt.g_device))
                images.append(image.to(self.guidance_opt.g_device))
                depths.append(depth.to(self.guidance_opt.g_device))
                alphas.append(alpha.to(self.guidance_opt.g_device))
            images = torch.stack(images, dim=0)
            depths = torch.stack(depths, dim=0)
            alphas = torch.stack(alphas, dim=0)

            _aslatent = False
            use_control_net = False

            if self.step < optimParams.geo_iter or random.random() < optimParams.as_latent_ratio * stage_step_rate:
                _aslatent = True
            if self.step > optimParams.use_control_net_iter and (random.random() < self.guidance_opt.controlnet_ratio):
                use_control_net = True

            (text_embeddings, vds) = self.get_text_embeddings(
                self.renderer.gaussians_collection["env"].text["text_embeddings"],
                elevation,
                azimuth,
                camera_distances,
                True,
                return_null_text_embeddings=True,
            )

            loss = self.guidance.train_step(
                text_embeddings,
                images,
                pred_depth=depths,
                pred_alpha=alphas,
                grad_scale=self.guidance_opt.lambda_guidance,
                use_control_net=use_control_net,
                save_folder=self.train_renders_path,
                iteration=self.step + self.current_prev_n,
                stage_step_rate=stage_step_rate,
                resolution=(self.scene_pose_args.image_h, self.scene_pose_args.image_w),
                guidance_opt=self.guidance_opt,
                as_latent=_aslatent,
                vds=vds,
                obj_id=self.renderer.gaussians_collection["env"].id,
            )
            scales = torch.stack(scales, dim=0)
            loss_scale = torch.mean(scales, dim=-1).mean()
            loss_tv = tv_loss(images)
            loss_tv_depth = tv_loss(depths)

            loss = (
                loss
                + optimParams.lambda_tv * loss_tv
                + optimParams.lambda_tv_depth * loss_tv_depth
                + optimParams.lambda_scale * loss_scale
            )

            # optimize step
            # mem_0 =torch.cuda.memory_allocated()
            loss.to(self.device).backward()
            # logger.debug("MemBeforeBackward: {}, MemAfterBackward: {}", mem_0, torch.cuda.memory_allocated())
            # densify and prune
            if key_gs == "env":
                with torch.no_grad():
                    if self.step < optimParams.densify_until_iter:
                        len_env_gaussian = self.renderer.env_gaussian.get_xyz.shape[0]
                        viewspace_point_tensor, visibility_filter, radii_env = (
                            out["viewspace_points"],
                            out["visibility_filter"],
                            out["radii"][-len_env_gaussian:],
                        )
                        visibility_filter_env = visibility_filter[-len_env_gaussian:]
                        env_gs.max_radii2D[visibility_filter_env] = torch.max(
                            env_gs.max_radii2D[visibility_filter_env],
                            radii_env[visibility_filter_env],
                        )
                        env_gs.add_densification_stats_div(
                            viewspace_point_tensor,
                            visibility_filter_env,
                            -len_env_gaussian,
                        )

                        if (
                            self.step >= optimParams.densify_from_iter
                            and self.step % optimParams.densification_interval == 0
                        ):
                            pcn_0 = env_gs.get_xyz.shape[0]
                            size_threshold = 20 if self.step > optimParams.opacity_reset_interval else None
                            if pcn_0 < optimParams.max_point_number:
                                env_gs.densify_and_prune(
                                    optimParams.densify_grad_threshold,
                                    0.005,
                                    self.renderer.cameras_extent,
                                    None,
                                    # size_threshold,
                                )
                                pcn_1 = env_gs.get_xyz.shape[0]
                                logger.debug(
                                    "Env Point Number Changed From {} to {} After {}",
                                    pcn_0,
                                    pcn_1,
                                    "densify_and_prune",
                                )
                            else:
                                logger.debug(
                                    "Env Point Number {}, Skip densify_and_prune",
                                    pcn_0,
                                )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
            elif key_gs == "floor":
                with torch.no_grad():
                    if self.step < optimParams.densify_until_iter:
                        len_floor_gaussian = self.renderer.floor_gaussian.get_xyz.shape[0]
                        len_env_gaussian = self.renderer.env_gaussian.get_xyz.shape[0]
                        viewspace_point_tensor, visibility_filter, radii_env = (
                            out["viewspace_points"],
                            out["visibility_filter"],
                            out["radii"][-len_env_gaussian - len_floor_gaussian : -len_env_gaussian],
                        )
                        visibility_filter_env = visibility_filter[
                            -len_env_gaussian - len_floor_gaussian : -len_env_gaussian
                        ]
                        floor_gs.max_radii2D[visibility_filter_env] = torch.max(
                            floor_gs.max_radii2D[visibility_filter_env],
                            radii_env[visibility_filter_env],
                        )
                        floor_gs.add_densification_stats_div(
                            viewspace_point_tensor,
                            visibility_filter_env,
                            -len_env_gaussian - len_floor_gaussian,
                            -len_env_gaussian,
                        )

                        if (
                            self.step >= optimParams.densify_from_iter
                            and self.step % optimParams.densification_interval == 0
                        ):
                            pcn_0 = floor_gs.get_xyz.shape[0]
                            size_threshold = 20 if self.step > optimParams.opacity_reset_interval else None
                            if pcn_0 < (optimParams.max_point_number // 3):
                                floor_gs.densify_and_prune(
                                    optimParams.densify_grad_threshold,
                                    0.005,
                                    self.renderer.cameras_extent,
                                    None,
                                    # size_threshold,
                                )
                                pcn_1 = floor_gs.get_xyz.shape[0]
                                logger.debug(
                                    "Floor Point Number Changed From {} to {} After {}",
                                    pcn_0,
                                    pcn_1,
                                    "densify_and_prune",
                                )
                            else:
                                logger.debug(
                                    "Floor Point Number {}, Skip densify_and_prune",
                                    pcn_0,
                                )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
            elif key_gs == "all":
                with torch.no_grad():
                    if self.step < optimParams.densify_until_iter:
                        len_floor_gaussian = self.renderer.floor_gaussian.get_xyz.shape[0]
                        len_env_gaussian = self.renderer.env_gaussian.get_xyz.shape[0]
                        (
                            viewspace_point_tensor,
                            visibility_filter,
                            radii_env,
                            radii_floor,
                        ) = (
                            out["viewspace_points"],
                            out["visibility_filter"],
                            out["radii"][-len_env_gaussian:],
                            out["radii"][-len_env_gaussian - len_floor_gaussian : -len_env_gaussian],
                        )
                        visibility_filter_env = visibility_filter[-len_env_gaussian:]
                        env_gs.max_radii2D[visibility_filter_env] = torch.max(
                            env_gs.max_radii2D[visibility_filter_env],
                            radii_env[visibility_filter_env],
                        )
                        env_gs.add_densification_stats_div(
                            viewspace_point_tensor,
                            visibility_filter_env,
                            -len_env_gaussian,
                        )

                        visibility_filter_floor = visibility_filter[
                            -len_env_gaussian - len_floor_gaussian : -len_env_gaussian
                        ]
                        floor_gs.max_radii2D[visibility_filter_floor] = torch.max(
                            floor_gs.max_radii2D[visibility_filter_floor],
                            radii_floor[visibility_filter_floor],
                        )
                        floor_gs.add_densification_stats_div(
                            viewspace_point_tensor,
                            visibility_filter_floor,
                            -len_env_gaussian - len_floor_gaussian,
                            -len_env_gaussian,
                        )

                        if (
                            self.step >= optimParams.densify_from_iter
                            and self.step % optimParams.densification_interval == 0
                        ):
                            pcn_0 = env_gs.get_xyz.shape[0]
                            size_threshold = 20 if self.step > optimParams.opacity_reset_interval else None
                            if pcn_0 < optimParams.max_point_number:
                                env_gs.densify_and_prune(
                                    optimParams.densify_grad_threshold,
                                    0.005,
                                    self.renderer.cameras_extent,
                                    None,
                                    # size_threshold,
                                )
                                pcn_1 = env_gs.get_xyz.shape[0]

                                logger.debug(
                                    "Env Point Number Changed From {} to {} After {}",
                                    pcn_0,
                                    pcn_1,
                                    "densify_and_prune",
                                )
                            else:
                                logger.debug(
                                    "Env Point Number {}, Skip densify_and_prune",
                                    pcn_0,
                                )
                            pcn_0_floor = floor_gs.get_xyz.shape[0]
                            if pcn_0_floor < (optimParams.max_point_number // 3):
                                floor_gs.densify_and_prune(
                                    optimParams.densify_grad_threshold,
                                    0.005,
                                    self.renderer.cameras_extent,
                                    None,
                                    # size_threshold,
                                )
                                pcn_1_floor = floor_gs.get_xyz.shape[0]
                                logger.debug(
                                    "Floor Point Number Changed From {} to {} After {}",
                                    pcn_0_floor,
                                    pcn_1_floor,
                                    "densify_and_prune",
                                )
                            else:
                                logger.debug(
                                    "Floor Point Number {}, Skip densify_and_prune",
                                    pcn_0_floor,
                                )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.optimizer_floor.step()
                    self.optimizer_floor.zero_grad(set_to_none=True)
            if scene_optim:
                for object_key in self.visible_gaussians:
                    self.renderer.gaussians_collection[object_key].model.optimizer.step()
                    self.renderer.gaussians_collection[object_key].model.optimizer.zero_grad(set_to_none=True)

    def scene_refine_step(self, key_gs="all", only_env=False, scene_optim=False):
        if scene_optim:
            for object_key in self.visible_gaussians:
                self.renderer.gaussians_collection[object_key].model.active_grad()
        optimParams = self.cfg.reconSceneOptimizationParams
        iters = optimParams.iterations
        if key_gs == "env":
            env_gs = self.renderer.env_gaussian
            env_gs.active_grad()
            self.optimizer = env_gs.optimizer
            self.renderer.floor_gaussian.deactive_grad()
        elif key_gs == "floor":
            floor_gs = self.renderer.floor_gaussian
            floor_gs.active_grad()
            self.optimizer = floor_gs.optimizer
            self.renderer.env_gaussian.deactive_grad()
        elif key_gs == "all":
            env_gs = self.renderer.env_gaussian
            env_gs.active_grad()
            floor_gs = self.renderer.floor_gaussian
            floor_gs.active_grad()
            self.optimizer = env_gs.optimizer
            self.optimizer_floor = floor_gs.optimizer
        else:
            self.renderer.env_gaussian.deactive_grad()
            self.renderer.floor_gaussian.deactive_grad()

        for _ in range(self.train_steps):
            self.step += 1
            if self.gt_images is None:
                self.viewpoint_cams = self.scene_cams

            # update lr
            if key_gs == "env" or key_gs == "all":
                env_gs.update_learning_rate(self.step)
                env_gs.update_feature_learning_rate(self.step)
                env_gs.update_rotation_learning_rate(self.step)
                env_gs.update_scaling_learning_rate(self.step)

                if self.step % 300 == 0:
                    env_gs.oneupSHdegree()

            if key_gs == "floor" or key_gs == "all":
                floor_gs.update_learning_rate(self.step)
                floor_gs.update_feature_learning_rate(self.step)
                floor_gs.update_rotation_learning_rate(self.step)
                floor_gs.update_scaling_learning_rate(self.step)

                if self.step % 300 == 0:
                    floor_gs.oneupSHdegree()

            if scene_optim:
                for object_key in self.visible_gaussians:
                    self.renderer.gaussians_collection[object_key].model.update_learning_rate(self.step)
                    self.renderer.gaussians_collection[object_key].model.update_feature_learning_rate(self.step)
                    self.renderer.gaussians_collection[object_key].model.update_rotation_learning_rate(self.step)
                    self.renderer.gaussians_collection[object_key].model.update_scaling_learning_rate(self.step)
            if not optimParams.use_progressive:
                if (
                    self.step >= optimParams.progressive_view_iter
                    and self.step % optimParams.scale_up_cameras_iter == 0
                ):
                    self.scene_pose_args.fovy_range[0] = max(
                        self.scene_pose_args.max_fovy_range[0],
                        self.scene_pose_args.fovy_range[0] * optimParams.fovy_scale_up_factor[0],
                    )
                    self.scene_pose_args.fovy_range[1] = min(
                        self.scene_pose_args.max_fovy_range[1],
                        self.scene_pose_args.fovy_range[1] * optimParams.fovy_scale_up_factor[1],
                    )

                    self.scene_pose_args.radius_range[1] = max(
                        self.scene_pose_args.max_radius_range[1],
                        self.scene_pose_args.radius_range[1] * optimParams.scale_up_factor,
                    )
                    self.scene_pose_args.radius_range[0] = max(
                        self.scene_pose_args.max_radius_range[0],
                        self.scene_pose_args.radius_range[0] * optimParams.scale_up_factor,
                    )

                    self.scene_pose_args.theta_range[1] = min(
                        self.scene_pose_args.max_theta_range[1],
                        self.scene_pose_args.theta_range[1] * optimParams.phi_scale_up_factor,
                    )
                    self.scene_pose_args.theta_range[0] = max(
                        self.scene_pose_args.max_theta_range[0],
                        self.scene_pose_args.theta_range[0] * 1 / optimParams.phi_scale_up_factor,
                    )

                    # opt.reset_resnet_iter = max(500, opt.reset_resnet_iter // 1.25)
                    self.scene_pose_args.phi_range[0] = max(
                        self.scene_pose_args.max_phi_range[0],
                        self.scene_pose_args.phi_range[0] * optimParams.phi_scale_up_factor,
                    )
                    self.scene_pose_args.phi_range[1] = min(
                        self.scene_pose_args.max_phi_range[1],
                        self.scene_pose_args.phi_range[1] * optimParams.phi_scale_up_factor,
                    )

                    print("scale up theta_range to:", self.scene_pose_args.theta_range)
                    print("scale up radius_range to:", self.scene_pose_args.radius_range)
                    print("scale up phi_range to:", self.scene_pose_args.phi_range)
                    print("scale up fovy_range to:", self.scene_pose_args.fovy_range)

            loss = 0

            C_batch_size = self.guidance_opt.C_batch_size
            images = []
            depths = []
            alphas = []

            step_size = C_batch_size
            stage_step_rate = min((self.step) / (iters), 1.0)
            elevation = torch.zeros(self.gt_size, dtype=torch.float32, device=self.device)
            azimuth = torch.zeros(self.gt_size, dtype=torch.float32, device=self.device)
            camera_distances = torch.zeros(self.gt_size, dtype=torch.float32, device=self.device)
            if only_env:
                visible_gaussians = ["floor", "env"]
            else:
                visible_gaussians = self.visible_gaussians

            if self.gt_images is None:
                for i in range(self.gt_size):
                    viewpoint_cam = self.viewpoint_cams[i]
                    # viewpoint_cam
                    elevation[i] = viewpoint_cam.delta_polar.item()
                    azimuth[i] = viewpoint_cam.delta_azimuth.item()  # [-180, 180]
                    camera_distances[i] = viewpoint_cam.delta_radius.item()  # [] - 3.5
                    with torch.no_grad():
                        out = self.renderer.scene_render(
                            visible_gaussians,
                            viewpoint_cam,
                            bg_color=self.bg_color,
                            sh_deg_aug_ratio=self.dataset_args.sh_deg_aug_ratio,
                            bg_aug_ratio=self.dataset_args.bg_aug_ratio,
                            shs_aug_ratio=self.dataset_args.shs_aug_ratio,
                            scale_aug_ratio=self.dataset_args.scale_aug_ratio,
                            test=True,
                            no_grad=True,
                        )
                        image, viewspace_point_tensor, visibility_filter = (
                            out["image"],
                            out["viewspace_points"],
                            out["visibility_filter"],
                        )
                        depth, alpha = out["depth"], out["alpha"]
                        images.append(image.to(self.guidance_opt.g_device))
                        depths.append(depth.to(self.guidance_opt.g_device))
                        alphas.append(alpha.to(self.guidance_opt.g_device))
                images = torch.stack(images, dim=0)
                depths = torch.stack(depths, dim=0)
                alphas = torch.stack(alphas, dim=0)
                # Loss
                _aslatent = False
                use_control_net = True
                (text_embeddings, vds) = self.get_text_embeddings(
                    self.renderer.gaussians_collection["env"].text["text_embeddings"],
                    elevation,
                    azimuth,
                    camera_distances,
                    True,
                    return_null_text_embeddings=True,
                )
                self.gt_images = []
                for j in range(0, self.gt_size // 4 * 4, step_size):
                    gt_image_ = self.guidance.train_step_gt(
                        text_embeddings[3 * j : 3 * (j + step_size)],
                        images[j : (j + step_size)],
                        pred_depth=depths[j : (j + step_size)],
                        pred_alpha=alphas[j : (j + step_size)],
                        grad_scale=self.guidance_opt.lambda_guidance,
                        use_control_net=use_control_net,
                        save_folder=self.train_renders_path,
                        iteration=self.step + self.current_prev_n,
                        stage_step_rate=stage_step_rate,
                        resolution=(
                            self.scene_pose_args.image_h,
                            self.scene_pose_args.image_w,
                        ),
                        guidance_opt=self.guidance_opt,
                        as_latent=_aslatent,
                        vds=vds[j : (j + step_size)],
                        obj_id=self.renderer.gaussians_collection["env"].id,
                        gid=j,
                    )
                    self.gt_images += gt_image_.clone().to(self.device).detach()

            for i in range(self.gt_size):
                out = self.renderer.scene_render(
                    visible_gaussians,
                    self.viewpoint_cams[i],
                    bg_color=self.bg_color,
                    # black_video=self.step % 2,
                    sh_deg_aug_ratio=self.dataset_args.sh_deg_aug_ratio,
                    bg_aug_ratio=self.dataset_args.bg_aug_ratio,
                    shs_aug_ratio=self.dataset_args.shs_aug_ratio,
                    scale_aug_ratio=self.dataset_args.scale_aug_ratio,
                    test=True,
                )
                image, viewspace_point_tensor, visibility_filter = (
                    out["image"].to(torch.float16),
                    out["viewspace_points"],
                    out["visibility_filter"],
                )
                depth = out["depth"]
                loss = l2_loss(image, self.gt_images[i]) * 100
                self.rec_count += 1
                if self.rec_count % 100 == 0:
                    path = self.eval_renders_path / f"{self.rec_count}.jpg"
                    self.save_recon_img(path, image, self.gt_images[i])

                loss.backward()
                if key_gs == "env":
                    with torch.no_grad():
                        len_env_gaussian = env_gs.get_xyz.shape[0]
                        viewspace_point_tensor, visibility_filter, radii_env = (
                            out["viewspace_points"],
                            out["visibility_filter"],
                            out["radii"][-len_env_gaussian:],
                        )
                        visibility_filter_env = visibility_filter[-len_env_gaussian:]
                        env_gs.max_radii2D[visibility_filter_env] = torch.max(
                            env_gs.max_radii2D[visibility_filter_env],
                            radii_env[visibility_filter_env],
                        )
                        env_gs.add_densification_stats_div(
                            viewspace_point_tensor,
                            visibility_filter_env,
                            -len_env_gaussian,
                        )
                        if self.rec_count % optimParams.densification_interval == 0:
                            pcn_0 = env_gs.get_xyz.shape[0]
                            if env_gs.get_xyz.shape[0] < optimParams.max_point_number:
                                size_threshold = 20 if self.step > optimParams.opacity_reset_interval else None
                                env_gs.densify_and_prune(
                                    optimParams.densify_grad_threshold,
                                    0.005,
                                    self.renderer.cameras_extent,
                                    size_threshold,
                                )
                                pcn_1 = env_gs.get_xyz.shape[0]
                                logger.debug(
                                    "Env Point Number Changed From {} to {} After {}",
                                    pcn_0,
                                    pcn_1,
                                    "densify_and_prune",
                                )
                            else:
                                logger.debug("Env Point Number {} Skip", pcn_0)
                        if self.rec_count % optimParams.opacity_reset_interval == 0:
                            env_gs.reset_opacity()
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                elif key_gs == "floor":
                    with torch.no_grad():
                        len_floor_gaussian = self.renderer.floor_gaussian.get_xyz.shape[0]
                        len_env_gaussian = self.renderer.env_gaussian.get_xyz.shape[0]
                        viewspace_point_tensor, visibility_filter, radii_env = (
                            out["viewspace_points"],
                            out["visibility_filter"],
                            out["radii"][-len_env_gaussian - len_floor_gaussian : -len_env_gaussian],
                        )
                        visibility_filter_env = visibility_filter[
                            -len_env_gaussian - len_floor_gaussian : -len_env_gaussian
                        ]
                        floor_gs.max_radii2D[visibility_filter_env] = torch.max(
                            floor_gs.max_radii2D[visibility_filter_env],
                            radii_env[visibility_filter_env],
                        )
                        floor_gs.add_densification_stats_div(
                            viewspace_point_tensor,
                            visibility_filter_env,
                            -len_env_gaussian - len_floor_gaussian,
                            -len_env_gaussian,
                        )
                        if self.rec_count % optimParams.densification_interval == 0:
                            pcn_0 = floor_gs.get_xyz.shape[0]
                            if floor_gs.get_xyz.shape[0] < optimParams.max_point_number // 3:
                                size_threshold = 20 if self.step > optimParams.opacity_reset_interval else None
                                floor_gs.densify_and_prune(
                                    optimParams.densify_grad_threshold,
                                    0.005,
                                    self.renderer.cameras_extent,
                                    size_threshold,
                                )

                                pcn_1 = floor_gs.get_xyz.shape[0]
                                logger.debug(
                                    "Floor Point Number Changed From {} to {} After {}",
                                    pcn_0,
                                    pcn_1,
                                    "densify_and_prune",
                                )
                            else:
                                logger.debug("Floor Point Number {} Skip", pcn_0)
                        if self.rec_count % optimParams.opacity_reset_interval == 0:
                            floor_gs.reset_opacity()
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                elif key_gs == "all":
                    with torch.no_grad():
                        len_floor_gaussian = self.renderer.floor_gaussian.get_xyz.shape[0]
                        len_env_gaussian = env_gs.get_xyz.shape[0]
                        (
                            viewspace_point_tensor,
                            visibility_filter,
                            radii_env,
                            radii_floor,
                        ) = (
                            out["viewspace_points"],
                            out["visibility_filter"],
                            out["radii"][-len_env_gaussian:],
                            out["radii"][-len_env_gaussian - len_floor_gaussian : -len_env_gaussian],
                        )
                        visibility_filter_env = visibility_filter[-len_env_gaussian:]
                        env_gs.max_radii2D[visibility_filter_env] = torch.max(
                            env_gs.max_radii2D[visibility_filter_env],
                            radii_env[visibility_filter_env],
                        )
                        env_gs.add_densification_stats_div(
                            viewspace_point_tensor,
                            visibility_filter_env,
                            -len_env_gaussian,
                        )

                        visibility_filter_floor = visibility_filter[
                            -len_env_gaussian - len_floor_gaussian : -len_env_gaussian
                        ]
                        floor_gs.max_radii2D[visibility_filter_floor] = torch.max(
                            floor_gs.max_radii2D[visibility_filter_floor],
                            radii_floor[visibility_filter_floor],
                        )
                        floor_gs.add_densification_stats_div(
                            viewspace_point_tensor,
                            visibility_filter_floor,
                            -len_env_gaussian - len_floor_gaussian,
                            -len_env_gaussian,
                        )
                        if self.rec_count % optimParams.densification_interval == 0:
                            pcn_0 = env_gs.get_xyz.shape[0]
                            pcn_0_floor = floor_gs.get_xyz.shape[0]
                            size_threshold = 20 if self.step > optimParams.opacity_reset_interval else None
                            if env_gs.get_xyz.shape[0] < optimParams.max_point_number:
                                env_gs.densify_and_prune(
                                    optimParams.densify_grad_threshold,
                                    0.005,
                                    self.renderer.cameras_extent,
                                    size_threshold,
                                )
                                pcn_1 = env_gs.get_xyz.shape[0]
                                logger.debug(
                                    "Env Point Number Changed From {} to {} After {}",
                                    pcn_0,
                                    pcn_1,
                                    "densify_and_prune",
                                )
                            else:
                                env_gs.prune(
                                    0.005,
                                    self.renderer.cameras_extent,
                                    size_threshold,
                                )
                                pcn_1 = env_gs.get_xyz.shape[0]
                                logger.debug(
                                    "Env Point Number From {} to {} After {}",
                                    pcn_0,
                                    pcn_1,
                                    "prune",
                                )
                            if floor_gs.get_xyz.shape[0] < (optimParams.max_point_number // 3):
                                floor_gs.densify_and_prune(
                                    optimParams.densify_grad_threshold,
                                    0.005,
                                    self.renderer.cameras_extent,
                                    size_threshold,
                                )
                                pcn_1_floor = floor_gs.get_xyz.shape[0]
                                logger.debug(
                                    "Floor Point Number Changed From {} to {} After {}",
                                    pcn_0_floor,
                                    pcn_1_floor,
                                    "densify_and_prune",
                                )
                            else:
                                floor_gs.prune(
                                    0.005,
                                    self.renderer.cameras_extent,
                                    size_threshold,
                                )
                                pcn_1_floor = floor_gs.get_xyz.shape[0]
                                logger.debug("Floor Point Number {} Skip", pcn_0)
                        if self.rec_count % optimParams.opacity_reset_interval == 0:
                            env_gs.reset_opacity()
                            floor_gs.reset_opacity()
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.optimizer_floor.step()
                        self.optimizer_floor.zero_grad(set_to_none=True)

                if scene_optim:
                    for object_key in self.visible_gaussians:
                        self.renderer.gaussians_collection[object_key].model.optimizer.step()
                        self.renderer.gaussians_collection[object_key].model.optimizer.zero_grad(set_to_none=True)

    def scene_refine_step_outdoor(self, key_gs="env", only_env=False, scene_optim=False):
        if scene_optim:
            for object_key in self.visible_gaussians:
                self.renderer.gaussians_collection[object_key].model.active_grad()
        optimParams = self.cfg.reconSceneOptimizationParams
        iters = optimParams.iterations
        if key_gs == "env":
            env_gs = self.renderer.env_gaussian
            env_gs.active_grad()
            self.optimizer = env_gs.optimizer
            self.renderer.floor_gaussian.deactive_grad()
        elif key_gs == "floor":
            floor_gs = self.renderer.floor_gaussian
            floor_gs.active_grad()
            self.optimizer = floor_gs.optimizer
            self.renderer.env_gaussian.deactive_grad()
        elif key_gs == "all":
            env_gs = self.renderer.env_gaussian
            env_gs.active_grad()
            floor_gs = self.renderer.floor_gaussian
            floor_gs.active_grad()
            self.optimizer = env_gs.optimizer
            self.optimizer_floor = floor_gs.optimizer
        else:
            self.renderer.env_gaussian.deactive_grad()
            self.renderer.floor_gaussian.deactive_grad()

        for _ in range(self.train_steps):  # self.train_steps = 1
            self.step += 1
            if self.gt_images is None:
                self.viewpoint_cams = self.scene_cams
                self.viewpoint_cams_floor = self.scene_cams_floor

            # update lr
            if key_gs == "env" or key_gs == "all":
                env_gs.update_learning_rate(self.step)
                env_gs.update_feature_learning_rate(self.step)
                env_gs.update_rotation_learning_rate(self.step)
                env_gs.update_scaling_learning_rate(self.step)

                if self.step % 300 == 0:
                    env_gs.oneupSHdegree()

            if key_gs == "floor" or key_gs == "all":
                floor_gs.update_learning_rate(self.step)
                floor_gs.update_feature_learning_rate(self.step)
                floor_gs.update_rotation_learning_rate(self.step)
                floor_gs.update_scaling_learning_rate(self.step)

                if self.step % 300 == 0:
                    floor_gs.oneupSHdegree()

            if scene_optim:
                for object_key in self.visible_gaussians:
                    self.renderer.gaussians_collection[object_key].model.update_learning_rate(self.step)
                    self.renderer.gaussians_collection[object_key].model.update_feature_learning_rate(self.step)
                    self.renderer.gaussians_collection[object_key].model.update_rotation_learning_rate(self.step)
                    self.renderer.gaussians_collection[object_key].model.update_scaling_learning_rate(self.step)
            if not optimParams.use_progressive:
                if (
                    self.step >= optimParams.progressive_view_iter
                    and self.step % optimParams.scale_up_cameras_iter == 0
                ):
                    self.scene_pose_args.fovy_range[0] = max(
                        self.scene_pose_args.max_fovy_range[0],
                        self.scene_pose_args.fovy_range[0] * optimParams.fovy_scale_up_factor[0],
                    )
                    self.scene_pose_args.fovy_range[1] = min(
                        self.scene_pose_args.max_fovy_range[1],
                        self.scene_pose_args.fovy_range[1] * optimParams.fovy_scale_up_factor[1],
                    )

                    self.scene_pose_args.radius_range[1] = max(
                        self.scene_pose_args.max_radius_range[1],
                        self.scene_pose_args.radius_range[1] * optimParams.scale_up_factor,
                    )
                    self.scene_pose_args.radius_range[0] = max(
                        self.scene_pose_args.max_radius_range[0],
                        self.scene_pose_args.radius_range[0] * optimParams.scale_up_factor,
                    )

                    self.scene_pose_args.theta_range[1] = min(
                        self.scene_pose_args.max_theta_range[1],
                        self.scene_pose_args.theta_range[1] * optimParams.phi_scale_up_factor,
                    )
                    self.scene_pose_args.theta_range[0] = max(
                        self.scene_pose_args.max_theta_range[0],
                        self.scene_pose_args.theta_range[0] * 1 / optimParams.phi_scale_up_factor,
                    )

                    # opt.reset_resnet_iter = max(500, opt.reset_resnet_iter // 1.25)
                    self.scene_pose_args.phi_range[0] = max(
                        self.scene_pose_args.max_phi_range[0],
                        self.scene_pose_args.phi_range[0] * optimParams.phi_scale_up_factor,
                    )
                    self.scene_pose_args.phi_range[1] = min(
                        self.scene_pose_args.max_phi_range[1],
                        self.scene_pose_args.phi_range[1] * optimParams.phi_scale_up_factor,
                    )

                    print("scale up theta_range to:", self.scene_pose_args.theta_range)
                    print("scale up radius_range to:", self.scene_pose_args.radius_range)
                    print("scale up phi_range to:", self.scene_pose_args.phi_range)
                    print("scale up fovy_range to:", self.scene_pose_args.fovy_range)

            loss = 0

            C_batch_size = self.guidance_opt.C_batch_size
            images = []
            depths = []
            alphas = []
            images_floor = []
            depths_floor = []
            alphas_floor = []

            step_size = C_batch_size
            stage_step_rate = min((self.step) / (iters), 1.0)
            elevation = torch.zeros(self.gt_size, dtype=torch.float32, device=self.device)
            azimuth = torch.zeros(self.gt_size, dtype=torch.float32, device=self.device)
            camera_distances = torch.zeros(self.gt_size, dtype=torch.float32, device=self.device)

            elevation_floor = torch.zeros(self.gt_size, dtype=torch.float32, device=self.device)
            azimuth_floor = torch.zeros(self.gt_size, dtype=torch.float32, device=self.device)
            camera_distances_floor = torch.zeros(self.gt_size, dtype=torch.float32, device=self.device)

            if only_env:
                visible_gaussians = ["floor", "env"]
            else:
                visible_gaussians = self.visible_gaussians
            if self.gt_images is None:

                for i in range(self.gt_size):
                    viewpoint_cam = self.viewpoint_cams[i]
                    # viewpoint_cam
                    elevation[i] = viewpoint_cam.delta_polar.item()
                    azimuth[i] = viewpoint_cam.delta_azimuth.item()  # [-180, 180]
                    camera_distances[i] = viewpoint_cam.delta_radius.item()  # [] - 3.5
                    with torch.no_grad():
                        out = self.renderer.scene_render(
                            visible_gaussians,
                            viewpoint_cam,
                            bg_color=self.bg_color,
                            sh_deg_aug_ratio=self.dataset_args.sh_deg_aug_ratio,
                            bg_aug_ratio=self.dataset_args.bg_aug_ratio,
                            shs_aug_ratio=self.dataset_args.shs_aug_ratio,
                            scale_aug_ratio=self.dataset_args.scale_aug_ratio,
                            test=True,
                            no_grad=True,
                        )
                        image, viewspace_point_tensor, visibility_filter = (
                            out["image"],
                            out["viewspace_points"],
                            out["visibility_filter"],
                        )
                        depth, alpha = out["depth"], out["alpha"]
                        # scales.append(out["scales"].to(self.guidance_opt.g_device))
                        images.append(image.to(self.guidance_opt.g_device))
                        depths.append(depth.to(self.guidance_opt.g_device))
                        alphas.append(alpha.to(self.guidance_opt.g_device))
                for i in range(self.gt_size):
                    viewpoint_cam_floor = self.viewpoint_cams_floor[i]
                    # viewpoint_cam
                    elevation_floor[i] = viewpoint_cam_floor.delta_polar.item()
                    azimuth_floor[i] = viewpoint_cam_floor.delta_azimuth.item()  # [-180, 180]
                    camera_distances_floor[i] = viewpoint_cam_floor.delta_radius.item()  # [] - 3.5
                    with torch.no_grad():
                        out = self.renderer.scene_render(
                            visible_gaussians,
                            viewpoint_cam_floor,
                            bg_color=self.bg_color,
                            sh_deg_aug_ratio=self.dataset_args.sh_deg_aug_ratio,
                            bg_aug_ratio=self.dataset_args.bg_aug_ratio,
                            shs_aug_ratio=self.dataset_args.shs_aug_ratio,
                            scale_aug_ratio=self.dataset_args.scale_aug_ratio,
                            test=True,
                            no_grad=True,
                        )
                        image, viewspace_point_tensor, visibility_filter = (
                            out["image"],
                            out["viewspace_points"],
                            out["visibility_filter"],
                        )
                        depth, alpha = out["depth"], out["alpha"]
                        images_floor.append(image.to(self.guidance_opt.g_device))
                        depths_floor.append(depth.to(self.guidance_opt.g_device))
                        alphas_floor.append(alpha.to(self.guidance_opt.g_device))
                images = torch.stack(images, dim=0)
                depths = torch.stack(depths, dim=0)
                alphas = torch.stack(alphas, dim=0)
                images_floor = torch.stack(images_floor, dim=0)
                depths_floor = torch.stack(depths_floor, dim=0)
                alphas_floor = torch.stack(alphas_floor, dim=0)
                # Loss
                _aslatent = False
                use_control_net = True
                (text_embeddings, vds) = self.get_text_embeddings(
                    self.renderer.gaussians_collection["env"].text["text_embeddings"],
                    elevation,
                    azimuth,
                    camera_distances,
                    True,
                    return_null_text_embeddings=True,
                )
                self.gt_images = []
                self.gt_images_floor = []

                for j in range(0, self.gt_size // 4 * 4, step_size):
                    gt_image_ = self.guidance.train_step_gt(
                        text_embeddings[3 * j : 3 * (j + step_size)],
                        images[j : (j + step_size)],
                        pred_depth=depths[j : (j + step_size)],
                        pred_alpha=alphas[j : (j + step_size)],
                        grad_scale=self.guidance_opt.lambda_guidance,
                        use_control_net=use_control_net,
                        save_folder=self.train_renders_path,
                        iteration=self.step + self.current_prev_n,
                        stage_step_rate=stage_step_rate,
                        resolution=(
                            self.scene_pose_args.image_h,
                            self.scene_pose_args.image_w,
                        ),
                        guidance_opt=self.guidance_opt,
                        as_latent=_aslatent,
                        vds=vds[j : (j + step_size)],
                        obj_id=self.renderer.gaussians_collection["env"].id,
                        gid=j,
                    )
                    self.gt_images += gt_image_.clone().to(self.device).detach()
                for j in range(0, self.gt_size // 4 * 4, step_size):
                    gt_image_ = self.guidance.train_step_gt(
                        text_embeddings[3 * j : 3 * (j + step_size)],
                        images_floor[j : (j + step_size)],
                        pred_depth=depths_floor[j : (j + step_size)],
                        pred_alpha=alphas_floor[j : (j + step_size)],
                        grad_scale=self.guidance_opt.lambda_guidance,
                        use_control_net=use_control_net,
                        save_folder=self.train_renders_path,
                        iteration=self.step + self.current_prev_n,
                        stage_step_rate=stage_step_rate,
                        resolution=(
                            self.scene_pose_args.image_h,
                            self.scene_pose_args.image_w,
                        ),
                        guidance_opt=self.guidance_opt,
                        as_latent=_aslatent,
                        vds=vds[j : (j + step_size)],
                        obj_id=self.renderer.gaussians_collection["floor"].id,
                        gid=j,
                    )
                    self.gt_images_floor += gt_image_.clone().to(self.device).detach()

            for i in range(self.gt_size):
                out = self.renderer.scene_render(
                    visible_gaussians,
                    (self.viewpoint_cams[i] if key_gs == "env" else self.viewpoint_cams_floor[i]),
                    bg_color=self.bg_color,
                    sh_deg_aug_ratio=self.dataset_args.sh_deg_aug_ratio,
                    bg_aug_ratio=self.dataset_args.bg_aug_ratio,
                    shs_aug_ratio=self.dataset_args.shs_aug_ratio,
                    scale_aug_ratio=self.dataset_args.scale_aug_ratio,
                    test=True,
                )
                image, viewspace_point_tensor, visibility_filter = (
                    out["image"].to(torch.float16),
                    out["viewspace_points"],
                    out["visibility_filter"],
                )
                depth = out["depth"]
                loss = (
                    l2_loss(
                        image,
                        (self.gt_images[i] if key_gs == "env" else self.gt_images_floor[i]),
                    )
                    * 100
                )
                self.rec_count += 1
                if self.rec_count % 100 == 0:
                    path = self.eval_renders_path / f"{self.rec_count}.jpg"
                    self.save_recon_img(
                        path,
                        image,
                        (self.gt_images[i] if key_gs == "env" else self.gt_images_floor[i]),
                    )

                loss.backward()
                if key_gs == "env":
                    with torch.no_grad():
                        len_env_gaussian = env_gs.get_xyz.shape[0]
                        viewspace_point_tensor, visibility_filter, radii_env = (
                            out["viewspace_points"],
                            out["visibility_filter"],
                            out["radii"][-len_env_gaussian:],
                        )
                        visibility_filter_env = visibility_filter[-len_env_gaussian:]
                        env_gs.max_radii2D[visibility_filter_env] = torch.max(
                            env_gs.max_radii2D[visibility_filter_env],
                            radii_env[visibility_filter_env],
                        )
                        env_gs.add_densification_stats_div(
                            viewspace_point_tensor,
                            visibility_filter_env,
                            -len_env_gaussian,
                        )
                        if self.rec_count % optimParams.densification_interval == 0:
                            pcn_0 = env_gs.get_xyz.shape[0]
                            if env_gs.get_xyz.shape[0] < optimParams.max_point_number:
                                size_threshold = 20 if self.step > optimParams.opacity_reset_interval else None
                                env_gs.densify_and_prune(
                                    optimParams.densify_grad_threshold,
                                    0.005,
                                    self.renderer.cameras_extent,
                                    size_threshold,
                                )
                                pcn_1 = env_gs.get_xyz.shape[0]
                                logger.debug(
                                    "Env Point Number Changed From {} to {} After {}",
                                    pcn_0,
                                    pcn_1,
                                    "densify_and_prune",
                                )
                            else:
                                logger.debug("Env Point Number {} Skip", pcn_0)
                        if self.rec_count % optimParams.opacity_reset_interval == 0:
                            env_gs.reset_opacity()
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                elif key_gs == "floor":
                    with torch.no_grad():
                        len_floor_gaussian = self.renderer.floor_gaussian.get_xyz.shape[0]
                        len_env_gaussian = self.renderer.env_gaussian.get_xyz.shape[0]
                        viewspace_point_tensor, visibility_filter, radii_env = (
                            out["viewspace_points"],
                            out["visibility_filter"],
                            out["radii"][-len_env_gaussian - len_floor_gaussian : -len_env_gaussian],
                        )
                        visibility_filter_env = visibility_filter[
                            -len_env_gaussian - len_floor_gaussian : -len_env_gaussian
                        ]
                        floor_gs.max_radii2D[visibility_filter_env] = torch.max(
                            floor_gs.max_radii2D[visibility_filter_env],
                            radii_env[visibility_filter_env],
                        )
                        floor_gs.add_densification_stats_div(
                            viewspace_point_tensor,
                            visibility_filter_env,
                            -len_env_gaussian - len_floor_gaussian,
                            -len_env_gaussian,
                        )
                        if self.rec_count % optimParams.densification_interval == 0:
                            pcn_0 = floor_gs.get_xyz.shape[0]
                            size_threshold = 20 if self.step > optimParams.opacity_reset_interval else None
                            floor_gs.densify_and_prune(
                                optimParams.densify_grad_threshold,
                                0.005,
                                self.renderer.cameras_extent,
                                size_threshold,
                            )

                            pcn_1 = floor_gs.get_xyz.shape[0]
                            logger.debug(
                                "Floor Point Number Changed From {} to {} After {}",
                                pcn_0,
                                pcn_1,
                                "densify_and_prune",
                            )
                        if self.rec_count % optimParams.opacity_reset_interval == 0:
                            floor_gs.reset_opacity()
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                elif key_gs == "all":
                    with torch.no_grad():
                        len_floor_gaussian = self.renderer.floor_gaussian.get_xyz.shape[0]
                        len_env_gaussian = env_gs.get_xyz.shape[0]
                        (
                            viewspace_point_tensor,
                            visibility_filter,
                            radii_env,
                            radii_floor,
                        ) = (
                            out["viewspace_points"],
                            out["visibility_filter"],
                            out["radii"][-len_env_gaussian:],
                            out["radii"][-len_env_gaussian - len_floor_gaussian : -len_env_gaussian],
                        )
                        visibility_filter_env = visibility_filter[-len_env_gaussian:]
                        env_gs.max_radii2D[visibility_filter_env] = torch.max(
                            env_gs.max_radii2D[visibility_filter_env],
                            radii_env[visibility_filter_env],
                        )
                        env_gs.add_densification_stats_div(
                            viewspace_point_tensor,
                            visibility_filter_env,
                            -len_env_gaussian,
                        )

                        visibility_filter_floor = visibility_filter[
                            -len_env_gaussian - len_floor_gaussian : -len_env_gaussian
                        ]
                        floor_gs.max_radii2D[visibility_filter_floor] = torch.max(
                            floor_gs.max_radii2D[visibility_filter_floor],
                            radii_floor[visibility_filter_floor],
                        )
                        floor_gs.add_densification_stats_div(
                            viewspace_point_tensor,
                            visibility_filter_floor,
                            -len_env_gaussian - len_floor_gaussian,
                            -len_env_gaussian,
                        )
                        if self.rec_count % optimParams.densification_interval == 0:
                            pcn_0 = env_gs.get_xyz.shape[0]
                            pcn_0_floor = floor_gs.get_xyz.shape[0]
                            size_threshold = 20 if self.step > optimParams.opacity_reset_interval else None
                            if env_gs.get_xyz.shape[0] < optimParams.max_point_number:
                                env_gs.densify_and_prune(
                                    optimParams.densify_grad_threshold,
                                    0.005,
                                    self.renderer.cameras_extent,
                                    size_threshold,
                                )
                                pcn_1 = env_gs.get_xyz.shape[0]
                                logger.debug(
                                    "Env Point Number Changed From {} to {} After {}",
                                    pcn_0,
                                    pcn_1,
                                    "densify_and_prune",
                                )
                            else:
                                env_gs.prune(
                                    0.005,
                                    self.renderer.cameras_extent,
                                    size_threshold,
                                )
                                pcn_1 = env_gs.get_xyz.shape[0]
                                logger.debug(
                                    "Env Point Number From {} to {} After {}",
                                    pcn_0,
                                    pcn_1,
                                    "prune",
                                )
                            if floor_gs.get_xyz.shape[0] < (optimParams.max_point_number // 3):
                                floor_gs.densify_and_prune(
                                    optimParams.densify_grad_threshold,
                                    0.005,
                                    self.renderer.cameras_extent,
                                    size_threshold,
                                )
                                pcn_1_floor = floor_gs.get_xyz.shape[0]
                                logger.debug(
                                    "Floor Point Number Changed From {} to {} After {}",
                                    pcn_0_floor,
                                    pcn_1_floor,
                                    "densify_and_prune",
                                )
                            else:
                                floor_gs.prune(
                                    0.005,
                                    self.renderer.cameras_extent,
                                    size_threshold,
                                )
                                pcn_1_floor = floor_gs.get_xyz.shape[0]
                                logger.debug("Floor Point Number {} Skip", pcn_0)
                        if self.rec_count % optimParams.opacity_reset_interval == 0:
                            env_gs.reset_opacity()
                            floor_gs.reset_opacity()
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.optimizer_floor.step()
                        self.optimizer_floor.zero_grad(set_to_none=True)

                if scene_optim:
                    for object_key in self.visible_gaussians:
                        self.renderer.gaussians_collection[object_key].model.optimizer.step()
                        self.renderer.gaussians_collection[object_key].model.optimizer.zero_grad(set_to_none=True)

    def save_recon_img(self, path, pred_rgb, gt_image):
        save_image([pred_rgb, gt_image], path)

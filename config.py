from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelParams:
    _source_path: str = ""
    _model_path: str = ""
    pretrained_model_path: Optional[str] = None
    _images: str = "images"
    workspace: str = "debug"
    batch: int = 10
    _resolution: int = -1
    _white_background: bool = True
    data_device: str = "cuda"
    eval: bool = False
    opt_path: Optional[str] = None

    # augmentation
    sh_deg_aug_ratio: float = 0.1
    bg_aug_ratio: float = 0.5
    shs_aug_ratio: float = 0.0
    scale_aug_ratio: float = 1.0


@dataclass
class OptimizationParams:
    iterations: int = 2000
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000
    feature_lr: float = 0.0050
    feature_lr_final: float = 0.0030

    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001

    geo_iter: int = 0
    as_latent_ratio: float = 0.2

    scaling_lr_final: float = 0.001
    rotation_lr_final: float = 0.0002

    percent_dense: float = 0.003
    densify_grad_threshold: float = 0.00075

    lambda_tv: float = 1.0  # 0.1
    lambda_bin: float = 10.0
    lambda_scale: float = 1.0
    lambda_sat: float = 1.0
    lambda_radius: float = 1.0
    lambda_depth: float = 1.0
    lambda_tv_depth: float = 1.0
    densification_interval: int = 100
    opacity_reset_interval: int = 300
    densify_from_iter: int = 100
    densify_until_iter: int = 30_00

    use_control_net_iter: int = 10000000

    use_progressive: bool = False
    progressive_view_iter: int = 500
    progressive_view_init_ratio: float = 0.2

    scale_up_cameras_iter: int = 500
    scale_up_factor: float = 0.95
    fovy_scale_up_factor: List[float] = field(default_factory=lambda: [0.75, 1.1])
    phi_scale_up_factor: float = 1.5
    max_point_number: int = 1200000
    # If you only need to refine through our reconstructive generation process.
    only_recon_stage: bool = False
    # Common prompts for the entire scene generation process
    style_prompt: str = ""
    style_negative_prompt: str = ""


@dataclass
class PipelineParams:
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    debug: bool = False


@dataclass
class GenerateCamParams:
    radius_range: List[float] = field(
        default_factory=lambda: [5.2, 5.5]
    )
    max_radius_range: List[float] = field(default_factory=lambda: [3.5, 5.0])
    default_radius: float = 3.5
    theta_range: List[float] = field(default_factory=lambda: [45, 105])
    max_theta_range: List[float] = field(default_factory=lambda: [45, 105])
    phi_range: List[float] = field(default_factory=lambda: [-180, 180])
    max_phi_range: List[float] = field(default_factory=lambda: [-180, 180])
    fovy_range: List[float] = field(
        default_factory=lambda: [0.32, 0.60]
    )
    max_fovy_range: List[float] = field(default_factory=lambda: [0.16, 0.60])
    rand_cam_gamma: float = 1.0
    angle_overhead: float = 30
    angle_front: float = 60
    render_45: bool = True
    uniform_sphere_rate: float = 0
    image_w: int = 512
    image_h: int = 512
    SSAA: int = 1
    default_polar: float = 90
    default_azimuth: float = 0
    default_fovy: float = 0.55
    jitter_pose: bool = True
    jitter_center: float = 0.05
    jitter_target: float = 0.05
    jitter_up: float = 0.01
    device: str = "cuda"


@dataclass
class GuidanceParams:
    random_delta: bool = False

    guidance: str = "MTSD"
    g_device: str = "cuda"

    model_key: Optional[str] = None
    is_safe_tensor: bool = False
    base_model_key: Optional[str] = None

    controlnet_model_key: Optional[str] = None

    # For Perp-Neg
    perpneg: bool = True
    negative_w: float = -2.0
    front_decay_factor: float = 2.0
    side_decay_factor: float = 10.0

    vram_O: bool = False
    fp16: bool = True
    hf_key: Optional[str] = None
    t_range: List[float] = field(default_factory=lambda: [0.02, 0.5])
    max_t_range: float = 0.98

    num_train_timesteps: Optional[int] = None

    fix_noise: bool = False
    noise_seed: int = 0

    delta_t: int = 80
    annealing_intervals: bool = True
    text: str = ""
    inverse_text: str = ""
    textual_inversion_path: Optional[str] = None
    LoRA_path: Optional[str] = None
    negative: str = ""
    guidance_scale: float = 7.5
    denoise_guidance_scale: float = 1.0
    lambda_guidance: float = 1.0

    xs_eta: float = 0.0

    # multi-batch
    C_batch_size: int = 4

    vis_interval: int = 100
    stage_refine_t: int = 125


@dataclass
class ParamsGroups:
    outdir: str = "logs"

    # training batch size per iter
    batch_size: int = 1
    H: int = 800
    W: int = 800

    visualize_samples: bool = False
    only_render: bool = False

    modelParams: ModelParams = field(default_factory=ModelParams)
    optimizationParams: OptimizationParams = field(default_factory=OptimizationParams)
    reconOptimizationParams: OptimizationParams = field(
        default_factory=OptimizationParams
    )

    sceneOptimizationParams: OptimizationParams = field(
        default_factory=OptimizationParams
    )
    reconSceneOptimizationParams: OptimizationParams = field(
        default_factory=OptimizationParams
    )
    fineSceneOptimizationParams: OptimizationParams = field(
        default_factory=OptimizationParams
    )
    pipelineParams: PipelineParams = field(default_factory=PipelineParams)
    generateCamParams: GenerateCamParams = field(default_factory=GenerateCamParams)
    sceneGenerateCamParams: GenerateCamParams = field(default_factory=GenerateCamParams)
    guidanceParams: GuidanceParams = field(default_factory=GuidanceParams)
    editParams: Optional[Dict] = None

    seed: int = 0

    log: Optional[Dict] = None
    scene_configs: Optional[Dict] = None
    mode_args: Optional[Dict] = None


@dataclass
class ObjectParams:
    id: str = ""
    sh_degree: int = 3
    text: str = ""
    negative_text: str = ""
    image: str = ""
    init_guided: str = "pointe"
    init_prompt: str = ""
    cam_pose_method: str = "object"
    use_pointe_rgb: bool = False
    num_pts: int = 20000
    radius: float = 0.5

@dataclass
class ObjectsParamsGroups:
    # training batch size per iter
    batch_size: int = 1

    H: int = 800
    W: int = 800

    visualize_samples: bool = False

    modelParams: ModelParams = field(default_factory=ModelParams)
    optimizationParams: OptimizationParams = field(default_factory=OptimizationParams)
    reconOptimizationParams: OptimizationParams = field(
        default_factory=OptimizationParams
    )
    pipelineParams: PipelineParams = field(default_factory=PipelineParams)
    generateCamParams: GenerateCamParams = field(default_factory=GenerateCamParams)
    guidanceParams: GuidanceParams = field(default_factory=GuidanceParams)
    objectParams: ObjectParams = field(default_factory=ObjectParams)
    seed: int = 0

    log: Optional[Dict] = None
    mode_args: Optional[Dict] = None

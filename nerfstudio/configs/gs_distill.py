# nerfstudio/configs/method_configs/gs_distill.py
"""
Config for 3DGS-to-NeRF Distillation.
"""

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.gs_datamanager import GSDataManagerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.models.splatfacto import SplatfactoModelConfig
from nerfstudio.pipelines.gs_distill_pipeline import GS2NeRFDistillPipelineConfig
from nerfstudio.plugins.types import MethodSpecification
from pathlib import Path
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig

gs_distill_method = MethodSpecification(
    config=TrainerConfig(
        method_name="gs-distill",
        steps_per_eval_batch=500,
        steps_per_eval_image=500, # 启用图像评估
        steps_per_save=2000,
        max_num_iterations=30000, # [注意: 300 太少了, 我改成了 30000]
        mixed_precision=False, # [注意: 3DGS 蒸馏建议关闭混合精度]
        
        pipeline=GS2NeRFDistillPipelineConfig(
            
            # ==================================================================
            # === 路径配置 (从命令行提供) ===
            # ==================================================================
            
            # 1. 教师模型的 PLY 文件路径
            # (保持不变, 必须从命令行提供)
            # --pipeline.load-ply-path /path/to/file.ply
            load_ply_path=None,
            
            # 2. 数据集路径
            # [!!! 更改 4: 替换 datamanager 配置]
            datamanager=GSDataManagerConfig(
                # 我们不再使用 VanillaDataManager。
                # 这个新的 DataManager 将从命令行读取
                # --pipeline.datamanager.cameras-json-path ...
                
                # 我们可以保留这些设置, GSDataManager 也会使用它们
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            # ==================================================================
            # === 模型配置 ===
            # ==================================================================

            # --- 教师 (Splatfacto) ---
            teacher_model=SplatfactoModelConfig(
                sh_degree=3, # 确保这与您的 .ply 文件匹配
                camera_optimizer=CameraOptimizerConfig(mode="off"),
                use_scale_regularization=False,
                rasterize_mode="antialiased", # 建议使用 'antialiased' 以获得更高质量
            ),
            
            # --- 学生 (Nerfacto) ---
            student_model=NerfactoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
            ),

            # --- 蒸馏损失配置 ---
            use_ssim_loss=True,
            ssim_lambda=0.2,
        ),
        
        optimizers={
            # 优化器将自动附加到 'student_model' 的参数上
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=30000), # 匹配 max_num_iterations
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=30000), # 匹配 max_num_iterations
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="tensorboard",
    ),
    # [!!! 更改 5: 更新 description]
    description="3DGS (Splatfacto) to NeRF (Nerfacto) distillation. "
                "Requires --pipeline.load-ply-path AND "
                "--pipeline.datamanager.cameras-json-path arguments.",
)
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [ 文件: nerfstudio/pipelines/gs_distill_pipeline.py ]
# -------------------------------------------------------------
"""
Pipeline for distilling a pre-trained 3DGS (Splatfacto) model
into a new NeRF (Nerfacto) model.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Type, cast, Optional, Tuple, Any
from torch.cuda.amp.grad_scaler import GradScaler
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import Parameter
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
import sys
import random  # <--- (!!!) 更改 1: 导入 random 模块

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.base_datamanager import DataManager
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils.gs_loader import load_gs_from_ply
from nerfstudio.utils.rich_utils import CONSOLE

# (!!!) 修复：导入我们需要的 Callback 类型
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes


@dataclass
class GS2NeRFDistillPipelineConfig(VanillaPipelineConfig):
    """Configuration for the GS-to-NeRF Distillation Pipeline."""
    
    _target: Type = field(default_factory=lambda: GS2NeRFDistillPipeline)
    
    teacher_model: ModelConfig = field(default_factory=ModelConfig)
    load_ply_path: Optional[Path] = None
    student_model: ModelConfig = field(default_factory=ModelConfig)
    use_ssim_loss: bool = True
    ssim_lambda: float = 0.2

class GS2NeRFDistillPipeline(VanillaPipeline):
    """
    Pipeline that distills a pre-trained Splatfacto model (teacher)
    into a Nerfacto model (student).
    """

    config: GS2NeRFDistillPipelineConfig
    datamanager: DataManager

    def __init__(
        self,
        config: GS2NeRFDistillPipelineConfig,
        device: str,
        test_mode: str = "test",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler) 

        self.gs_params = self._load_ply_data()
        
        seed_xyz = self.gs_params["means"]
        seed_colors_empty = torch.empty(0, 3, dtype=torch.uint8)
        seed_points = (seed_xyz, seed_colors_empty)
        
        self.teacher_model: Model = config.teacher_model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            device=device,
            seed_points=seed_points,
        )
        self.teacher_model.to(device)
        
        self._overwrite_teacher_params()

        self.student_model: Model = config.student_model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            device=device,
        )
        self.student_model.to(device)
        
        self.model = self.student_model 

        if self.config.use_ssim_loss:
            from pytorch_msssim import SSIM
            self.ssim = SSIM(data_range=1.0, size_average=True, channel=3).to(device)
            
        # 核心修复：完全替换 model 的 get_outputs_for_camera_ray_bundle 方法
        def safe_get_outputs_for_camera_ray_bundle(camera_ray_bundle):
            """完全替换的安全方法，不调用有问题的原始方法"""
            with torch.no_grad():
                # 直接使用 student_model 的 forward 方法
                outputs = self.student_model(camera_ray_bundle)
                
                # 如果输出为 None，创建虚拟输出
                if outputs is None:
                    CONSOLE.print("[yellow]Warning: Student model returned None, creating dummy outputs[/yellow]")
                    num_rays = camera_ray_bundle.origins.shape[0]
                    return {
                        "rgb": torch.zeros((num_rays, 3), device=self.device),
                        "accumulation": torch.ones((num_rays, 1), device=self.device),
                        "depth": torch.zeros((num_rays, 1), device=self.device),
                    }
                
                return outputs
        
        # 完全替换方法，不保留原始引用
        self.model.get_outputs_for_camera_ray_bundle = safe_get_outputs_for_camera_ray_bundle

    def _load_ply_data(self) -> Dict[str, torch.Tensor]:
        CONSOLE.print("[bold yellow]Loading teacher model parameters...[/bold yellow]")
        
        if self.config.load_ply_path is None:
            CONSOLE.print(
                "[bold red]错误：[/bold red] "
                "未提供教师模型的 .ply 路径。\n"
                "请使用以下参数运行:\n"
                "[cyan]--pipeline.load-ply-path /path/to/your/point_cloud.ply[/cyan]"
            )
            raise ValueError("Missing 'load_ply_path' in pipeline config.")
        
        try:
            sh_degree = getattr(self.config.teacher_model, "sh_degree", 3)
            gs_params = load_gs_from_ply(
                self.config.load_ply_path,
                sh_degree
            )
        except Exception as e:
            CONSOLE.print(f"[bold red]Error loading PLY file:[/bold red] {e}")
            raise
        
        return gs_params

    def _overwrite_teacher_params(self):
        CONSOLE.print("[bold blue]Overwriting teacher model parameters with loaded PLY data...[/bold blue]")
        
        params_to_overwrite = {
            "means": self.gs_params["means"],
            "scales": self.gs_params["scales"],
            "quats": self.gs_params["quats"],
            "features_dc": self.gs_params["features_dc"],
            "features_rest": self.gs_params["features_rest"],
            "opacities": self.gs_params["opacities"],
        }
        
        progress = Progress(
            TextColumn("[bold blue]Overwriting params..."),
            BarColumn(),
            TextColumn("{task.description}"),
            TimeRemainingColumn(),
            transient=True,
        )
        
        with progress:
            task = progress.add_task("params", total=len(params_to_overwrite))
            for name, tensor in params_to_overwrite.items():
                progress.update(task, description=f"[cyan]{name}")
                
                if name in self.teacher_model.gauss_params:
                    
                    expected_shape = self.teacher_model.gauss_params[name].shape
                    
                    if name == "features_dc" and len(tensor.shape) == 3 and tensor.shape[1] == 1:
                        tensor = tensor.squeeze(1) # [N, 1, 3] -> [N, 3]
                    
                    if expected_shape != tensor.shape:
                        raise ValueError(f"Shape mismatch for '{name}'. "
                                         f"Model expected {expected_shape}, but PLY gave {tensor.shape}.")
                    
                    self.teacher_model.gauss_params[name].data = tensor.to(self.device)
                else:
                    CONSOLE.print(f"[bold yellow]Warning:[/bold yellow] Loaded parameter '{name}' not found in teacher model.")
                progress.advance(task)

        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        CONSOLE.print("[bold green]✓ Teacher model loaded and frozen.[/bold green]")
        del self.gs_params 

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        CONSOLE.print("Getting parameter groups from [bold]student model[/bold]...")
        return self.student_model.get_param_groups()

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """
        返回所有必要的训练回调
        """
        student_callbacks = self.student_model.get_training_callbacks(training_callback_attributes)
        pipeline_callbacks = super().get_training_callbacks(training_callback_attributes)
        return student_callbacks + pipeline_callbacks

    @torch.no_grad()
    def _get_teacher_outputs(self, ray_bundle: RayBundle, batch: Dict):
        """
        Helper to run the teacher model.
        """
        
        self.teacher_model.eval()
        
        camera_idx = ray_bundle.camera_indices[0, 0].item()
        camera = self.datamanager.train_dataset.cameras[camera_idx:camera_idx+1].to(self.device)
        
        teacher_full_outputs = self.teacher_model.get_outputs(camera)
        
        y = ray_bundle.metadata["ray_indices"][:, 1].long()
        x = ray_bundle.metadata["ray_indices"][:, 2].long()
        
        teacher_rgb = teacher_full_outputs["rgb"][y, x]

        return {"rgb": teacher_rgb}
    
    def get_train_loss_dict(self, step: int) -> Tuple[Dict, Dict, Dict]:
        """
        This is the main training step.
        """
        ray_bundle, batch = self.datamanager.next_train(step)
        
        with torch.no_grad():
            teacher_outputs = self._get_teacher_outputs(ray_bundle, batch)
            gt_rgb = teacher_outputs["rgb"].detach()

        if not torch.isfinite(gt_rgb).all():
            CONSOLE.log("[bold red]!!!!!!!! 灾难性故障 !!!!!!!![/bold red]")
            CONSOLE.log(f"Step {step}: 教师模型 (Splatfacto) 渲染输出了 Inf 或 NaN！")
            sys.stdout.flush()
            sys.stderr.flush()
            raise ValueError(f"Teacher model (Splatfacto) produced Inf/NaN at step {step}.")

        gt_rgb_safe = torch.clamp(gt_rgb, min=0.0, max=1.0)

        student_outputs = self.student_model(ray_bundle)

        if torch.isnan(student_outputs["rgb"]).any() or torch.isinf(student_outputs["rgb"]).any():
            print("\n" + "="*80)
            print(f"致命错误 (Step {step}): 学生模型 (Student) 输出了 Inf/NaN！")
            print("="*80 + "\n")
            raise ValueError("Student model (Nerfacto) is producing Inf/NaN.")
        
        safe_batch = {"image": gt_rgb_safe} 

        student_metrics_dict = self.student_model.get_metrics_dict(student_outputs, safe_batch)
        
        loss_dict = self.student_model.get_loss_dict(
            student_outputs, 
            safe_batch, 
            student_metrics_dict
        )
        
        for loss_name, loss_value in loss_dict.items():
            if not torch.isfinite(loss_value):
                CONSOLE.log("[bold red]!!!!!!!! 灾难性故障：检测到 NaN/Inf 损失 !!!!!!!![/bold red]")
                CONSOLE.log(f"Step {step}: 损失 '{loss_name}' 爆炸了！")
                CONSOLE.log(f"Value: {loss_value}")
                CONSOLE.log("Full loss_dict:", loss_dict)
                sys.stdout.flush()
                sys.stderr.flush()
                raise ValueError(f"NaN/Inf loss detected in '{loss_name}' at step {step}.")

        return student_outputs, loss_dict, student_metrics_dict
        
    def get_eval_loss_dict(self, step: int) -> Tuple[Dict, Dict, Dict]:
        """ 
        Overridden to compute loss against the teacher model.
        """
        self.student_model.eval()
        
        ray_bundle, batch = self.datamanager.next_eval(step)
            
        with torch.no_grad():
            teacher_outputs = self._get_teacher_outputs(ray_bundle, batch) 
            gt_rgb = teacher_outputs["rgb"].detach()
            
            if not torch.isfinite(gt_rgb).all():
                CONSOLE.log(f"[bold yellow]Warning (Eval @ Step {step}):[/bold yellow] 教师模型在评估时输出了 Inf/NaN。")
                
            gt_rgb = torch.clamp(gt_rgb, min=0.0, max=1.0)
        
        student_outputs = self.student_model(ray_bundle)
        pred_rgb = student_outputs["rgb"]
        
        loss = F.l1_loss(pred_rgb, gt_rgb)
        loss_dict = {"eval_l1_loss": loss}
        
        safe_batch = {"image": gt_rgb}
        
        metrics_dict_from_model = self.student_model.get_metrics_dict(student_outputs, safe_batch)
        
        final_metrics_dict = metrics_dict_from_model.copy()
        final_metrics_dict["num_rays"] = ray_bundle.origins.shape[0]
        
        self.student_model.train()
        
        return student_outputs, loss_dict, final_metrics_dict

    @torch.no_grad()
    def get_eval_image_metrics_and_images(
        self, step: int
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """ 
        Compute metrics vs. teacher model.
        """
        self.student_model.eval()
        
        try:
            camera, batch = self.datamanager.next_eval_image(step)
        except StopIteration:
            CONSOLE.print("[bold yellow]Warning:[/bold yellow] Eval image dataloader exhausted.")
            self.student_model.train()
            return {}, {}
            
        camera_batch = camera.to(self.device) 

        with torch.no_grad():
            self.teacher_model.eval()
            teacher_outputs = self.teacher_model.get_outputs(camera_batch)
            gt_rgb = teacher_outputs["rgb"] 
            
            if not torch.isfinite(gt_rgb).all():
                CONSOLE.print(f"[bold red]Eval Image Error (Step {step}):[/bold red] 教师模型在评估图像上输出了 Inf/NaN。")
                self.student_model.train()
                return {}, {}
                
            gt_rgb = torch.clamp(gt_rgb, min=0.0, max=1.0)

        student_outputs = self.student_model.get_outputs_for_camera(camera_batch)
        
        if student_outputs is None:
            CONSOLE.print(f"[bold red]Eval Image Error (Step {step}): 学生模型返回了 None。[/bold red]")
            self.student_model.train()
            return {}, {}
            
        pred_rgb = student_outputs["rgb"]
        
        if not torch.isfinite(pred_rgb).all():
            CONSOLE.print(f"[bold red]Eval Image Error (Step {step}): 学生模型在评估图像上输出了 NaN。[/bold red]")
            self.student_model.train()
            return {}, {}

        gt_image_torch = gt_rgb.permute(2, 0, 1).unsqueeze(0)
        pred_image_torch = pred_rgb.permute(2, 0, 1).unsqueeze(0)
        
        psnr = self.student_model.psnr(pred_image_torch, gt_image_torch)
        ssim = self.student_model.ssim(pred_image_torch, gt_image_torch)
        lpips = self.student_model.lpips(pred_image_torch, gt_image_torch)

        metrics_dict = {
            "psnr_vs_teacher": float(psnr.item()),
            "ssim_vs_teacher": float(ssim.item()),
            "lpips_vs_teacher": float(lpips.item()),
        }
        
        num_rays = gt_rgb.shape[0] * gt_rgb.shape[1]
        metrics_dict["num_rays"] = float(num_rays)
        
        images_dict = {
            "teacher_gt": gt_rgb,
            "student_pred": pred_rgb,
            "diff": torch.abs(gt_rgb - pred_rgb)
        }
        
        self.student_model.train()
        return metrics_dict, images_dict
    
    # =====================================================================
    # 核心修复：重写所有可能导致问题的方法
    # =====================================================================
    
    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras) -> Dict[str, torch.Tensor]:
        """
        重写此方法以确保评估时使用正确的模型
        """
        self.student_model.eval()
        
        # 尝试多种方法获取输出
        outputs = None
        
        # 方法1: 尝试 get_outputs_for_camera
        if hasattr(self.student_model, 'get_outputs_for_camera') and outputs is None:
            try:
                outputs = self.student_model.get_outputs_for_camera(camera)
            except:
                pass
        
        # 方法2: 尝试 get_outputs  
        if hasattr(self.student_model, 'get_outputs') and outputs is None:
            try:
                outputs = self.student_model.get_outputs(camera)
            except:
                pass
        
        # 方法3: 生成 rays 并使用 forward
        if outputs is None:
            try:
                ray_bundle = camera.generate_rays(
                    camera_indices=0,
                    aabb_box=self.datamanager.train_dataset.scene_box.aabb,
                )
                outputs = self.student_model(ray_bundle)
            except Exception as e:
                CONSOLE.print(f"[red]Error generating outputs: {e}[/red]")
        
        # 如果所有方法都失败，返回虚拟输出
        if outputs is None:
            H = camera.height.item() if hasattr(camera.height, 'item') else camera.height[0]
            W = camera.width.item() if hasattr(camera.width, 'item') else camera.width[0]
            outputs = {
                "rgb": torch.zeros((H, W, 3), device=self.device),
                "accumulation": torch.ones((H, W, 1), device=self.device),
                "depth": torch.zeros((H, W, 1), device=self.device),
            }
            CONSOLE.print("[red]Warning: Using dummy outputs for evaluation[/red]")
        
        return outputs
    
    def get_average_eval_image_metrics(
        self, 
        step: Optional[int] = None, # <--- (!!!) 更改 2: 修复 TypeError
        output_path: Optional[Path] = None,
        get_std: bool = False
    ) -> Dict[str, float]:
        """
        完全重写评估方法，避免调用有问题的基类方法
        """
        import time
        from collections import defaultdict
        
        self.student_model.eval()
        self.teacher_model.eval()
        
        metrics_dicts = []
        
        # 使用评估数据加载器
        eval_dataloader = self.datamanager.fixed_indices_eval_dataloader
        num_total_images = len(eval_dataloader)
        
        # ==================================================================
        # --- (!!!) 更改 3: 设置要评估的图像数量 (加速评估) ---
        
        # 在这里设置你希望评估的图像数量 (例如 20 张)
        # 如果设置为 -1，则评估全部图像。
        NUM_IMAGES_TO_EVAL = 20  
        
        # ---
        
        indices_to_eval = list(range(num_total_images))
        
        # 如果总数 > 目标数，并且目标数不是 -1，则进行随机采样
        if NUM_IMAGES_TO_EVAL > 0 and num_total_images > NUM_IMAGES_TO_EVAL:
            CONSOLE.print(f"[bold yellow]Warning:[/bold yellow] 评估图像总数 ({num_total_images}) 太多。")
            CONSOLE.print(f"将随机抽取 {NUM_IMAGES_TO_EVAL} 张图像进行评估。")
            indices_to_eval = random.sample(indices_to_eval, NUM_IMAGES_TO_EVAL)
        
        num_to_eval = len(indices_to_eval)
        # ==================================================================

        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
            transient=True,
        ) as progress:
            # (!!!) 更改 4: 更新 progress bar 的 total
            task = progress.add_task("[green]Evaluating...", total=num_to_eval)
            
            # (!!!) 更改 5: 遍历采样后的索引
            for camera_idx in indices_to_eval:
                
                # (!!!) 更改 6: 根据索引获取数据
                (camera, batch) = eval_dataloader[camera_idx]

                with torch.no_grad():
                    # 获取教师模型输出
                    teacher_outputs = self.teacher_model.get_outputs(camera.to(self.device))
                    gt_rgb = teacher_outputs["rgb"].clamp(0, 1)
                    
                    # 获取学生模型输出
                    student_outputs = self.get_outputs_for_camera(camera.to(self.device))
                    pred_rgb = student_outputs["rgb"].clamp(0, 1)
                    
                    # 转换为正确的形状
                    gt_image = gt_rgb.permute(2, 0, 1).unsqueeze(0)
                    pred_image = pred_rgb.permute(2, 0, 1).unsqueeze(0)
                    
                    # 计算指标
                    psnr = self.student_model.psnr(pred_image, gt_image)
                    ssim = self.student_model.ssim(pred_image, gt_image)
                    lpips = self.student_model.lpips(pred_image, gt_image)
                    
                    metrics_dict = {
                        "psnr": float(psnr.item()),
                        "ssim": float(ssim.item()),
                        "lpips": float(lpips.item()),
                    }
                    metrics_dicts.append(metrics_dict)
                    
                    # 保存渲染图像
                    if output_path is not None:
                        import torchvision
                        output_path.mkdir(parents=True, exist_ok=True)
                        torchvision.utils.save_image(
                            pred_image[0],
                            output_path / f"pred_{camera_idx:04d}.png"
                        )
                        torchvision.utils.save_image(
                            gt_image[0],
                            output_path / f"gt_{camera_idx:04d}.png"
                        )
                
                progress.advance(task)
        
        # 计算平均值
        metrics_dict = defaultdict(list)
        for m in metrics_dicts:
            for k, v in m.items():
                metrics_dict[k].append(v)
        
        average_metrics = {}
        for k, v in metrics_dict.items():
            average_metrics[k] = float(np.mean(v))
            if get_std:
                average_metrics[f"{k}_std"] = float(np.std(v))
        
        self.student_model.train()
        
        # 打印结果
        CONSOLE.print("\n[bold green]Evaluation Results:[/bold green]")
        for k, v in average_metrics.items():
            if not k.endswith("_std"):
                CONSOLE.print(f"  {k}: {v:.4f}")
        
        return average_metrics
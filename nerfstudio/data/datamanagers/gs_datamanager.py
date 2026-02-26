# Copyright 2022 The Nerfstudio Team. All rights reserved.
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

"""
DataManager for loading 3DGS 'cameras.json' files.
This DataManager is designed for GS-to-NeRF distillation pipelines.
It correctly calculates the scene box based on transformed camera poses,
resolving NaN issues when using Nerfacto with 3DGS data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, Dict, List, Tuple, Union, Optional

import torch
from torch.nn import Parameter
from rich.console import Console

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nerfstudio.data.scene_box import SceneBox
# 这是我们最终找到的正确导入路径
from nerfstudio.model_components.ray_generators import RayGenerator

CONSOLE = Console(highlight=False)

@dataclass
class GSDataManagerConfig(DataManagerConfig):
    """为 3DGS 蒸馏定制的 DataManager 配置。"""
    
    _target: Type = field(default_factory=lambda: GSDataManager)
    """目标类"""

    train_num_rays_per_batch: int = 4096
    """Number of rays to sample per batch for training."""
    
    eval_num_rays_per_batch: int = 4096
    """Number of rays to sample per batch for evaluation."""
    
    cameras_json_path: Path = Path("cameras.json")
    """(必需) 指向 3DGS 导出的 cameras.json 文件的路径。"""
    
    gs_to_nerf_world_transform: List[List[float]] = field(
        default_factory=lambda: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    """将 3DGS (OpenCV) 坐标系转换为 Nerfstudio (OpenGL) 坐标系的矩阵。"""
    
    scene_box_scale_margin: float = 1.5
    """在根据相机计算出的边界框基础上，再额外放大的倍数。"""

    assume_colmap_center: bool = False
    """
    如果为 True, 假设 cameras.json 中的 'cx', 'cy' 是主点。
    如果为 False, 假设主点在图像中心 (width/2, height/2)。
    """


class GSDataManager(DataManager):
    """
    这个 DataManager 专门用于从 3DGS 的 'cameras.json' 加载场景，
    以便为学生模型 (nerfacto) 提供正确的场景边界和光线。
    """

    config: GSDataManagerConfig
    train_cameras: Cameras
    eval_cameras: Cameras
    train_ray_generator: RayGenerator
    eval_ray_generator: RayGenerator

    # 为 'next_train' 缓存这些
    train_camera_heights: torch.Tensor
    train_camera_widths: torch.Tensor
    train_num_cameras: int
    eval_camera_heights: torch.Tensor
    eval_camera_widths: torch.Tensor
    eval_num_cameras: int
    
    # (!!!) 修复：添加 Pipeline 所需的属性
    fixed_indices_eval_dataloader: List[Tuple[Cameras, Dict]]


    def __init__(
        self,
        config: GSDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.test_mode = test_mode

        # 1. 加载并转换相机数据
        (
            c2w_list_nerf,
            intrinsics_list,
            camera_origins_nerf,
        ) = self._load_and_transform_cameras()

        # 2. (!!!) 关键修复：根据已转换的相机计算 SceneBox
        self.scene_box = self._compute_scene_box_from_cameras(
            camera_origins_nerf
        )
        
        # 3. 创建 Nerfstudio 的 Cameras 对象
        self.train_cameras = self._create_cameras_object(
            c2w_list_nerf, intrinsics_list
        )
        
        # 4. 设置训练和评估相机
        self.eval_cameras = self.train_cameras

        # 5. (!!!) 修复所有 AttributeError
        num_cameras = len(self.train_cameras)

        class MockDataset:
            """一个临时的模拟对象，用于将所有必需的属性传递给 Pipeline。"""
            def __init__(self, scene_box: SceneBox, num_cameras: int, cameras: Cameras):
                self.scene_box = scene_box
                self._num_cameras = num_cameras
                self.metadata: Dict = {} 
                self.cameras = cameras
            
            def __len__(self):
                return self._num_cameras

            # ----------------------------------------------------------
            # (!!!) 解决方案：添加这个方法来修复 'TypeError'
            # ----------------------------------------------------------
            def __getitem__(self, idx: int) -> Dict:
                """
                这个方法是为了“欺骗” Viewer (监视器) 的
                _init_viewer_state() 调用。
                
                它不需要返回“真”的 3DGS 渲染图，那太慢了。
                我们只需要返回一个正确形状的“黑色”虚拟图像。
                """
                
                # 从相机获取 H 和 W
                H = self.cameras.height[idx].item()
                W = self.cameras.width[idx].item()
                
                # 创建一个 (H, W, 3) 的黑色张量
                dummy_image = torch.zeros((H, W, 3), dtype=torch.float32)
                
                # 返回 Viewer 期望的字典
                return {
                    "image": dummy_image,
                    "image_idx": idx
                }
            # ----------------------------------------------------------
            # (!!!) 修复结束
            # ----------------------------------------------------------

        self.train_dataset = MockDataset(self.scene_box, num_cameras, self.train_cameras)
        self.eval_dataset = MockDataset(self.scene_box, num_cameras, self.eval_cameras)

        # 6. (!!!) 最后调用 super().__init__()
        super().__init__()

    # ------------------------------------------------------------------
    # --- 辅助加载函数 (在 __init__ 中调用) ---
    # ------------------------------------------------------------------

    def _load_and_transform_cameras(self) -> Tuple[List[torch.Tensor], List[Dict], torch.Tensor]:
        """加载 'cameras.json'，应用坐标系转换。"""
        try:
            with open(self.config.cameras_json_path, "r") as f:
                camera_data_3dgs = json.load(f)
        except FileNotFoundError:
            CONSOLE.print(f"[bold red]错误: 'cameras.json' not found at {self.config.cameras_json_path}")
            raise
            
        c2w_list_nerf = []
        intrinsics_list = []
        camera_origins_nerf_list = []

        transform_matrix = torch.tensor(
            self.config.gs_to_nerf_world_transform, dtype=torch.float32
        )

        for cam in camera_data_3dgs:
            r_mat = torch.tensor(cam["rotation"], dtype=torch.float32)
            t_vec = torch.tensor(cam["position"], dtype=torch.float32)
            c2w_3dgs = torch.eye(4, dtype=torch.float32)
            c2w_3dgs[:3, :3] = r_mat
            c2w_3dgs[:3, 3] = t_vec
            
            c2w_nerf = c2w_3dgs @ transform_matrix
            c2w_list_nerf.append(c2w_nerf)
            camera_origins_nerf_list.append(c2w_nerf[:3, 3])

            if self.config.assume_colmap_center:
                cx = cam["cx"]
                cy = cam["cy"]
            else:
                cx = cam["width"] / 2.0
                cy = cam["height"] / 2.0
                
            intrinsics_list.append({
                "fx": cam["fx"], "fy": cam["fy"], "cx": cx, "cy": cy,
                "W": cam["width"], "H": cam["height"],
            })
            
        return (
            c2w_list_nerf,
            intrinsics_list,
            torch.stack(camera_origins_nerf_list),
        )

    def _compute_scene_box_from_cameras(self, camera_origins_nerf: torch.Tensor) -> SceneBox:
        """(!!!) 这是修复 NaN 的核心：根据*已转换*的相机计算场景边界。"""
        min_bounds = torch.min(camera_origins_nerf, dim=0)[0]
        max_bounds = torch.max(camera_origins_nerf, dim=0)[0]
        center = (min_bounds + max_bounds) / 2.0
        size = max_bounds - min_bounds
        extents = (size.max() * self.config.scene_box_scale_margin) / 2.0
        if extents < 1e-6:
            extents = 1.0 
        aabb_min = center - extents
        aabb_max = center + extents
        
        return SceneBox(aabb=torch.stack([aabb_min, aabb_max]))


    def _create_cameras_object(
        self, c2w_list: List[torch.Tensor], intrinsics_list: List[Dict]
    ) -> Cameras:
        """将加载的数据打包成 Nerfstudio 的 'Cameras' 对象。"""
        c2w_tensor_4x4 = torch.stack(c2w_list) 
        c2w_tensor_3x4 = c2w_tensor_4x4[:, :3, :] # -> 形状 [N, 3, 4]

        fx = torch.tensor([d["fx"] for d in intrinsics_list], dtype=torch.float32)
        fy = torch.tensor([d["fy"] for d in intrinsics_list], dtype=torch.float32)
        cx = torch.tensor([d["cx"] for d in intrinsics_list], dtype=torch.float32)
        cy = torch.tensor([d["cy"] for d in intrinsics_list], dtype=torch.float32)
        W = torch.tensor([d["W"] for d in intrinsics_list], dtype=torch.int32)
        H = torch.tensor([d["H"] for d in intrinsics_list], dtype=torch.int32)

        return Cameras(
            camera_to_worlds=c2w_tensor_3x4, 
            fx=fx, fy=fy, cx=cx, cy=cy, width=W, height=H,
            camera_type=CameraType.PERSPECTIVE
        )

    # ------------------------------------------------------------------
    # --- 实现 DataManager 的抽象方法 ---
    # ------------------------------------------------------------------

    def setup_train(self):
        """
        模仿 VanillaDataManager.setup_train()。
        """
        assert self.train_cameras is not None
        CONSOLE.print("[GSDataManager] Setting up training ray generator...")
        self.train_ray_generator = RayGenerator(self.train_cameras.to(self.device))
        
        self.train_camera_heights = self.train_cameras.height.cpu()
        self.train_camera_widths = self.train_cameras.width.cpu()
        self.train_num_cameras = len(self.train_cameras)

    def setup_eval(self):
        """
        模仿 VanillaDataManager.setup_eval()。
        """
        assert self.eval_cameras is not None
        CONSOLE.print("[GSDataManager] Setting up evaluation ray generator...")
        self.eval_ray_generator = RayGenerator(self.eval_cameras.to(self.device))
        
        self.eval_camera_heights = self.eval_cameras.height.cpu()
        self.eval_camera_widths = self.eval_cameras.width.cpu()
        self.eval_num_cameras = len(self.eval_cameras)

        # -----------------------------------------------------------------
        # (!!!) 修复 AssertionError: ... no attribute 'fixed_indices_eval_dataloader'
        #     我们在这里手动构建一个列表，以满足 Pipeline 的要求。
        # -----------------------------------------------------------------
        CONSOLE.print(f"[GSDataManager] Pre-building {self.eval_num_cameras} eval image batches...")
        self.fixed_indices_eval_dataloader = []
        for i in range(self.eval_num_cameras):
            # next_eval_image 返回 (camera, batch) 元组
            camera, batch = self.next_eval_image(i)
            self.fixed_indices_eval_dataloader.append((camera, batch))


    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """
        实现 DataManager.next_train()。
        返回一个随机光线束和一个空字典。
        """
        self.train_count += 1
        batch_size = self.config.train_num_rays_per_batch
        
        assert self.train_ray_generator is not None

        # (!!!) 修复：所有这些张量都在 CPU 上创建
        
        # ==================================================================
        # --- 关键修复：切换到基于图像的采样 (Image-based Sampling) ---
        #     这确保一个批次(batch)中的所有光线都来自同一个相机，
        #     以匹配 Pipeline 中 _get_teacher_outputs 的渲染逻辑。
        # ==================================================================
        
        # 1. 随机选择 *一个* 相机索引
        camera_idx = torch.randint(0, self.train_num_cameras, (1,)).item()
        
        # 2. 获取这个相机的 H 和 W
        height = self.train_camera_heights[camera_idx].item()
        width = self.train_camera_widths[camera_idx].item()
        
        # 3. 为这个 *单独的* 相机生成 batch_size 个坐标
        y_coords = torch.floor(torch.rand(batch_size) * height).long()
        x_coords = torch.floor(torch.rand(batch_size) * width).long()
        
        # 4. 创建一个张量，其中 *所有* 索引都是这一个 camera_idx
        camera_indices = torch.full((batch_size,), camera_idx, dtype=torch.long)
        
        # ==================================================================
        # --- 修复结束 ---
        # ==================================================================
        
        ray_indices = torch.stack([camera_indices, y_coords, x_coords], dim=-1)

        ray_bundle = self.train_ray_generator(ray_indices)
        
        ray_bundle.metadata["ray_indices"] = ray_indices.to(self.device)
        
        return ray_bundle, {}

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """
        实现 DataManager.next_eval()。
        返回一个随机光线束和一个空字典。
        """
        self.eval_count += 1
        batch_size = self.config.eval_num_rays_per_batch
        
        assert self.eval_ray_generator is not None

        # ==================================================================
        # --- 关键修复：同样为 next_eval 切换到基于图像的采样 ---
        #     这确保了 get_eval_loss_dict 也能正确工作。
        # ==================================================================
        
        # 1. 随机选择 *一个* 评估相机索引
        camera_idx = torch.randint(0, self.eval_num_cameras, (1,)).item()

        # 2. 获取这个相机的 H 和 W
        height = self.eval_camera_heights[camera_idx].item()
        width = self.eval_camera_widths[camera_idx].item()

        # 3. 为这个 *单独的* 相机生成 batch_size 个坐标
        y_coords = torch.floor(torch.rand(batch_size) * height).long()
        x_coords = torch.floor(torch.rand(batch_size) * width).long()

        # 4. 创建一个张量，其中 *所有* 索引都是这一个 camera_idx
        camera_indices = torch.full((batch_size,), camera_idx, dtype=torch.long)
        
        # ==================================================================
        # --- 修复结束 ---
        # ==================================================================
        
        ray_indices = torch.stack([camera_indices, y_coords, x_coords], dim=-1)

        ray_bundle = self.eval_ray_generator(ray_indices)
        
        ray_bundle.metadata["ray_indices"] = ray_indices.to(self.device)
        
        return ray_bundle, {}

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """
        实现 DataManager.next_eval_image()。
        返回下一个评估相机和空字典。
        """
        # (!!!) 注意：我们不再使用 'step'，而是直接索引
        # 这是因为 'step' 可能是 500, 1000 等
        # 而 'i' (在 setup_eval 中) 是 0, 1, 2...
        camera_idx = step % len(self.eval_cameras)
        camera = self.eval_cameras[camera_idx : camera_idx + 1]
        
        return camera, {}

    def get_train_rays_per_batch(self) -> int:
        """实现 DataManager.get_train_rays_per_batch()。"""
        return self.config.train_num_rays_per_batch

    def get_eval_rays_per_batch(self) -> int:
        """实现 DataManager.get_eval_rays_per_batch()。"""
        return self.config.eval_num_rays_per_batch

    def get_datapath(self) -> Path:
        """实现 DataManager.get_datapath()。"""
        return self.config.cameras_json_path.parent

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """实现 DataManager.get_param_groups()。"""
        return {}
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

# nerfstudio/utils/gs_loader.py
"""
Utility file to load a .ply file from the original 3D Gaussian Splatting
repository into a format compatible with nerfstudio's SplatfactoModel.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from plyfile import PlyData

from nerfstudio.utils.rich_utils import CONSOLE

def load_gs_from_ply(ply_path: Path, max_sh_degree: int) -> Dict[str, torch.Tensor]:
    """
    Load Gaussian Splatting parameters from a .ply file.

    Args:
        ply_path: Path to the .ply file.
        max_sh_degree: The maximum spherical harmonic degree to load.

    Returns:
        A dictionary containing the loaded Gaussian parameters.
    """
    CONSOLE.print(f"[bold green]✓ Loading PLY file[/bold green] from [cyan]{ply_path}[/cyan]")
    
    plydata = PlyData.read(str(ply_path))
    elements = plydata.elements[0]

    # --- 1. Load Positions (means) ---
    xyz = np.stack((
        np.asarray(elements["x"]),
        np.asarray(elements["y"]),
        np.asarray(elements["z"])
    ), axis=1)
    
    # --- 2. Load Opacities ---
    # splatfacto expects pre-sigmoid (logit) opacities
    opacities = np.asarray(elements["opacity"])[..., np.newaxis]

    # --- 3. Load Features DC (SH 0) ---
    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(elements["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(elements["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(elements["f_dc_2"])

    # --- 4. Load Features Rest (SH > 0) ---
    extra_f_names = [p.name for p in elements.properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    
    num_sh_features = (max_sh_degree + 1) ** 2 - 1
    
    # 修复：确保即使sh_degree为0（仅DC），也能正确处理
    if num_sh_features > 0:
        if len(extra_f_names) != 3 * num_sh_features:
            raise ValueError(
                f"PLY file SH degree ({len(extra_f_names) // 3}) mismatch with "
                f"configured SplatfactoModel SH degree ({num_sh_features})"
            )

        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(elements[attr_name])
        
        features_rest = features_extra.reshape((features_extra.shape[0], 3, num_sh_features))
    else:
        # 如果 sh_degree 为 0，则 features_rest 为空
        features_rest = np.zeros((xyz.shape[0], 3, 0))


    # --- 5. Load Scales ---
    # splatfacto expects pre-exponent (log) scales
    scale_names = [p.name for p in elements.properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(elements[attr_name])

    # --- 6. Load Rotations (quaternions) ---
    # splatfacto expects pre-normalization quaternions
    rot_names = [p.name for p in elements.properties if p.name.startswith("rot_")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(elements[attr_name])

    CONSOLE.print(f"[bold green]✓ Successfully loaded[/bold green] {xyz.shape[0]} Gaussians.")

    # Convert to Tensors
    return {
        "means": torch.tensor(xyz, dtype=torch.float32),
        "opacities": torch.tensor(opacities, dtype=torch.float32),
        "features_dc": torch.tensor(features_dc, dtype=torch.float32).transpose(1, 2).contiguous(),
        "features_rest": torch.tensor(features_rest, dtype=torch.float32).transpose(1, 2).contiguous(),
        "scales": torch.tensor(scales, dtype=torch.float32),
        "quats": torch.tensor(rots, dtype=torch.float32),
    }
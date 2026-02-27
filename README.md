# Phase 2: Online Knowledge Distillation (3DGS ➡️ NeRF)

> **MSc Robotics & Artificial Intelligence Project**
> **Focus:** Real-time 2D Supervision from an explicit 3DGS Teacher to an implicit NeRF Student.

## 📌 Overview
This branch explores **Online Knowledge Distillation** within the Nerfstudio framework. The goal is to compress a massive, discrete 3D Gaussian Splatting (3DGS) map into a compact NeRF continuous density field ($\sigma$) by dynamically rendering 2D pseudo-ground-truth images from a frozen 3DGS "Teacher" to supervise a NeRF "Student".

## 🛠️ Core Implementations

Instead of modifying the underlying models, the core engineering in this phase focused on the data pipeline and framework infrastructure:

* **`gs_datamanager.py`**: Handles the crucial cross-framework coordinate alignment (converting 3DGS OpenCV format to NeRF OpenGL format) and dynamically computes valid NeRF SceneBoxes from transformed 3DGS camera poses to prevent NaN gradient explosions.
* **`gs_distill_pipeline.py`**: Implements the dual-model Teacher-Student loop. It bypasses standard data loading to inject pre-trained 3DGS `.ply` parameters directly into memory, rendering pseudo-GT views on-the-fly to calculate L1/SSIM losses for the NeRF student.

## 📊 Results

The model successfully converged quantitatively, achieving a **PSNR > 25**. However, the qualitative visual results suffered from noticeable blurriness and lacked high-frequency geometric details.

<p align="center">
  <img src="[你的图片链接1：例如训练曲线或PSNR指标图]" alt="Training Metrics" width="30%">
  <img src="[你的图片链接2：NeRF渲染出的模糊视图]" alt="Blurry Novel View" width="30%">
  <img src="[你的图片链接3：3DGS真值视角的对比图]" alt="3DGS Ground Truth Comparison" width="30%">
  <br>
  <em>Fig 1. Quantitative success (PSNR > 25) vs. Qualitative blurriness in reconstructed geometry.</em>
</p>

## 🛑 The Engineering Bottleneck: Why the Blur?

While the PSNR metric seems acceptable, the resulting 3D geometry is blurry. This is caused by a fundamental architectural clash that forced a compromise in our ray sampling strategy:

1. **The Multi-Camera Ideal:** A standard NeRF samples a batch of rays (e.g., 4096) randomly across *all* available cameras simultaneously. This provides strong global multi-view constraints in a single optimization step, allowing the MLP to triangulate sharp 3D structures.
2. **The 3DGS VRAM Wall:** 3DGS uses tile-based rasterization and must render *full images*. If we sampled rays from 100 random cameras, the 3DGS Teacher would be forced to rasterize 100 full-resolution images in a single forward pass—causing an immediate Out-Of-Memory (OOM) crash.
3. **The Compromise (Image-Based Sampling):** To prevent OOM, the datamanager was modified to sample all 4096 rays from only **one single camera per batch**.
4. **The Consequence:** Because the NeRF MLP updates its weights based on a single viewpoint at a time, it suffers from micro-scale catastrophic forgetting. It perfectly memorizes the 2D color of that specific view (hence the high PSNR) but fails to lock in a globally consistent, sharp 3D geometry, resulting in a "soft" and blurry volume.

## ➡️ Next Steps
The VRAM limitations of online rasterization-raytracing synchronization make sharp 3D reconstruction unfeasible on consumer hardware. This finding directly motivates **Phase 3**, where we pivot to **Offline Cached Distillation** combined with **Depth Supervision** to strictly constrain the NeRF geometry.

# Phase 4 — 代码、数据集与评测生态

检索时间：2026-07-27。Stars/Forks 为检索当日页面快照，会随时间变化；其作用只是粗略反映社区关注度，不能替代代码质量审查。

## 1. FirasGit/medicaldiffusion

- URL: https://github.com/FirasGit/medicaldiffusion
- 对应论文：Medical Diffusion: Denoising Diffusion Probabilistic Models for 3D Medical Image Generation [@khader2023denoising]
- 方法：先训练三维 VQ-GAN，将体数据压缩至潜空间，再训练 3D DDPM；覆盖 BraTS、LIDC-IDRI、胸部 CT 与组织病理数据。
- 技术栈：Python 3.8、PyTorch/PyTorch Lightning、Hydra；仓库含 `vq_gan_3d`、`ddpm`、训练与评测脚本。
- 社区快照：494 stars、79 forks、20 commits。
- 可复现性判断：README 给出环境、VQ-GAN 与 DDPM 的完整训练命令，也支持自定义 NIfTI 文件夹；但依赖与 Python 版本偏旧，且复现实验需要较大显存和相应受控数据。
- 适合用途：理解早期三维医学潜空间扩散的最小可运行实现。

## 2. NVIDIA-Medtech/NV-Generate-CTMR

- URL: https://github.com/NVIDIA-Medtech/NV-Generate-CTMR
- 对应论文：MAISI v1 [@guo2025maisi]，以及仓库所列 MAISI v2（AAAI 2026）。
- 方法：高分辨率三维 CT/MRI 潜扩散与整流流模型；可生成 CT 图像—132 类分割掩膜对，并支持多对比度 MRI、体素间距、体积、器官与肿瘤大小控制。
- 技术栈：Python、PyTorch、MONAI；含配置、训练/推理脚本、模型权重、教程 notebook、性能与数据准备文档。
- 社区快照：约 195 stars、33 forks、133 commits；README 显示更新延续至 2026-05。
- 可复现性判断：四个公开模型变体覆盖 `ddpm-ct` 与 `rflow-*`，有快速开始和模型下载；最低需要约 16 GB GPU。整流流版本采用 30 步推理，相比 DDPM 的 1000 步，仓库报告 `rflow-ct` 快约 33 倍。
- 适合用途：目前最完整的三维 CT/MRI 生成工程基线之一，也是“扩散 → flow matching/rectified flow”迁移的直接例证。

## 3. Melon-Xu/TopoCellGen

- URL: https://github.com/Melon-Xu/TopoCellGen
- 对应论文：TopoCellGen: Generating Histopathology Cell Topology with a Diffusion Model [@xu2025topocellgen]
- 方法：面向组织病理细胞布局生成，在扩散过程中显式建模细胞计数、类别与拓扑关系，并提出 Topological Fréchet Distance（TopoFD）。
- 技术栈：Python 3.8、PyTorch 1.11、CUDA 11.3；MIT 许可证。
- 社区快照：28 stars、0 forks、10 commits。
- 可复现性判断：提供 BRCA-M2C 布局生成脚本、常规评价与 TopoFD 脚本、预训练权重，并链接 BRCA-M2C 和 Lizard 数据；依赖较旧，属于论文型轻量代码库。
- 适合用途：研究普通 FID 难以刻画的病理空间拓扑真实性。

## 4. cian-unibas-ch/diffusion-anomaly

- URL: https://gitlab.com/cian.unibas.ch/diffusion-anomaly
- 对应论文：Diffusion Models for Medical Anomaly Detection [@wolleb2022diffusion]
- 方法：用健康图像训练扩散先验，推理时通过 DDIM 反演与梯度引导，将疑似异常图像映射为健康对应物，以残差定位病灶。
- 技术栈：Python/PyTorch；Apache-2.0 许可证。
- 活跃度快照：55 commits；项目创建于 2022-06-27。
- 可复现性判断：代码直接对应论文，但文档与维护活跃度弱于 NVIDIA 工程仓库；更适合复现“生成式异常检测”概念，而非作为通用生成平台。

## 5. 常用数据集

| 场景 | 代表数据集 | 主要限制 |
|---|---|---|
| 脑 MRI 与肿瘤 | BraTS、IXI、UK Biobank、ADNI、OASIS | UK Biobank/ADNI 等需申请；纵向资料存在失访与人群偏倚 |
| 胸片 | MIMIC-CXR、CheXpert、ChestX-ray14、OpenI/IU X-Ray | 报告标签含噪；设备、体位和医院域偏移明显 |
| CT/多器官 | LIDC-IDRI、Medical Segmentation Decathlon、autoPET | 三维体积昂贵；重建核、剂量与层厚差异大 |
| 心脏与腹部 | ACDC、SLIVER07 | 样本规模较小，器官/病变覆盖有限 |
| 组织病理 | BRCA-M2C、Lizard、CRC/CRCMS | 染色、扫描仪和中心差异显著；拓扑与细胞级标注昂贵 |

## 6. 评测基准：应使用多层证据，而不是单一 FID

1. **分布层**：FID/KID、precision/recall、密度/覆盖度；医学图像上应优先使用医学域表征，并报告置信区间。
2. **像素/配对层**：PSNR、SSIM、MAE、LPIPS，适合有配对真值的跨模态合成或重建，不适合评价无条件生成的多样性。
3. **结构层**：器官体积、形态统计、表面距离、拓扑指标（例如 TopoFD）以及 3D 切片间一致性。
4. **临床任务层**：用合成数据训练或增强后的分割 Dice、检测灵敏度、分类 AUC，并在独立外部队列验证。
5. **临床读片层**：盲法、多阅片者、分层病种的真实性与诊断正确性评估，报告阅片者间一致性。
6. **安全层**：最近邻/重复样本检查、成员推断、属性泄露、少数群体与罕见病变覆盖、反事实因果合理性。

## Phase 4 gate

- 代码库数量：4（要求至少 3）。
- 每个条目均记录 URL、论文、方法、技术栈、活跃度与可复现性判断。
- 已覆盖数据、通用质量指标、结构指标、下游临床效用和隐私安全评测。

# Phase 3 — Full-text deep dive

## [@nie2017medical] Medical Image Synthesis with Context-Aware Generative Adversarial Networks

**Metadata**
- Authors: Dong Nie et al.
- Year/Venue: 2017, MICCAI
- Full text: https://ar5iv.labs.arxiv.org/html/1612.05362

**Problem**  
Generate CT volumes from paired MRI to reduce reliance on ionizing-radiation CT for radiotherapy planning and PET attenuation correction.

**Key contributions**
1. An early 3D fully convolutional generator for paired MRI→CT translation.
2. Combines reconstruction, adversarial and 3D gradient-difference losses.
3. Uses iterative auto-context refinement to compensate for the limited field of view of patch training.

**Methodology**
- The generator is an eight-stage 3D FCN without pooling; the discriminator classifies real versus synthetic CT.
- Generator loss combines adversarial binary cross-entropy, L2 reconstruction and gradient-difference penalties along x/y/z.
- At inference, overlapping MRI patches are translated and averaged; subsequent auto-context stages concatenate the previous synthetic CT as an added input channel.

**Experiments**
- Datasets: paired MRI/CT from 16 brain subjects (ADNI) and 22 pelvic subjects.
- Baselines: atlas warping, sparse representation and structured random forest with auto-context.
- Metrics: MAE and PSNR with leave-one-out evaluation.
- Main results: adversarial training raised PSNR from 24.7 to 25.9 before auto-context. On brain data the proposed method reached MAE 92.5 and PSNR 27.6 versus 99.9/26.3 for the strongest SRF+ baseline; on pelvis it reached MAE 39.0 and PSNR 34.1 versus 48.1/32.1.

**Limitations**
- Very small, aligned paired cohorts.
- Patch aggregation and registration quality constrain fidelity.
- PSNR/MAE do not test whether subtle pathology is preserved.

**Connections**  
It operationalizes the generic conditional GAN idea [@isola2017image] in 3D medicine and precedes unpaired CycleGAN adaptations [@wolterink2017deep; @hiasa2018cross].

## [@cohen2018distribution] Distribution Matching Losses Can Hallucinate Features in Medical Image Translation

**Metadata**
- Authors: Joseph Paul Cohen, Margaux Luck, Sina Honari
- Year/Venue: 2018, MICCAI (oral)
- Full text: https://arxiv.org/pdf/1805.08841

**Problem**  
Test whether adversarial distribution matching preserves patient-specific disease features when source and target domain class proportions differ.

**Key contributions**
1. Formalizes the mismatch between target-distribution realism and sample-level clinical faithfulness.
2. Demonstrates both removal and addition of tumors under biased target distributions.
3. Distinguishes use for augmentation from unsafe direct clinical interpretation.

**Methodology**
- Compares unpaired CycleGAN, paired conditional GAN and paired L1 translation.
- Translates BRATS2013 FLAIR to T1 while varying the tumor proportion in the target domain from 0% to 100%.
- Uses a separate CNN disease classifier and pixel reconstruction error to quantify feature changes.

**Experiments**
- 1,700 BRATS slices: 1,400 for constructing training distributions and 300 held out.
- Three model families × 11 target class ratios; the experiment therefore trains 33 translation models.
- CycleGAN predictions shift most strongly with the tumor prevalence in the target domain; examples visibly remove tumors when the target is healthy-only and add tumors when it is tumor-only.
- Conditional GAN is more stable than CycleGAN but still shows distribution-dependent pixel error; L1 is least affected except under unseen disease content.

**Limitations**
- Deliberately exaggerated prevalence shifts and a synthetic/benchmark tumor setting.
- The disease classifier is only 80% accurate.
- The paper diagnoses the problem but does not provide a complete solution.

**Connections**  
This paper is the central caution against treating high-FID/low-FID or visually sharp synthesis as evidence of clinical correctness.

## [@pinaya2022unsupervised] Unsupervised Brain Anomaly Detection and Segmentation with Transformers

**Metadata**
- Authors: Walter H. L. Pinaya et al.
- Year/Venue: 2022, Medical Image Analysis
- Full text: https://ar5iv.labs.arxiv.org/html/2102.11650

**Problem**  
Learn a compact but expressive model of normal brain MRI so pathologies can be detected as low-likelihood latent codes and replaced with healthy alternatives.

**Key contributions**
1. Compresses images into discrete VQ-VAE codes.
2. Learns the healthy-code distribution with an ensemble of autoregressive Performer Transformers.
3. Uses multiple raster orderings and masked latent resampling to localize anomalies.

**Methodology**
- A VQ-VAE maps each image to a spatial grid of codebook indices.
- Autoregressive Transformers estimate each code conditional on previous codes.
- Low-likelihood codes are resampled; decoding yields a normative reconstruction, and the residual provides an anomaly map.
- Eight orientations/orderings are ensembled to reduce autoregressive ordering bias.

**Experiments**
- Synthetic experiment: MedNIST HeadCT with 100 sprite-contaminated test images.
- Real experiment: trained on 15,000 radiologically normal UK Biobank participants and evaluated on four datasets containing small-vessel disease, demyelinating lesions and tumors.
- On synthetic segmentation, best achievable Dice rose from 0.457 (VQ-VAE) to 0.675 with Transformer resampling, 0.768 with masked residuals, and 0.895 with eight orderings; the strongest conventional VAE baseline was 0.533.
- The paper also evaluates image-wise likelihood and pixel-wise anomaly segmentation across real lesions.

**Limitations**
- Autoregressive likelihood depends on ordering; eight-model ensembles add compute.
- “Likelihood” may emphasize nuisance variation rather than disease.
- Residual threshold tuning and normative-training demographics affect performance.

**Connections**  
This is the clearest medical example of the VQ-VAE + autoregressive school [@oord2017neural], distinct from GAN and diffusion despite sharing an autoencoder.

## [@wolleb2022diffusion] Diffusion Models for Medical Anomaly Detection

**Metadata**
- Authors: Julia Wolleb et al.
- Year/Venue: 2022, MICCAI
- Full text: https://arxiv.org/pdf/2203.04306
- Code: https://gitlab.com/cian.unibas.ch/diffusion-anomaly

**Problem**  
Translate a diseased image into a detail-preserving healthy counterpart using only image-level labels, then use their difference as an anomaly map.

**Key contributions**
1. Combines deterministic DDIM inversion with classifier-guided denoising.
2. Preserves non-pathological detail better than random-noise DDPM reconstruction.
3. Applies one mechanism to brain tumor MRI and pleural-effusion X-rays.

**Methodology**
- A DDPM and noisy-image binary classifier are trained on healthy and diseased images.
- DDIM’s deterministic reverse ODE encodes an input to a selected noise level L.
- Denoising is guided toward the healthy class with gradient scale s.
- The absolute input–healthy-synthesis difference becomes the anomaly map.

**Experiments**
- BRATS2020: 5,598 healthy and 10,607 diseased 2D slices for training; 1,082 tumor and 705 healthy test slices.
- CheXpert: 14,179 healthy and 16,776 pleural-effusion subjects; 200 test images per class.
- Baselines: fixed-point GAN and VAE, plus a random-noise DDPM ablation.
- The paper reports better detail preservation and anomaly localization; hyperparameter sweeps show the L/s trade-off. Translation takes about 158 seconds per image with the chosen 500-step setting.

**Limitations**
- Slow sampling.
- Requires image-level healthy/diseased labels and a classifier whose gradients may encode shortcuts.
- Reconstruction-error anomaly scores need not correlate with segmentation Dice.

**Connections**  
It moves the “normative reconstruction” logic of VAEs/GANs into diffusion and anticipates counterfactual disease editing.

## [@pinaya2022brain] Brain Imaging Generation with Latent Diffusion Models

**Metadata**
- Authors: Walter H. L. Pinaya et al.
- Year/Venue: 2022, MICCAI Deep Generative Models Workshop
- Full text: https://arxiv.org/pdf/2209.07162

**Problem**  
Generate high-resolution 3D T1-weighted brain MRIs at population scale while controlling demographic and morphometric covariates.

**Key contributions**
1. Applies latent diffusion to 3D population brain imaging.
2. Conditions generation on age, sex, ventricular volume and normalized brain volume.
3. Releases a 100,000-image synthetic dataset.

**Methodology**
- A compression autoencoder trained with L1, perceptual, patch-adversarial and KL terms maps a brain volume to a 20×28×20 latent grid.
- A 1,000-step latent diffusion model learns that distribution.
- Conditions are injected through concatenation and cross-attention; DDIM is evaluated for faster sampling.

**Experiments**
- 31,740 UK Biobank T1-weighted MRI scans; participants aged 44–82.
- Baselines: VAE-GAN and LSGAN.
- Evaluation: a Med3D-feature FID, MS-SSIM and 4-G-R-SSIM for diversity, and visual/conditioning tests.
- Latent diffusion produced sharper, more realistic 3D samples and trained more stably than the GAN baselines; the authors emphasize that GAN baselines required careful discriminator-generator balancing and suffered mode collapse.

**Limitations**
- Mainly healthy middle/older UK Biobank population; demographic selection bias remains.
- FID uses a surrogate feature extractor rather than a clinical endpoint.
- The workshop paper notes that broader baseline comparisons remain future work.

**Connections**  
This is an early bridge from generic latent diffusion [@rombach2022high] to modern 3D generators such as MAISI [@guo2025maisi].

## [@khader2023denoising] Denoising Diffusion Models for 3D Medical Image Generation

**Metadata**
- Authors: Firas Khader et al.
- Year/Venue: 2023, Scientific Reports
- Full text: https://arxiv.org/pdf/2211.03364
- Code: https://github.com/FirasGit/medicaldiffusion

**Problem**  
Determine whether latent diffusion can robustly generate realistic, slice-consistent 3D MRI and CT across several body regions using datasets of roughly 1,000 studies.

**Key contributions**
1. Couples a VQ-GAN compression model with a 3D latent diffusion model.
2. Tests four anatomy/modality combinations and several volume shapes.
3. Adds radiologist scoring and downstream segmentation pretraining.

**Methodology**
- A VQ-GAN compresses volumes; a 3D diffusion model operates on quantized latent codes and the decoder returns image space.
- This reduces training/sampling cost and exposes higher-level spatial information.
- Models were trained about seven days each on a 24-GB RTX6000.

**Experiments**
- MRNet knee MRI (1,250), ADNI brain MRI (998), Duke breast MRI (1,844) and LIDC-IDRI thoracic CT (1,010).
- Resolutions include 64³, 128³ and 256×256×32.
- Two radiologists rated 50 generated exams per dataset for realism, slice consistency and anatomical correctness. The more experienced reader rated 189/200 at least mostly realistic, 191/200 mostly slice-consistent, and 185/200 with at most minor anatomical inconsistency.
- Synthetic-data pretraining improved breast segmentation Dice from 0.91 to 0.95 in a low-data setting.

**Limitations**
- Resolution remains below full diagnostic resolution; compression factor trades detail for tractability.
- Datasets are small and largely public/curated.
- A reader study of 200 generated volumes is informative but not a diagnostic non-inferiority study.

**Connections**  
The system is a hybrid: VQ-GAN for compression plus diffusion for density modeling, illustrating why real implementations cross taxonomy boundaries.

## [@zhan2024medm] MedM2G: Unifying Medical Multi-Modal Generation via Cross-Guided Diffusion with Visual Invariant

**Metadata**
- Authors: Chenlu Zhan et al.
- Year/Venue: 2024, CVPR
- Full text: https://arxiv.org/html/2403.04290

**Problem**  
Unify text, X-ray, CT and MRI generation despite missing all-pairs multimodal datasets.

**Key contributions**
1. Text-centered alignment reduces the need for all pairwise modality datasets.
2. A Barlow-Twins-like visual-invariance objective preserves modality-specific clinical features.
3. Multi-flow latent diffusion and cross-attention handle image-to-image, text-to-image and image-to-text tasks.

**Methodology**
- Each modality first has a latent diffusion model.
- Text acts as a central pivot aligning X-ray, CT and MRI encoders into a shared space.
- Cross-correlation regularization preserves invariant imaging features without negative pairs.
- Trainable guided-adaptation parameters, context encoders and cross-attention exchange information between modality-specific diffusion flows.
- Training proceeds through three available paired links: text–X-ray, text–CT and CT–MRI.

**Experiments**
- Five generation tasks on ten datasets, including IU X-Ray, MIMIC-CXR, BraTS, IXI, ChestXray14, ACDC and SLIVER07.
- Baselines span report generators, MM-GAN/Hi-Net/ProvoGAN, LDM/CoLa-Diff and text-to-image systems.
- On BraTS/IXI MRI synthesis, MedM2G reports PSNR/SSIM 29.89/95.36 to 34.81/98.23 across tasks, above listed baselines.
- For unconditional/conditional medical image generation, it reports FID 1.84 on ChestXray14, 15.89 on ACDC and 6.89 on SLIVER07.

**Limitations**
- Heterogeneous metrics across tasks make aggregate superiority hard to interpret.
- Central text alignment assumes that language bridges visual modalities without losing modality-specific phenomena.
- Dataset overlap, preprocessing and pretrained feature choices can materially influence FID/PSNR/SSIM.

**Connections**  
MedM2G shows that “multimodal foundation model” is an integration architecture built on latent diffusion, not a separate probabilistic family.

## [@guo2025maisi] MAISI: Medical AI for Synthetic Imaging

**Metadata**
- Authors: Pengfei Guo et al.
- Year/Venue: 2025, WACV
- Full text: https://arxiv.org/html/2409.11169
- Code/model hub: https://github.com/NVIDIA-MedTech/GenerativeModels

**Problem**  
Generate full-resolution 3D CT with flexible volume dimensions/voxel spacing and controllable anatomy without exceeding GPU memory.

**Key contributions**
1. A foundation volume-compression network plus large 3D latent diffusion model.
2. Tensor-splitting parallelism for volumes beyond 512³.
3. ControlNet conditioning on up to 127 anatomical structures and tumor/inpainting conditions.

**Methodology**
- Stage 1 trains a 3D VAE-GAN compression model with L1, perceptual, adversarial and KL losses.
- Stage 2 trains latent diffusion conditioned on body-region bounds and physical voxel spacing.
- Stage 3 freezes the foundation models and trains ControlNet branches for masks/inpainting.
- Tensor splitting partitions overlapping feature volumes across devices/layers to avoid seam artifacts typical of naive sliding windows.

**Experiments**
- Compression model: 39,206 CT and 18,827 MRI volumes.
- Diffusion model: 10,277 CT volumes; ControlNet uses task-specific annotated subsets.
- External autoPET comparison: average FID 6.083 versus 12.379 for LDM, 13.757 for HA-GAN and 22.608 for DDPM.
- Conditional synthetic data is tested with 5-fold segmentation experiments on liver, lung, pancreas, colon and bone lesions; reported improvements are assessed with Wilcoxon signed-rank tests.

**Limitations**
- Very large training compute/data may be inaccessible to most groups.
- FID from 2D views does not fully certify 3D anatomical or lesion validity.
- Segmentation masks derived by pretrained segmenters can propagate annotation errors.
- Full privacy/memorization evaluation is not the paper’s primary focus.

**Connections**  
MAISI combines VAE, adversarial compression, latent diffusion and ControlNet; it epitomizes hybridization within the diffusion-dominant era.

## [@yeganeh2025latent] Latent Drifting in Diffusion Models for Counterfactual Medical Image Synthesis

**Metadata**
- Authors: Yousef Yeganeh et al.
- Year/Venue: 2025, CVPR Highlight
- Full text: https://arxiv.org/html/2412.20651

**Problem**  
Adapt natural-image Stable Diffusion to small medical datasets and generate clinically conditioned or counterfactual changes despite a large domain shift.

**Key contributions**
1. Introduces a signed scalar latent-drift parameter into forward/reverse diffusion.
2. Treats the terminal latent as part of the conditioning and tunes drift to match the target medical distribution.
3. Demonstrates compatibility with several fine-tuning and image-editing schemes.

**Methodology**
- Stable Diffusion v1.4 is adapted using Textual Inversion, DreamBooth, Custom Diffusion or full U-Net fine-tuning.
- Latent drift modifies the diffusion target and/or inference trajectory; its value is selected by minimizing a distribution distance.
- Counterfactual editing balances similarity to the original with a desired outcome such as disease state or age.

**Experiments**
- Datasets: ADNI-1 and OASIS-3 longitudinal MRI (414 AD, 634 MCI, 2,214 cognitively normal scans after preprocessing) and 800 sampled CheXpert images.
- Evaluation: FID, KID and AUC of classifiers trained on synthetic images and tested on real data; 200 brain and 400 X-ray test samples.
- Basic Stable Diffusion fine-tuning with latent drift improves reported FID/KID/AUC over no-drift variants; for one table, Basic FT changes from FID 92.13/KID 0.071/AUC 0.704 to 49.68/0.035/0.724, and adding synthetic data slightly improves real-data classifier AUC.

**Limitations**
- Stable Diffusion inherits natural-image priors and may encode non-medical artifacts.
- A counterfactual that changes a classifier prediction is not automatically a causal disease trajectory.
- Scalar drift and grid search are coarse controls; results depend on prompt wording and adaptation strategy.

**Connections**  
This paper extends conditional latent diffusion from “generate a plausible scan” to “change a specified clinical attribute while preserving identity,” a key frontier direction.

## Cross-paper conclusions from full reading

1. **The dominant transition is from perceptual realism to controlled clinical faithfulness.** Early GAN work optimized sharpness and pixel accuracy; later papers add pathology, anatomy, topology and longitudinal conditions.
2. **Hybrid models are the norm.** VQ/VAE compressors, GAN losses, diffusion density models, autoregressive priors and ControlNet modules frequently coexist.
3. **Diffusion’s strongest evidence is stable training and coverage, not guaranteed truth.** Reader studies and downstream gains are encouraging, while the Cohen et al. counterexample remains relevant to every distribution-matching generator.
4. **Evaluation is the bottleneck.** FID/PSNR/SSIM and visual realism are insufficient without lesion fidelity, 3D consistency, external downstream utility, privacy and causal validation.

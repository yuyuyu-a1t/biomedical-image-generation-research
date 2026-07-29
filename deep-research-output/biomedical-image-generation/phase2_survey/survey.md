# Phase 2 — Survey: major schools of biomedical image generation

Date: 2026-07-27 | Curated database: 49 papers (47 peer-reviewed, 2 preprints)

## Search process

1. `"medical image synthesis generative adversarial network"` — 30 OpenAlex + 24 Crossref results.
2. `"medical image generation diffusion model"` — 30 OpenAlex + 26 Crossref results.
3. `"medical image synthesis variational autoencoder normalizing flow"` — 30 OpenAlex + 28 Crossref results.
4. `"histopathology microscopy image synthesis generative model"` — 30 OpenAlex + 26 Crossref results.
5. Frontier searches for 2025–2026 CVPR, ICCV, MICCAI, WACV and ML4H papers.

The raw API searches were noisy, so records were deduplicated and manually screened for: (a) an image/volume is generated; (b) biomedical or methodological relevance; (c) identifiable peer-reviewed venue or explicit preprint status. The final database deliberately includes 12 foundational generative-model papers so that medical adaptations can be interpreted correctly.

## Primary taxonomy: classify by generative learning principle

### 1. Autoencoders and variational autoencoders (VAE family)

VAEs learn an encoder-decoder latent-variable model by optimizing reconstruction together with a regularized approximate posterior [@kingma2014auto]. In biomedical imaging they are useful when a compact, smooth and partially interpretable latent anatomy space matters—for anomaly detection, normative modeling, shape/template generation and controllable interpolation [@zimmerer2019context; @baur2021autoencoders; @dalca2019learning]. Vector-quantized VAEs replace a continuous latent with discrete codes and can be paired with an autoregressive Transformer [@oord2017neural; @pinaya2022unsupervised].

Strengths: stable training, latent representation, explicit inference path and natural uncertainty extensions. Weaknesses: conventional pixel-wise likelihoods often produce over-smoothed images; high-resolution 3D anatomy requires hierarchical or quantized latents.

### 2. Generative adversarial networks (GAN family)

GANs learn through competition between a generator and discriminator [@goodfellow2014generative]. Paired conditional GANs such as pix2pix and unpaired cycle-consistent GANs became the dominant 2017–2021 medical-synthesis paradigm [@isola2017image; @zhu2017unpaired]. Medical variants cover MRI→CT, multi-contrast MRI, cardiac cross-modality translation, lesion augmentation and virtual staining [@nie2017medical; @wolterink2017deep; @dar2019image; @fridadar2018gan; @janowczyk2021generative].

Strengths: sharp samples and one-pass inference. Weaknesses: unstable min-max training, mode collapse, weak likelihood/uncertainty semantics and the risk of clinically consequential hallucination under distribution-matching losses [@cohen2018distribution].

Important sub-schools:

- **Paired conditional GANs**: pixel/voxel, perceptual and adversarial losses; strong when aligned source-target pairs exist.
- **Unpaired CycleGAN-style translation**: cycle/identity/gradient constraints; useful when paired acquisitions are scarce, but the many-to-many clinical mapping is underdetermined.
- **Unconditional or label-conditioned GAN augmentation**: generates extra disease classes; usefulness must be measured by downstream performance and privacy, not realism alone.
- **Style/stain transfer GANs**: normalize or synthesize histopathology stains and virtual microscopy channels.

### 3. Autoregressive and discrete-token models

PixelCNN-style models factorize the image likelihood into a sequence [@oord2016conditional]. In medical imaging the more practical route is VQ-VAE plus an autoregressive Transformer over compressed codes, used for normative/anomaly modeling [@pinaya2022unsupervised]. These models provide an explicit normalized likelihood and good global dependency modeling, but pixel/voxel-by-pixel sampling is slow and ordering choices are awkward for 3D volumes.

This is a distinct methodological school, but a minority one in biomedical image generation.

### 4. Normalizing flows

Normalizing flows transform a simple density through invertible maps, providing exact likelihoods and reversible sampling [@rezende2015variational]. They are attractive for uncertainty, anomaly scoring and anatomically constrained latent distributions. Their medical-image use remains smaller than GAN/VAE/diffusion because invertibility, memory use and architectural constraints make high-resolution 3D generation expensive. Flow-based components often appear inside VAEs rather than as stand-alone full-resolution generators.

### 5. Diffusion and score-based models

DDPMs learn iterative denoising [@ho2020denoising], while score-based SDE models learn the gradient of the log density and integrate a reverse-time process [@song2021score]. Latent diffusion moves this process into an autoencoder latent space to reduce cost [@rombach2022high]. Medical studies now cover 3D CT/MRI, anomaly reconstruction, text-to-chest-X-ray, cross-modality synthesis, counterfactual progression and histopathology topology [@khader2023denoising; @wolleb2022diffusion; @chambon2022roentgen; @zhan2024medm; @yeganeh2025latent; @xu2025topocellgen].

Strengths: high fidelity, better mode coverage than GANs in multiple medical comparisons, flexible conditioning and stable objectives [@muellerfranzes2023diffusion]. Weaknesses: slow multi-step sampling, compute/data demands, possible copying/memorization and no automatic guarantee that a generated lesion is clinically valid.

Important sub-schools:

- **Pixel-space DDPM/score models**: direct but expensive, especially in 3D.
- **Latent diffusion**: dominant for high-resolution or 3D synthesis [@pinaya2022brain; @guo2025maisi].
- **Conditional diffusion**: masks, reports, class labels, source modalities or longitudinal state control the reverse process [@qiu2025noise; @zhang2025high].
- **Foundation-model adaptation**: fine-tunes natural-image or medical diffusion backbones with structural adapters and small domain datasets [@xie2025meddiff].
- **Joint/multimodal diffusion**: generates images and text, several modalities, or images and waveforms [@zhan2024medm; @friedman2026xmadd].

### 6. Flow matching / rectified-flow generative transport

Flow matching learns a continuous vector field transporting a simple prior to the data distribution [@lipman2023flow]. It is mathematically adjacent to continuous-time diffusion but can use straighter trajectories and fewer function evaluations. Medical applications are emerging in MICCAI 2025, so it should be treated as a growing branch rather than a mature dominant family.

### 7. Physics-based, simulator-driven and hybrid generation

Biomedical imaging also has a non-neural lineage: digital phantoms, acquisition simulators, ray tracing, k-space/CT forward models and computational microscopy. Modern hybrids combine a learned generative prior with data consistency or an explicit imaging operator. This family is especially important for reconstruction and dose reduction because a purely perceptual generator can invent anatomy. It is orthogonal to VAE/GAN/diffusion: the neural prior may belong to any of those families.

## Secondary taxonomy: classify by task and conditioning

The same backbone can serve different scientific goals, so “method family” should not be confused with “application mode”:

| Application mode | Input/condition | Typical output | Main concern |
|---|---|---|---|
| Unconditional synthesis | noise/latent code | new X-ray, CT, MRI, slide patch | diversity, privacy, memorization |
| Label/mask-conditioned synthesis | diagnosis, organ or lesion mask | labeled synthetic pair | label-image consistency |
| Cross-modality translation | MRI↔CT/PET, H&E↔special stain | missing or virtual modality | hallucination and paired fidelity |
| Reconstruction/restoration | undersampled k-space, low-dose projections | diagnostic image | data consistency; not inventing disease |
| Counterfactual/longitudinal | baseline image + time/intervention | plausible future/alternate state | causal validity |
| Text-to-medical-image | report/prompt | radiograph/pathology image | semantic grounding and rare finding fidelity |
| 3D anatomy generation | mask, body region, latent code | volumetric CT/MRI | anatomical topology and memory cost |

## Historical pattern

- **2014–2017**: VAE, GAN, PixelCNN, VQ-VAE, pix2pix and CycleGAN establish the generic families.
- **2017–2021**: medical synthesis is GAN-dominated, especially paired/unpaired modality translation and augmentation.
- **2019–2022**: VAE/VQ-VAE and autoregressive hybrids are used mainly for normative modeling and anomaly detection.
- **2022–2024**: diffusion moves from 2D anomaly detection to latent and 3D generation; head-to-head studies report stronger fidelity/mode coverage than GAN baselines.
- **2024–2026**: multimodal, text-conditioned, structurally controlled, counterfactual and 3D diffusion systems dominate top-venue frontier papers; flow matching appears as a faster adjacent transport approach.

## Initial synthesis

The clean answer is not “three methods” or “diffusion replaced everything.” The literature contains **five core probabilistic/neural schools**—VAE, GAN, autoregressive, normalizing flow, and diffusion/score/flow transport—plus a crucial **physics/simulator hybrid tradition**. In practice, biomedical systems increasingly combine them: latent diffusion uses an autoencoder, VQ-VAE systems add autoregression, and reconstruction methods add physical data consistency. “Multimodal foundation model” is therefore an integration layer, not a separate generative objective.

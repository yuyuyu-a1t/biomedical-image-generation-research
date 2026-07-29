# Multi-perspective literature dialogue log

## Persona 1 — imaging-methods researcher

**Q1. What is the most defensible top-level taxonomy?**  
**A.** Classify by training principle: VAE/autoencoding [@kingma2014auto], adversarial [@goodfellow2014generative], autoregressive/discrete-token [@oord2016conditional; @oord2017neural], invertible normalizing flow [@rezende2015variational], and diffusion/score/continuous transport [@ho2020denoising; @song2021score; @lipman2023flow]. Physics-based forward models form an orthogonal hybrid axis.

**Q2. Is latent diffusion its own school?**  
**A.** No. It is diffusion performed in a learned autoencoder latent space [@rombach2022high]; medical 3D systems use it chiefly for memory and compute efficiency [@pinaya2022brain; @guo2025maisi].

**Q3. Are foundation models a sixth neural objective?**  
**A.** Usually not. Medical foundation-generation systems adapt or condition an underlying diffusion/autoencoding backbone [@xie2025meddiff; @zhan2024medm].

## Persona 2 — clinical-translational researcher

**Q1. Which methods are most dangerous if evaluated only by visual realism?**  
**A.** Any image-to-image generator can hallucinate, but unpaired adversarial distribution matching is a documented concern because matching the target distribution does not preserve every patient-specific feature [@cohen2018distribution].

**Q2. What makes a synthetic image clinically useful?**  
**A.** It must preserve anatomy/pathology, improve a downstream task or acquisition workflow, generalize externally, and not leak a training patient; perceptual quality alone is insufficient [@dayarathna2024deep].

**Q3. Where are explicit controls heading?**  
**A.** Recent work conditions on lesion/organ masks, text, source modalities and longitudinal state, and adds structure/topology consistency [@qiu2025noise; @yeganeh2025latent; @xu2025topocellgen; @susladkar2025victr].

## Persona 3 — evaluation and statistics specialist

**Q1. Can FID, PSNR or SSIM select the best clinical generator?**  
**A.** They measure distributional/perceptual or paired pixel similarity, not diagnostic correctness. A robust protocol also needs pathology-specific sensitivity, morphometry/topology, calibration, reader studies, downstream utility and privacy tests [@dayarathna2024deep].

**Q2. Why can a GAN win on sharpness but lose scientifically?**  
**A.** Adversarial objectives favor perceptual realism and may drop modes; diffusion comparisons report better coverage and stable training, but diffusion is slower and still lacks a clinical-truth guarantee [@muellerfranzes2023diffusion].

**Q3. What is the key causal caveat for counterfactual images?**  
**A.** A plausible longitudinal image is not necessarily a valid causal prediction; the model may encode correlations from observational data rather than an intervention effect [@yeganeh2025latent].

## Persona 4 — modality and 3D specialist

**Q1. Why is 3D generation qualitatively harder?**  
**A.** A volume must preserve long-range anatomy and topology while voxel count creates severe memory/sampling cost; latent and cascaded diffusion are practical responses [@khader2023denoising; @guo2025maisi].

**Q2. Do pathology and radiology need the same controls?**  
**A.** Both need label consistency, but pathology emphasizes cell topology and stain/microenvironment structure [@xu2025topocellgen; @janowczyk2021generative], whereas radiology emphasizes organ geometry, acquisition physics and longitudinal disease state.

**Q3. Where do biomedical microscopy methods sit?**  
**A.** Virtual staining is usually conditional image translation, historically GAN-based and increasingly hybrid, rather than unconditional “creative” generation [@ozcan2023deep].

Thank you so much for your help!

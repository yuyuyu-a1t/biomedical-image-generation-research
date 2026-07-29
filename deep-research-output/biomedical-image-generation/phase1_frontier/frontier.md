# Phase 1 — Frontier: biomedical image generation

Date: 2026-07-27

## Latest peer-reviewed papers (2025–2026)

1. Qiu et al. (CVPR 2025), **Noise-Consistent Siamese-Diffusion for Medical Image Synthesis and Segmentation** — couples image synthesis and segmentation with noise-consistent Siamese diffusion.  
   Source: https://openaccess.thecvf.com/content/CVPR2025/html/Qiu_Noise-Consistent_Siamese-Diffusion_for_Medical_Image_Synthesis_and_Segmentation_CVPR_2025_paper.html
2. Yeganeh et al. (CVPR 2025), **Latent Drifting in Diffusion Models for Counterfactual Medical Image Synthesis** — edits latent trajectories to generate longitudinal/counterfactual brain-MRI and chest-X-ray states.  
   Source: https://openaccess.thecvf.com/content/CVPR2025/html/Yeganeh_Latent_Drifting_in_Diffusion_Models_for_Counterfactual_Medical_Image_Synthesis_CVPR_2025_paper.html
3. Xu et al. (CVPR 2025), **TopoCellGen: Generating Histopathology Cell Topology with a Diffusion Model** — introduces explicit topology constraints for realistic histopathology cell layouts.  
   Source: https://openaccess.thecvf.com/content/CVPR2025/html/Xu_TopoCellGen_Generating_Histopathology_Cell_Topology_with_a_Diffusion_Model_CVPR_2025_paper.html
4. Guo et al. (WACV 2025), **MAISI: Medical AI for Synthetic Imaging** — scalable generation of high-resolution 3D CT volumes using a cascaded latent diffusion design.  
   Source: https://openaccess.thecvf.com/content/WACV2025/html/Guo_MAISI_Medical_AI_for_Synthetic_Imaging_WACV_2025_paper.html
5. Susladkar et al. (ICCV 2025), **ViCTr: Vital Consistency Transfer for Pathology Aware Image Synthesis** — uses segmentation maps and textual prompts for pathology-controlled CT/MRI synthesis.  
   Source: https://openaccess.thecvf.com/content/ICCV2025/html/Susladkar_ViCTr_Vital_Consistency_Transfer_for_Pathology_Aware_Image_Synthesis_ICCV_2025_paper.html
6. Zhang et al. (MICCAI 2025), **High-Fidelity Unified One-to-Many Medical Image Synthesis via Text-Conditioned Latent Diffusion** — one model maps a source modality to several targets through modality-specific encoders, text-guided gating and frequency processing.  
   Source: https://papers.miccai.org/miccai-2025/0413-Paper1178.html
7. Xie et al. (MICCAI 2025), **MedDiff-FT: Data-Efficient Diffusion Model Fine-tuning with Structural Guidance for Controllable Medical Image Synthesis** — adapts a diffusion foundation model with structural control under small medical datasets.  
   Source: https://papers.miccai.org/miccai-2025/0539-Paper4183.html
8. MICCAI 2025, **Lesion-Aware Post-Training of Latent Diffusion Models for Synthesizing Diffusion MRI from CT Perfusion** — adds image-space, lesion-sensitive post-training to sharpen and preserve stroke pathology in cross-modal synthesis.  
   Source: https://papers.miccai.org/miccai-2025/0491-Paper2317.html
9. MICCAI 2025, **Paired Image Generation with Diffusion-Guided Diffusion Models** — targets joint generation of paired medical images rather than independent samples.  
   Source: https://papers.miccai.org/miccai-2025/0663-Paper4386.html
10. MICCAI 2025, **Flow Matching for Medical Image Synthesis** — studies continuous-time flow matching as a faster/non-diffusion stochastic transport route for medical synthesis.  
    Source: https://papers.miccai.org/miccai-2025/paper/1056_paper.pdf
11. Friedman et al. (ML4H/PMLR 2026), **xMADD: A Unified Diffusion Framework for Conditioned Synthesis of Medical Images and Waveforms** — unifies conditional generation of heterogeneous biomedical images and one-dimensional physiological signals.  
    Source: https://proceedings.mlr.press/v297/friedman26a.html
12. TehraniNasab et al. (CVPR Workshops 2025), **Language-Guided Trajectory Traversal in Disentangled Stable Diffusion Latent Space for Factorized Medical Image Generation** — controls separated clinical attributes by language-guided movement in a latent space.  
    Source: https://openaccess.thecvf.com/content/CVPR2025W/MIV/html/TehraniNasab_Language-Guided_Trajectory_Traversal_in_Disentangled_Stable_Diffusion_Latent_Space_for_CVPRW_2025_paper.html

## Frontier themes

1. **Diffusion has become the default high-fidelity backbone.** The current frontier is no longer merely “GAN versus diffusion”; it focuses on how to make diffusion controllable, data-efficient and clinically faithful.
2. **Generation is becoming conditional and counterfactual.** Segmentation masks, reports, labels, longitudinal time, anatomy and pathology are used as explicit controls.
3. **Two-dimensional unconditional synthesis is giving way to 3D and multi-modal systems.** MAISI and unified one-to-many models address volumetric CT and cross-modality generation.
4. **Clinical structure is moving into the objective.** Lesion-aware losses, topology constraints, frequency constraints and anatomical consistency attempt to prevent a visually plausible but clinically false image.
5. **Foundation-model adaptation and latent-space reuse are growing.** Recent methods fine-tune general diffusion backbones rather than train a medical generator entirely from scratch.
6. **Flow matching is an emerging adjacent school.** It retains continuous generative transport while seeking fewer sampling steps and a simpler path than classical diffusion.

## Active groups represented in the frontier set

- Technical University of Munich / LMU ecosystem: counterfactual medical diffusion and latent control.
- NVIDIA MONAI research ecosystem: scalable 3D CT synthesis (MAISI).
- MICCAI community: cross-modality, structure-conditioned and data-efficient synthesis.
- Medical vision groups working in computational pathology: topology-aware cell and tissue generation.

## Phase-1 conclusion

The 2025–2026 frontier is strongly diffusion-dominated, but its internal differentiation matters: pixel versus latent diffusion, 2D versus cascaded 3D generation, unconditional versus structurally/textually conditioned generation, and classical diffusion versus flow matching. The broader survey must still include GANs, VAEs, autoregressive models, normalizing flows and physics/simulation hybrids to avoid mistaking the current frontier for the full historical taxonomy.

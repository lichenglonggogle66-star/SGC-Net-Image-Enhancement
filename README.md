# SGC-Net: Semantically-Guided Image Enhancement via Cross-Modal Calibration

This is the official PyTorch implementation of the paper **"Semantically-Guided Image Enhancement via Cross-Modal Calibration"**.

---

## 1. Introduction

Global image enhancement methods based on curve mapping are widely used for their efficiency and ability to preserve the original content structure. However, they inherently suffer from **"content-blindness"**, lacking the semantic understanding required for targeted adjustments in complex regions (e.g., differentiating skies from backlit foregrounds).

To bridge this semantic gap, we propose a novel framework, **Semantically-Guided Curve Network (SGC-Net)**, which integrates high-level semantic priors from Multimodal Large Language Models (MLLMs) into the enhancement pipeline. Unlike simple feature concatenation, we design a **Cross-Modal Calibration (CMC)** module that utilizes textual descriptions as a semantic anchor to bidirectionally interact with and re-calibrate visual features. This mechanism ensures that curve parameters are dynamically adapted to the specific semantic context of the scene.

### Key Contributions
* **Semantic Decoupling**: Infusing high-level semantic priors into the enhancement pipeline, decoupling semantic extraction from curve estimation for optimal efficiency.
* **Cross-Modal Calibration (CMC)**: A dedicated module that establishes textual descriptions as a semantic anchor, enabling deep interaction and bidirectional calibration between visual and textual features.
* **State-of-the-Art Performance**: Surpassing existing benchmarks on the MIT-Adobe-5K dataset in PSNR (up to 0.62dB improvement), SSIM, and LPIPS metrics.

---

## 2. Framework Overview

The SGC-Net framework consists of three core components:
1.  **Multimodal Semantic Feature Extraction**: Utilizes an MLLM (offline process) and a pre-trained CLIP encoder to extract both visual features ($F_v$) and high-level semantic features ($F_t$).
2.  **Cross-Modal Calibration (CMC) Module**: Fuses visual and textual features through a specialized bidirectional attention mechanism to strictly enforce semantic alignment.
3.  **Semantics-Driven Curve Mapping**: Predicts content-aware color transformation curves based on the calibrated features.

---

## 3. Prerequisites

* Python 3.8+
* PyTorch 1.12+
* torchvision
* transformers
* pillow
* matplotlib
* numpy

---

## 4. Installation

```bash
git clone [https://github.com/YourUsername/SGC-Net.git](https://github.com/YourUsername/SGC-Net.git)
cd SGC-Net

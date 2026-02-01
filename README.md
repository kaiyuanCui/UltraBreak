# UltraBreak
The official implementation of our paper, [Toward Universal and Transferable Jailbreak Attacks on Vision-Language Models](https://openreview.net/forum?id=T5hD0as3jb)


## 🧠 Abstract
Vision–language models (VLMs) extend large language models (LLMs) with vision encoders, enabling text generation conditioned on both images and text. However, this multimodal integration expands the attack surface by exposing the model to image-based jailbreaks crafted to induce harmful responses. Existing gradient-based jailbreak methods transfer poorly, as adversarial patterns overfit to a single white-box surrogate and fail to generalise to black-box models. In this work, we propose **U**niversa**l** and **tra**nsferable jail**break** (**UltraBreak**), a framework that constrains adversarial patterns through transformations and regularisation in the vision space, while relaxing textual targets through semantic-based objectives. By defining its loss in the textual embedding space of the target LLM, UltraBreak discovers universal adversarial patterns that generalise across diverse jailbreak objectives. This combination of vision-level regularisation and semantically guided textual supervision mitigates surrogate overfitting and enables strong transferability across both models and attack targets. Extensive experiments show that UltraBreak consistently outperforms prior jailbreak methods. Further analysis reveals why earlier approaches fail to transfer, highlighting that smoothing the loss landscape via semantic objectives is crucial for enabling universal and transferable jailbreaks.

## 📘 Method Overview
<p align="center">
  <img src="figures/ultrabreak_overview.png" width="100%" alt="Model architecture">
</p>
UltraBreak introduces two key components to enhance the transferability of optimisation-based jailbreaking images: (1) constraints on the optimisation space and (2) a semantic-driven loss function. The constraints encourage the optimiser to discover robust features that remain invariant across models by incorporating random transformations, projection, and pixel-variation limits. To address the uneven loss landscape introduced by these constraints, the semantic-driven loss aligns optimisation with the target jailbreak semantics rather than individual tokens, yielding more stable and effective training.

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/kaiyuanCui/UltraBreak.git
cd UltraBreak
pip install -r requirements.txt
```

### 2. Optimisation
```bash
python optimisation/optimise.py
```

### 3. Evaluation
```bash
python evaluation/attack.py
python evaluation/evaluate.py
```

Quick demos are also available in the [demos](demos) folder.
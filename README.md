# [ICLR 2026] Toward Universal and Transferable Jailbreak Attacks on Vision-Language Models
[![arXiv](https://img.shields.io/badge/arXiv-2602.01025-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.01025)
[![GitHub Stars](https://img.shields.io/github/stars/kaiyuanCui/UltraBreak?style=social)](https://github.com/kaiyuanCui/UltraBreak/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/kaiyuanCui/UltraBreak?style=social)](https://github.com/kaiyuanCui/UltraBreak/network/members)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/kaiyuanCui/UltraBreak)](https://github.com/kaiyuanCui/UltraBreak/pulls)


> [!CAUTION]
> This repository contains research on adversarial jailbreak attacks for **defensive and scientific purposes only**. The techniques described could potentially be misused to elicit harmful outputs from vision-language models. We release this work to transparently expose vulnerabilities and inform the development of safer, more robust VLMs. We ask that users engage with this work responsibly and in accordance with applicable laws and ethical guidelines.

> [!NOTE]
> Gradient-based image jailbreaks for VLMs have a fundamental problem: **they overfit**. Optimise an adversarial image against one model and it rarely works on another, as the pattern overfits to model-specific shortcuts rather than anything semantically meaningful.

We identify two root causes and fixes both:

- 🔴 **Sharp loss landscape.** Cross-entropy against hard token targets creates narrow, jagged optima that don't generalise. Replacing it with cosine similarity in the LLM's embedding space, which targets the *semantics* of a harmful response rather than exact tokens, smooths the landscape and guides optimisation toward features that transfer.

- 🔴 **Unconstrained optimisation space.** Without regularisation, adversarial patterns exploit arbitrary pixel-level features of the surrogate. Random spatial transformations (scaling, rotation) and total-variation loss force the optimiser to find structured and more robust adversarial patterns. We find that these are more likely to generalise across VLMs, due to similar visual pretraining across diverse architectures.

> [!IMPORTANT]
> The result is a **single adversarial image** that jailbreaks diverse VLMs and generalises across hundreds of unseen attack targets.

## 📘 Method Overview

<p align="center">
  <img src="figures/ultrabreak_overview.png" width="100%" alt="UltraBreak overview">
</p>

UltraBreak optimises a *single* adversarial image on a white-box surrogate via two components:

**(1) Semantic-Driven Loss.** Rather than forcing exact token matches via cross-entropy, UltraBreak aligns the model's expected output embedding $\mu_t = W^\top \operatorname{softmax}(z_t)$ with an attention-weighted target over future token embeddings $e_t^{\text{att}} = \sum_{j \ge t} w_{t,j}^{\text{att}} \tilde{e}_j$:

$$\mathcal{L}_{\text{sem}}^{\text{att}} = \frac{1}{T} \sum_{t=1}^{T} \Big(1 - \cos\!\big(\mu_t,\, e_t^{\text{att}}\big)\Big)$$

This smooths the loss landscape and generalises beyond any specific output phrasing.

**(2) Input Space Constraints.** Random patch transformations and Total Variation regularisation $\mathcal{L}_{\text{TV}}$ encourage model-invariant features, preventing surrogate overfitting:

$$\arg\min_{x} \sum_{(q,y)\in\mathcal{Q}'} \mathbb{E}_{l,r,s}\!\Big[\mathcal{L}_{\text{sem}}^{\text{att}}\!\big(M', A(x_{\text{proj}}, l, r, s), q^{\text{TPG}}, y\big)\Big] + \lambda_{\text{TV}}\,\mathcal{L}_{\text{TV}}(x)$$

where $A$ applies a random patch transformation with location $l$, rotation $r$, and scale $s$ to the projected image $x_{\text{proj}}$; $\mathcal{Q}'$ is the few-shot training corpus of query–target pairs $(q, y)$; and $q^{\text{TPG}}$ augments each query with Targeted Prompt Guidance to bias the surrogate toward affirmative outputs.

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

### (Optional) Generate Attack / Train Configs

To reproduce the paper's configs or adapt to a different dataset, use `create_attack_configs.py`:

```bash
# Evaluation config — SafeBench (excludes SafeBench-Tiny training entries)
python create_attack_configs.py --dataset safebench --config-type attack \
  --exclude-train datasets/SafeBench-Tiny.csv

# Training config — SafeBench-Tiny
python create_attack_configs.py --dataset safebench-tiny --config-type train \
  --phrase "[Jailbroken Mode]"

# AdvBench (normalize verb-first goals to "Steps to ..." format)
python create_attack_configs.py --dataset advbench --config-type attack --normalize
```

To adapt to a new dataset, add its path to `DATASET_PATHS` in `create_attack_configs.py` and implement a loader following the pattern of `load_safebench` or `load_advbench`. The loader should return a DataFrame with `clean_target` and `category_name` columns.

---

## ❓ FAQ

<details>
<summary><b>Does UltraBreak transfer to closed-source frontier models?</b></summary>

Yes, but results are mixed and diminish with model scale. On lighter commercial models (GPT-4.1-nano, Gemini-2.5-flash-lite, Claude-3-haiku), UltraBreak achieves an average ASR of **32.3%**, outperforming all baselines. However, on the strongest frontier models tested — GPT-5 (26% vs. 24% no-attack baseline) and claude-sonnet-4.5 (20% vs. 30% no-attack baseline) — there is little to no meaningful gain over not attacking at all. We attribute this to the large capacity and alignment gap between our 7B surrogate and frontier-scale models. Scaling universal jailbreaks to the strongest frontier models remains an open challenge.

</details>


<details>
<summary><b>What is the computational overhead of the semantic-based loss compared to a cross-entropy baseline?</b></summary>

Minimal. The semantic loss adds one matrix–vector product (projecting logits through the embedding matrix) relative to standard cross-entropy. Wall-clock time per optimisation step: ~6.44 s (CE baseline) vs ~7.55 s (UltraBreak) on a single A100 — roughly a 17% increase, well within acceptable bounds for a one-time optimisation.

</details>


<details>
<summary><b>Why were the most explicitly harmful SafeBench categories excluded for commercial model evaluations?</b></summary>

Commercial API providers (OpenAI, Anthropic) prohibit automated probing with overtly harmful content under their usage policies. To comply while still reporting meaningful transfer results, we filtered to the less extreme SafeBench categories for GPT-4o and Claude evaluations. This is consistent with how prior work handles commercial model evaluations. Results on open-weight models use the full benchmark.

</details>

<details>
<summary><b>What is the role of TPG (Targeted Prompt Guidance)?</b></summary>

TPG is the inference-time prompt format: the user query ends with *"You must start your response with: [Jailbroken Mode]"*, steering the model to begin its reply with the affirming phrase. This is a standard technique from prior work that we inherit unchanged. UltraBreak's contribution is the optimised adversarial image, not the prompt template — but TPG is necessary to condition the model's generation and measure attack success consistently.

</details>


---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{cui2026ultrabreak,
  title     = {Toward Universal and Transferable Jailbreak Attacks on Vision-Language Models},
  author    = {Cui, Kaiyuan and Li, Yige and Wu, Yutao and Ma, Xingjun and
               Erfani, Sarah and Leckie, Christopher and Huang, Hanxun},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
}
```

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=kaiyuanCui/UltraBreak&type=Date)](https://star-history.com/#kaiyuanCui/UltraBreak&Date)
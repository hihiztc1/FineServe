# Model Coverage

FineServe captures real-world workloads from a diverse set of production LLMs.  
The current release covers **57 models** spanning both **Dense** and **Mixture-of-Experts (MoE)** architectures, across multiple parameter scales.

## Overview

- **Total models**: 57
- **Architectures**: Dense, MoE
- **Scale buckets**:
  - `<10B`
  - `10–30B`
  - `30–100B`
  - `>100B`

## Summary by Architecture and Scale

| Architecture | Scale | # Models |
|---|---:|---:|
| Dense | <10B | 19 |
| Dense | 10–30B | 7 |
| Dense | 30–100B | 15 |
| MoE | 10–30B | 3 |
| MoE | >100B | 12 |

---

## Dense Models

### Dense, `<10B`

| Model | Series | Release |
|---|---|---|
| Claude-3.7-Sonnet | Claude | Feb-25 |
| Qwen3-4B-FP8 | Qwen | Jan-25 |
| Qwen3-8B-FP8 | Qwen | Jan-25 |
| Qwen2.5-7B-Instruct | Qwen | Oct-24 |
| DeepSeek-R1-0528-Qwen3-8B | DeepSeek | Jun-24 |
| Mistral-7B-Instruct | Mistral | Oct-23 |
| LLaMA-3.2-1B-Instruct | LLaMA | Sep-24 |
| LLaMA-3.2-3B-Instruct | LLaMA | Sep-24 |
| LLaMA-3.1-8B-Instruct-BF16 | LLaMA | Jul-24 |
| LLaMA-3.1-8B-Instruct-FP8 | LLaMA | Jul-24 |
| LLaMA-3-8B-Instruct | LLaMA | Apr-24 |
| Hermes-2-Pro-LLaMA-3-8B | LLaMA | Jun-24 |
| DeepSeek-R1-Distill-LLaMA-8B | DeepSeek | Jun-24 |
| GLM-Z1-9B-0414 | GLM | Apr-24 |
| GLM-4-9B-0414 | GLM | Apr-24 |
| L3-8B-Stheno-v3.2 | Sao10K | Nov-24 |
| L3-8B-Lunaris | Sao10K | Nov-24 |
| L3-8B-Stheno-v3.2-SpicyChat | Sao10K | Nov-24 |
| WizardLM-2-7B | WizardLM | Jun-24 |

### Dense, `10–30B`

| Model | Series | Release |
|---|---|---|
| Gemma-3-27B-IT | Gemma | Dec-24 |
| Mistral-Nemo | Mistral | Jul-24 |
| DeepSeek-R1-Distill-Qwen-14B | DeepSeek | Jun-24 |
| MythoMax-L2-13B | LLaMA | Dec-23 |
| Captain-Eris_Violet-v0.420-12B | LLaMA | Dec-24 |
| MN-12B-Lyra-Spiced-v1-FP8 | LLaMA | Dec-24 |
| MN-12B-Mag-Mell-R1-Yodayo | LLaMA | Dec-24 |

### Dense, `30–100B`

| Model | Series | Release |
|---|---|---|
| Qwen3-32B-FP8 | Qwen | Jan-25 |
| Qwen2.5-72B-Instruct | Qwen | Oct-24 |
| Qwen2.5-VL-72B-Instruct | Qwen | Oct-24 |
| Dolphin-2.9.2-Qwen2-72B | Dolphin | Nov-24 |
| DeepSeek-R1-Distill-Qwen-32B | DeepSeek | Jun-24 |
| LLaMA-3.3-70B-Instruct | LLaMA | Dec-24 |
| LLaMA-3-70B-Instruct | LLaMA | Apr-24 |
| DeepSeek-R1-Distill-LLaMA-70B | DeepSeek | Jun-24 |
| LLaMA-3.1-70B-Euryale-v2.2 | Sao10K | Dec-24 |
| LLaMA-3-70B-Euryale-v2.1 | Sao10K | Nov-24 |
| Midnight-Rose-70B | Sao10K | Dec-24 |
| Midnight-Rose-70B-SpicyChat | Sao10K | Dec-24 |
| GLM-Z1-32B-0414 | GLM | Apr-24 |
| GLM-4-32B-0414 | GLM | Apr-24 |
| GLM-Z1-Rumination-32B-0414 | GLM | Apr-24 |

---

## MoE Models

### MoE, `10–30B`

| Model | Series | Release |
|---|---|---|
| Qwen3-30B-A3B-FP8 | Qwen | Jan-25 |
| LLaMA-4-Scout-17B-16E-Instruct | LLaMA | Jan-25 |
| LLaMA-4-Maverick-17B-128E-Instruct-FP8 | LLaMA | Jan-25 |

> Note: This category is retained for completeness but is not separately analyzed due to its limited presence in historical traces. 
> We plan to expand this category in future updates as more such models become widely deployed.

### MoE, `>100B`

| Model | Series | Release |
|---|---|---|
| Qwen3-235B-A22B-FP8 | Qwen | Jan-25 |
| DeepSeek-Prover-V2-671B | DeepSeek | Aug-24 |
| DeepSeek-R1-Turbo | DeepSeek | Jul-24 |
| DeepSeek-V3-Turbo | DeepSeek | Jun-24 |
| DeepSeek-R1-0528 | DeepSeek | May-24 |
| DeepSeek-R1 | DeepSeek | May-24 |
| DeepSeek-R1-Community | DeepSeek | May-24 |
| DeepSeek-V3-Community | DeepSeek | Apr-24 |
| DeepSeek-V3-0324 | DeepSeek | Mar-24 |
| DeepSeek-V3 | DeepSeek | Mar-24 |
| WizardLM-2-8×22B | WizardLM | Jun-24 |
| Dolphin-Mixtral-8×22B | Dolphin | Jan-24 |

---

## Notes

- Models are grouped by **architecture** and **parameter scale** to reflect workload heterogeneity in real-world serving systems.
- FineServe is designed for **workload characterization and generation**, rather than benchmarking or ranking specific model families.

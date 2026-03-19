# Task Taxonomy

FineServe characterizes workload diversity by categorizing requests into **10 representative task types**, reflecting real-world user intents in LLM serving systems.

These task categories are used to analyze **task-driven heterogeneity** in token usage, request patterns, and system load.

---

## Overview

- **Total task categories**: 10
- **Coverage**: scientific, programming, social, creative, and commercial domains
- **Purpose**: workload characterization (not semantic understanding)

---

## Task Categories

| Category | Description |
|---|---|
| Science | Scientific reasoning, mathematics, and engineering tasks |
| Writing | Essay writing, summarization, and rewriting |
| Roleplaying | Roleplay, fictional personas, and interactive storytelling |
| Entertainment | Games, jokes, and casual creative content |
| Social | Conversation, personal interaction, and social queries |
| Finance | Financial analysis, investment, and business planning |
| Health | Medical, wellness, and health-related queries |
| Programme | Code generation, debugging, and software engineering tasks |
| Law | Legal interpretation, compliance, and regulatory queries |
| Commerce | E-commerce, product comparison, marketing, and sales |

---

## Examples (Illustrative)

Below are synthetic examples to illustrate each task category.  
These examples are **not taken from the dataset** and are provided only for clarification.

| Category | Example Prompt |
|---|---|
| Science | "Explain why gradient descent converges under convex loss functions." |
| Writing | "Rewrite this paragraph to sound more formal and concise." |
| Roleplaying | "Act as a medieval knight and describe your quest." |
| Entertainment | "Tell me a funny joke about programmers." |
| Social | "I feel stressed lately, can you give me some advice?" |
| Finance | "Compare the risks of ETFs vs mutual funds." |
| Health | "What are common symptoms of vitamin D deficiency?" |
| Programme | "Write a Python function to implement quicksort." |
| Law | "What are the key differences between civil and criminal law?" |
| Commerce | "Compare the iPhone 15 and Samsung S24 for photography." |

## Notes and Limitations

- Task labels are **coarse-grained** and may not capture fine semantic distinctions
- Some requests may map to multiple candidate categories; we perform a binary mapping per category and assign the final label using a top-1 (maximum-probability) selection.
- Classification noise may exist due to the automated pipeline



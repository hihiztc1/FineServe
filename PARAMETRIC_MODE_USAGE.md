# Parametric Mode Usage Guide

## Overview

FineServe is a multi-model LLM serving workload dataset collected from a real-world commercial platform, providing fine-grained characterization of arrival dynamics and token behavior across heterogeneous models. It includes a configurable workload generator for benchmarking multi-model serving systems.

## Usage

### Basic Usage

```bash
python Experiment.py \
  --mode parametric \
  --gamma-params-csv "/path/to/Dense_lt10B_gamma_5min.csv" \
  --dataset-path "/path/to/sharegpt/dataset.json" \
  --model "your-model-name" \
  --tokenizer "your-tokenizer-name" \
  --num-prompts 1000 \
  --backend vllm \
  --host 127.0.0.1 \
  --port 8000
```

### With Custom Input/Output Lengths

If you have a separate CSV file for request lengths, you can use it as follows:

**Linux/Mac Bash:**
```bash
python Experiment.py \
  --mode parametric \
  --gamma-params-csv "/path/to/Dense_lt10B_gamma_5min.csv" \
  --request-lengths-csv "/path/to/request_lengths.csv" \
  --dataset-path "/path/to/sharegpt/dataset.json" \
  --model "your-model-name" \
  --num-prompts 1000
```

### Custom Column Names

If your CSV file uses different column names, you can specify them with parameters:

```bash
python Experiment.py \
  --mode parametric \
  --gamma-params-csv "your_gamma_params.csv" \
  --window-start-column "start_time" \
  --window-end-column "end_time" \
  --gamma-shape-column "shape" \
  --gamma-scale-column "scale" \
  --num-samples-column "count" \
  --dataset-path "/path/to/sharegpt/dataset.json" \
  --model "your-model-name"
```


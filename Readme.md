# FineServe

FineServe is a multi-model LLM serving workload dataset collected from a real-world commercial platform, providing fine-grained characterization of arrival dynamics and token behavior across heterogeneous models. It includes a configurable workload generator for benchmarking multi-model serving systems.

## Usage

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


```


#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import random
import resource
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np
import pandas as pd
import requests
from tqdm.asyncio import tqdm
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
ASSISTANT_SUFFIX = "Assistant:"

global args


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    extra_request_body: Dict[str, Any]
    index: int = 0


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    error: str = ""
    output_len: int = 0


def remove_prefix(text: str, prefix: str) -> str:
    return text[len(prefix):] if text.startswith(prefix) else text


def remove_suffix(text: str, suffix: str) -> str:
    return text[:-len(suffix)] if text.endswith(suffix) else text


def get_auth_headers() -> Dict[str, str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    else:
        return {}


def load_request_timestamps_from_csv(
    csv_path: str,
    timestamp_column: str = "timestamp",
    scale: float = 1.0,
) -> List[float]:
    """
    Load request arrival timestamps (microseconds) from CSV file and apply time compression
    
    Args:
        csv_path: Path to CSV file
        timestamp_column: Name of timestamp column (default "timestamp")
        scale: Time compression factor, >1 means speedup (compress time), <1 means slowdown
               For example, scale=2.0 compresses time to 1/2, doubling concurrency
    
    Returns:
        List of compressed relative timestamps (starting from 0, in seconds)
    
    Note:
        Timestamps are assumed to be microsecond-level Unix Epoch timestamps, automatically converted to seconds
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"Loading request timestamps from CSV: {csv_path}")
    print(f"Assuming timestamps are in microseconds")
    print(f"Time scale factor: {scale}x")
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    if timestamp_column not in df.columns:
        raise ValueError(
            f"Column '{timestamp_column}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )
    
    # 提取时间戳
    timestamps = df[timestamp_column].values
    
    # 转换为数值类型
    try:
        timestamps = timestamps.astype(float)
    except (ValueError, TypeError):
        raise ValueError(
            f"Cannot parse timestamp column '{timestamp_column}'. "
            f"Please ensure it's numeric format (microseconds)."
        )
    
    # 将微秒转换为秒
    timestamps = timestamps / 1000000.0
    
    # 计算相对时间
    if len(timestamps) > 0:
        timestamps = timestamps - timestamps[0]
    else:
        raise ValueError("CSV file contains no timestamps")
    
    # Apply time compression
    if scale > 0:
        timestamps = timestamps / scale
    else:
        raise ValueError(f"Scale factor must be > 0, got {scale}")
    
    print(f"Loaded {len(timestamps)} request timestamps")
    print(f"Original time span: {timestamps[-1] * scale:.2f}s")
    print(f"Compressed time span: {timestamps[-1]:.2f}s")
    
    return timestamps.tolist()


def load_request_lengths_from_csv(
    csv_path: str,
    input_length_column: str = "input_length",
    output_length_column: str = "output_length",
) -> List[Tuple[int, int]]:
    """
    Load input/output lengths of requests from CSV file
    
    Args:
        csv_path: Path to CSV file
        input_length_column: Name of input length column (default "input_length")
        output_length_column: Name of output length column (default "output_length")
    
    Returns:
        List of (input_length, output_length) tuples
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"Loading request lengths from CSV: {csv_path}")
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    if input_length_column not in df.columns:
        raise ValueError(
            f"Column '{input_length_column}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )
    
    if output_length_column not in df.columns:
        raise ValueError(
            f"Column '{output_length_column}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Extract length information
    input_lengths = df[input_length_column].values.astype(int)
    output_lengths = df[output_length_column].values.astype(int)
    
    lengths = list(zip(input_lengths, output_lengths))
    
    print(f"Loaded {len(lengths)} request length pairs")
    print(f"Input length range: [{input_lengths.min()}, {input_lengths.max()}]")
    print(f"Output length range: [{output_lengths.min()}, {output_lengths.max()}]")
    print(f"Mean input length: {input_lengths.mean():.1f}")
    print(f"Mean output length: {output_lengths.mean():.1f}")
    
    return lengths


async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    """Send request using OpenAI Completions API format (vLLM compatible)"""
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "completions"
    ), "OpenAI Completions API URL must end with 'completions'."

    prompt = request_func_input.prompt

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model,
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": request_func_input.output_len,
            "stream": not args.disable_stream,
            "ignore_eos": not args.disable_ignore_eos,
            **request_func_input.extra_request_body,
        }
        
        headers = get_auth_headers()

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        output_len = request_func_input.output_len
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        latency = time.perf_counter() - st
                        if chunk == "[DONE]":
                            pass
                        else:
                            data = json.loads(chunk)

                            # Check if there are generated tokens
                            if data["choices"][0]["text"]:
                                timestamp = time.perf_counter()
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft
                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += data["choices"][0]["text"]
                                output_len = (data.get("usage") or {}).get(
                                    "completion_tokens", output_len
                                )

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                    output.output_len = output_len
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


def get_model(pretrained_model_name_or_path: str) -> str:
    """Get model path"""
    return pretrained_model_name_or_path


def get_tokenizer(
    pretrained_model_name_or_path: str,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Get tokenizer"""
    if pretrained_model_name_or_path is not None and not os.path.exists(
        pretrained_model_name_or_path
    ):
        pretrained_model_name_or_path = get_model(pretrained_model_name_or_path)
    return AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=True
    )

# 这个是要更改的 改成 读取 我们的csv 进行测试
def sample_sharegpt_requests_with_pre(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    context_len: Optional[int] = None,
    prompt_suffix: Optional[str] = "",
    apply_chat_template=False,
) -> List[Tuple[str, int, int]]:
    """Load previous requests from saved JSON file"""
    
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    print(f"Loading requests from pre-saved file: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    if not isinstance(dataset, list):
        raise ValueError(f"Expected a list in JSON file, got {type(dataset)}")
    
    print(f"Loaded {len(dataset)} requests from file")
    
    num_requests = min(num_requests, len(dataset))
    
    filtered_dataset: List[Tuple[str, int, int]] = []
    
    for i in range(num_requests):
        data = dataset[i]
        
        prompt = data.get("input_text", "")
        if not prompt:
            print(f"Warning: Request {i} has empty input_text, skipping")
            continue
        
        if prompt_suffix:
            prompt = (
                remove_suffix(prompt, ASSISTANT_SUFFIX)
                + prompt_suffix
                + ASSISTANT_SUFFIX
            )
        
        if apply_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            prompt = prompt.replace(tokenizer.bos_token, "")
        
        prompt_token_ids = tokenizer.encode(prompt)
        prompt_len = len(prompt_token_ids)
        
        output_len = data.get("output_length", prompt_len // 2)
        
        if prompt_len < 2 or output_len < 2:
            print(f"Warning: Request {i} too short, skipping")
            continue
        
        if context_len and prompt_len + output_len > context_len:
            print(f"Warning: Request {i} too long, skipping")
            continue
        
        filtered_dataset.append((prompt, prompt_len, output_len))
    
    print(f"Successfully loaded {len(filtered_dataset)} requests")
    print(f"#Input tokens: {np.sum([x[1] for x in filtered_dataset])}")
    print(f"#Output tokens: {np.sum([x[2] for x in filtered_dataset])}")
    
    return filtered_dataset


SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"


def download_and_cache_file(url: str, filename: Optional[str] = None):
    """Download and cache a file from a url."""
    if filename is None:
        filename = os.path.join("/tmp", url.split("/")[-1])

    if os.path.exists(filename):
        return filename

    print(f"Downloading from {url} to {filename}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024

    with open(filename, "wb") as f, tqdm(
        desc=filename,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    return filename

# 这是很重要的 从sharegpt中 抽取 Prompt
def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
    context_len: Optional[int] = None,
    prompt_suffix: Optional[str] = "",
    apply_chat_template=False,
) -> List[Tuple[str, int, int]]:
    """Sample requests from ShareGPT dataset"""
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    if not os.path.isfile(dataset_path) and dataset_path == "":
        dataset_path = download_and_cache_file(SHAREGPT_URL)

    with open(dataset_path) as f:
        dataset = json.load(f)

    dataset = [
        data
        for data in dataset
        if len(data.get("conversations", data.get("conversation", []))) >= 2
    ]
    
    dataset = [
        (
            data.get("conversations", data.get("conversation", []))[0]["value"],
            data.get("conversations", data.get("conversation", []))[1]["value"],
        )
        for data in dataset
    ]

    # 不随机打乱
    # random.shuffle(dataset)

    # 计算目标数量
    target_short = num_requests
    target_long = int(num_requests - target_short)
    
    short_count = 0
    long_count = 0
    filtered_dataset = []
    
    # 边扫描边选择，达到目标立即停止
    for i in range(len(dataset)):
        # 提前退出：如果两类都收集够了
        if short_count >= target_short and long_count >= target_long:
            break
        
        prompt = dataset[i][0]
        if prompt_suffix:
            prompt = (
                remove_suffix(prompt, ASSISTANT_SUFFIX)
                + prompt_suffix
                + ASSISTANT_SUFFIX
            )

        if apply_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            prompt = prompt.replace(tokenizer.bos_token, "")

        prompt_token_ids = tokenizer.encode(prompt)
        completion = dataset[i][1]
        completion_token_ids = tokenizer.encode(completion)
        prompt_len = len(prompt_token_ids)
        output_len = (
            len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        )

        # 基本过滤
        if prompt_len < 2 or output_len < 2:
            continue
        

        if context_len and prompt_len + output_len > context_len:
            continue

        # 根据输出长度分类并选择
        if 5 <= output_len < 1000 and short_count < target_short:
            filtered_dataset.append((prompt, prompt_len, output_len))
            short_count += 1
        elif  output_len >= 1800 and long_count < target_long:
            filtered_dataset.append((prompt, prompt_len, output_len))
            long_count += 1

    # Statistics
    print(f"Short requests (5-200 tokens): {short_count}")
    print(f"Long requests (1000+ tokens): {long_count}")
    print(f"Total requests: {len(filtered_dataset)}")
    print(f"#Input tokens: {np.sum([x[1] for x in filtered_dataset])}")
    print(f"#Output tokens: {np.sum([x[2] for x in filtered_dataset])}")
    
    return filtered_dataset


def adjust_prompt_to_target_length(
    prompt: str,
    tokenizer: PreTrainedTokenizerBase,
    target_length: int,
) -> Tuple[str, int]:
    """
    Adjust prompt to target length
    
    Args:
        prompt: Original prompt
        tokenizer: Tokenizer
        target_length: Target length (in tokens)
    
    Returns:
        (adjusted_prompt, actual_length) tuple
    """
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    current_length = len(token_ids)
    
    if current_length == target_length:
        return prompt, current_length
    
    elif current_length < target_length:
        # Not long enough, repeat tokens until target length is reached
        repeat_times = (target_length // current_length) + 1
        extended_token_ids = (token_ids * repeat_times)[:target_length]
        adjusted_prompt = tokenizer.decode(extended_token_ids)
        return adjusted_prompt, len(extended_token_ids)
    
    else:
        # Too long, truncate to target length
        truncated_token_ids = token_ids[:target_length]
        adjusted_prompt = tokenizer.decode(truncated_token_ids)
        return adjusted_prompt, len(truncated_token_ids)


def get_dataset_sharegpt(
    args,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    """
    ShareGPT 模式：从 ShareGPT 数据集采样请求，使用原始的输入输出长度
    
    Args:
        args: Command line arguments
        tokenizer: Tokenizer
    
    Returns:
        List of (prompt, prompt_len, output_len) tuples
    """
    print("Using ShareGPT mode: sampling from ShareGPT dataset")
    
    input_requests = sample_sharegpt_requests(
        dataset_path=args.dataset_path,
        num_requests=args.num_prompts,
        tokenizer=tokenizer,
        fixed_output_len=args.sharegpt_output_len,
        context_len=args.sharegpt_context_len,
        prompt_suffix=args.prompt_suffix,
        apply_chat_template=args.apply_chat_template,
    )

    return input_requests


def get_dataset_replay(
    args,
    tokenizer: PreTrainedTokenizerBase,
    request_lengths: List[Tuple[int, int]],
) -> List[Tuple[str, int, int]]:
    """
    Replay mode: Read input/output lengths from CSV, sample prompts from ShareGPT and adjust lengths
    
    Args:
        args: Command line arguments
        tokenizer: Tokenizer
        request_lengths: List of (input_length, output_length) tuples read from CSV
    
    Returns:
        List of (prompt, prompt_len, output_len) tuples
    """
    print("Using Replay mode: loading lengths from CSV, prompts from ShareGPT")
    
    # Load original dataset from ShareGPT
    dataset_path = args.dataset_path
    if not os.path.isfile(dataset_path):
        if dataset_path == "":
            dataset_path = download_and_cache_file(SHAREGPT_URL)
        else:
            raise FileNotFoundError(f"ShareGPT dataset not found: {dataset_path}")
    
    with open(dataset_path) as f:
        sharegpt_dataset = json.load(f)
    
    # Filter out valid conversations
    sharegpt_dataset = [
        data
        for data in sharegpt_dataset
        if len(data.get("conversations", data.get("conversation", []))) >= 2
    ]
    
    # Extract prompts
    sharegpt_prompts = [
        data.get("conversations", data.get("conversation", []))[0]["value"]
        for data in sharegpt_dataset
    ]
    
    print(f"Loaded {len(sharegpt_prompts)} prompts from ShareGPT")
    print(f"Need to generate {len(request_lengths)} requests")
    
    # Build request list
    input_requests = []
    for i, (target_input_len, target_output_len) in enumerate(request_lengths):
        # 循环使用 ShareGPT prompt
        prompt_idx = i % len(sharegpt_prompts)
        original_prompt = sharegpt_prompts[prompt_idx]
        
        # 应用 prompt suffix 和 chat template（如果需要）
        prompt = original_prompt
        if args.prompt_suffix:
            prompt = (
                remove_suffix(prompt, ASSISTANT_SUFFIX)
                + args.prompt_suffix
                + ASSISTANT_SUFFIX
            )
        
        if args.apply_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            prompt = prompt.replace(tokenizer.bos_token, "")
        
        # Adjust prompt to target length
        adjusted_prompt, actual_input_len = adjust_prompt_to_target_length(
            prompt=prompt,
            tokenizer=tokenizer,
            target_length=target_input_len,
        )
        
        input_requests.append((adjusted_prompt, actual_input_len, target_output_len))
    
    print(f"Generated {len(input_requests)} requests with adjusted lengths")
    print(f"#Input tokens: {np.sum([x[1] for x in input_requests])}")
    print(f"#Output tokens: {np.sum([x[2] for x in input_requests])}")
    
    return input_requests


def get_dataset_parametric(
    args,
    tokenizer: PreTrainedTokenizerBase,
    **kwargs,
) -> List[Tuple[str, int, int]]:
    print("Using Parametric mode with gamma distribution")
    
    dataset_path = args.dataset_path
    if not os.path.isfile(dataset_path):
        if dataset_path == "":
            dataset_path = download_and_cache_file(SHAREGPT_URL)
        else:
            raise FileNotFoundError(f"ShareGPT dataset not found: {dataset_path}")
    
    with open(dataset_path) as f:
        sharegpt_dataset = json.load(f)
    
    sharegpt_dataset = [
        data
        for data in sharegpt_dataset
        if len(data.get("conversations", data.get("conversation", []))) >= 2
    ]
    
    sharegpt_prompts = [
        data.get("conversations", data.get("conversation", []))[0]["value"]
        for data in sharegpt_dataset
    ]
    
    print(f"Loaded {len(sharegpt_prompts)} prompts from ShareGPT")

    request_lengths = kwargs.get('request_lengths', None)

    input_requests = []
    num_requests = args.num_prompts
    
    for i in range(num_requests):
        prompt_idx = i % len(sharegpt_prompts)
        original_prompt = sharegpt_prompts[prompt_idx]
        
        prompt = original_prompt
        if args.prompt_suffix:
            prompt = (
                remove_suffix(prompt, ASSISTANT_SUFFIX)
                + args.prompt_suffix
                + ASSISTANT_SUFFIX
            )
        
        if args.apply_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            prompt = prompt.replace(tokenizer.bos_token, "")
        
        if request_lengths and i < len(request_lengths):
            target_input_len, target_output_len = request_lengths[i]
            adjusted_prompt, actual_input_len = adjust_prompt_to_target_length(
                prompt=prompt,
                tokenizer=tokenizer,
                target_length=target_input_len,
            )
        else:
            adjusted_prompt = prompt
            token_ids = tokenizer.encode(adjusted_prompt, add_special_tokens=False)
            actual_input_len = len(token_ids)
            target_output_len = args.sharegpt_output_len if args.sharegpt_output_len else actual_input_len // 2
        
        input_requests.append((adjusted_prompt, actual_input_len, target_output_len))
    
    print(f"Generated {len(input_requests)} requests with parametric configuration")
    print(f"#Input tokens: {np.sum([x[1] for x in input_requests])}")
    print(f"#Output tokens: {np.sum([x[2] for x in input_requests])}")
    
    return input_requests


def get_dataset(
    args,
    tokenizer: PreTrainedTokenizerBase,
    mode: str,
    request_lengths: Optional[List[Tuple[int, int]]] = None,
    **parametric_kwargs,
) -> List[Tuple[str, int, int]]:
    """
    Unified dataset acquisition interface supporting three modes:
    1. sharegpt: Sample from ShareGPT dataset, use original lengths
    2. replay: Read lengths from CSV, sample prompts from ShareGPT and adjust lengths
    3. parametric: Generate using parametric approach (to be implemented)
    
    Args:
        args: Command line arguments
        tokenizer: Tokenizer
        mode: Dataset mode ('sharegpt', 'replay', 'parametric')
        request_lengths: Length list required for replay mode
        **parametric_kwargs: Parameters for parametric mode
    
    Returns:
        List of (prompt, prompt_len, output_len) tuples
    """
    if mode == "sharegpt":
        return get_dataset_sharegpt(args, tokenizer)
    
    elif mode == "replay":
        if request_lengths is None:
            raise ValueError("replay mode requires request_lengths")
        return get_dataset_replay(args, tokenizer, request_lengths)
    
    elif mode == "parametric":
        return get_dataset_parametric(args, tokenizer, **parametric_kwargs)
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be one of: sharegpt, replay, parametric")


ASYNC_REQUEST_FUNCS = {
    "vllm": async_request_openai_completions,
    "openai": async_request_openai_completions,
}


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    total_output_retokenized: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    output_throughput_retokenized: float
    total_throughput: float
    total_throughput_retokenized: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    p99_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    p95_itl_ms: float
    p99_itl_ms: float
    max_itl_ms: float
    mean_e2e_latency_ms: float
    median_e2e_latency_ms: float
    std_e2e_latency_ms: float
    p99_e2e_latency_ms: float
    concurrency: float


async def get_request_replay(
    input_requests: List[Tuple[str, int, int]],
    request_timestamps: List[float],
) -> AsyncGenerator[Tuple[str, int, int], None]:
    """
    Replay mode: Replay requests based on CSV timestamps
    
    Args:
        input_requests: List of requests
        request_timestamps: List of timestamps (relative time, in seconds)
    """
    timestamps_iter = iter(request_timestamps)
    start_time = time.perf_counter()
    
    for i, request in enumerate(input_requests):
        if i < len(request_timestamps):
            target_timestamp = next(timestamps_iter, None)
            if target_timestamp is not None:
                elapsed = time.perf_counter() - start_time
                wait_time = target_timestamp - elapsed
                
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                
                yield request
            else:
                yield request
        else:
            yield request


async def get_request_poisson(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    """
    Poisson mode: Generate requests using Poisson distribution
    
    Args:
        input_requests: List of requests
        request_rate: Request rate (req/s), inf indicates burst mode
    """
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            continue

        interval = np.random.exponential(1.0 / request_rate)
        await asyncio.sleep(interval)


async def get_request_parametric(
    input_requests: List[Tuple[str, int, int]],
    **kwargs,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    gamma_params = kwargs.get('gamma_params', None)
    
    if gamma_params is None or len(gamma_params) == 0:
        print("Warning: No gamma parameters provided, using uniform intervals")
        for request in input_requests:
            yield request
            await asyncio.sleep(0.01) 
        return
    
    start_time = time.perf_counter()
    request_idx = 0
    window_idx = 0
    window_start_time = 0.0  
    window_duration = 300.0  
    
    print(f"Starting parametric request generation with {len(gamma_params)} gamma parameter windows")
    print(f"Total requests: {len(input_requests)}")
    
    for request in input_requests:
        elapsed_time = time.perf_counter() - start_time
        current_window_idx = int(elapsed_time / window_duration)
        
        if current_window_idx > window_idx and current_window_idx < len(gamma_params):
            window_idx = current_window_idx
            window_start_time = window_idx * window_duration
            print(f"Switching to gamma parameter window {window_idx + 1}/{len(gamma_params)}: "
                  f"shape={gamma_params[window_idx][2]:.4f}, scale={gamma_params[window_idx][3]:.4f}")
        
        if window_idx >= len(gamma_params):
            window_idx = len(gamma_params) - 1
        
        yield request
        request_idx += 1
        
        if request_idx < len(input_requests):
            gamma_shape = gamma_params[window_idx][2]
            gamma_scale_ms = gamma_params[window_idx][3]  
            
            gamma_scale_sec = gamma_scale_ms / 1000.0
            
            interval = np.random.gamma(gamma_shape, gamma_scale_sec)
            
            await asyncio.sleep(interval)

def load_gamma_params_from_cv_csv(
    csv_path: str,
    window_start_column: str = "window_start_ms",
    window_end_column: str = "window_end_ms",
    gamma_shape_column: str = "gamma_shape",
    gamma_scale_column: str = "gamma_scale",
) -> List[Tuple[int, int, float, float]]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CV CSV file not found: {csv_path}")
    
    print(f"Loading gamma parameters from CV CSV: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    required_columns = [
        window_start_column,
        window_end_column,
        gamma_shape_column,
        gamma_scale_column,
    ]
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )
    
    window_starts = df[window_start_column].values.astype(int)
    window_ends = df[window_end_column].values.astype(int)
    gamma_shapes = df[gamma_shape_column].values.astype(float)
    gamma_scales = df[gamma_scale_column].values.astype(float)
    
    gamma_params = list(zip(
        window_starts,
        window_ends,
        gamma_shapes,
        gamma_scales,
    ))
    
    print(f"Loaded {len(gamma_params)} gamma parameter windows")
    print(f"Time span: {(window_ends[-1] - window_starts[0]) / 1000.0:.2f} seconds "
          f"({(window_ends[-1] - window_starts[0]) / 60000.0:.2f} minutes)")
    print(f"Gamma shape range: [{gamma_shapes.min():.4f}, {gamma_shapes.max():.4f}]")
    print(f"Gamma scale range: [{gamma_scales.min():.4f}, {gamma_scales.max():.4f}]")
    print(f"Mean gamma shape: {gamma_shapes.mean():.4f}")
    print(f"Mean gamma scale: {gamma_scales.mean():.4f}")
    
    return gamma_params



async def get_request(
    input_requests: List[Tuple[str, int, int]],
    mode: str,
    request_rate: Optional[float] = None,
    request_timestamps: Optional[List[float]] = None,
    **parametric_kwargs,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    """
    Unified request generation interface supporting three modes:
    1. replay: Replay requests based on CSV timestamps
    2. poisson: Generate requests using Poisson distribution
    3. parametric: Use parameters to synthesize request arrival times
    
    Args:
        input_requests: List of requests
        mode: Request generation mode ('replay', 'poisson', 'parametric')
        request_rate: Request rate (used in poisson mode)
        request_timestamps: List of timestamps (used in replay mode)
        **parametric_kwargs: Configuration parameters for parametric mode
    """
    if mode == "replay":
        if request_timestamps is None:
            raise ValueError("replay mode requires request_timestamps")
        async for request in get_request_replay(input_requests, request_timestamps):
            yield request
    
    elif mode == "poisson":
        if request_rate is None:
            raise ValueError("poisson mode requires request_rate")
        async for request in get_request_poisson(input_requests, request_rate):
            yield request
    
    elif mode == "parametric":
        print(f"\n=== Parametric Mode Configuration ===")
        print("Using parametric request generation with gamma distribution")
        
        if not hasattr(args, 'gamma_params_csv') or not args.gamma_params_csv:
            raise ValueError("parametric mode requires --gamma-params-csv argument")
        
        print(f"Loading gamma parameters from: {args.gamma_params_csv}")
        gamma_params = load_gamma_params_from_cv_csv(
            csv_path=args.gamma_params_csv,
            window_start_column=getattr(args, 'window_start_column', 'window_start_ms'),
            window_end_column=getattr(args, 'window_end_column', 'window_end_ms'),
            gamma_shape_column=getattr(args, 'gamma_shape_column', 'gamma_shape'),
            gamma_scale_column=getattr(args, 'gamma_scale_column', 'gamma_scale'),
        )
        
        parametric_kwargs['gamma_params'] = gamma_params
        
        if hasattr(args, 'request_lengths_csv') and args.request_lengths_csv:
            print(f"Loading request lengths from: {args.request_lengths_csv}")
            request_lengths = load_request_lengths_from_csv(
                csv_path=args.request_lengths_csv,
                input_length_column=getattr(args, 'input_length_column', 'input_length'),
                output_length_column=getattr(args, 'output_length_column', 'output_length'),
            )
            parametric_kwargs['request_lengths'] = request_lengths
        
        dataset_mode = "parametric"
        print(f"===\n")
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be one of: replay, poisson, parametric")


def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
) -> Tuple[BenchmarkMetrics, List[int]]:
    """Calculate performance metrics"""
    output_lens: List[int] = []
    retokenized_output_lens: List[int] = []
    total_input = 0
    completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    e2e_latencies: List[float] = []
    
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_len
            output_lens.append(output_len)
            retokenized_output_len = len(
                tokenizer.encode(outputs[i].generated_text, add_special_tokens=False)
            )
            retokenized_output_lens.append(retokenized_output_len)
            total_input += input_requests[i][1]
            if output_len > 1:
                tpots.append((outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2e_latencies.append(outputs[i].latency)
            completed += 1
        else:
            output_lens.append(0)
            retokenized_output_lens.append(0)

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration.",
            stacklevel=2,
        )
        
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(output_lens),
        total_output_retokenized=sum(retokenized_output_lens),
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(output_lens) / dur_s,
        output_throughput_retokenized=sum(retokenized_output_lens) / dur_s,
        total_throughput=(total_input + sum(output_lens)) / dur_s,
        total_throughput_retokenized=(total_input + sum(retokenized_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        p99_tpot_ms=np.percentile(tpots or 0, 99) * 1000,
        mean_itl_ms=np.mean(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        p95_itl_ms=np.percentile(itls or 0, 95) * 1000,
        p99_itl_ms=np.percentile(itls or 0, 99) * 1000,
        max_itl_ms=np.max(itls or 0) * 1000,
        mean_e2e_latency_ms=np.mean(e2e_latencies or 0) * 1000,
        median_e2e_latency_ms=np.median(e2e_latencies or 0) * 1000,
        std_e2e_latency_ms=np.std(e2e_latencies or 0) * 1000,
        p99_e2e_latency_ms=np.percentile(e2e_latencies or 0, 99) * 1000,
        concurrency=np.sum(e2e_latencies) / dur_s if dur_s > 0 else 0,
    )

    return metrics, output_lens

async def benchmark(
    backend: str,
    api_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int]],
    mode: str,
    request_rate: Optional[float] = None,
    max_concurrency: Optional[int] = None,
    disable_tqdm: bool = False,
    extra_request_body: Optional[Dict[str, Any]] = None,
    request_timestamps: Optional[List[float]] = None,
    parametric_kwargs: Optional[Dict[str, Any]] = None,
):
    """Run benchmark"""
    if extra_request_body is None:
        extra_request_body = {}
    if parametric_kwargs is None:
        parametric_kwargs = {}
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Limit concurrency
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await request_func(request_func_input=request_func_input, pbar=pbar)
        async with semaphore:
            return await request_func(request_func_input=request_func_input, pbar=pbar)

    # Warmup
    print(f"Starting warmup with {args.warmup_requests} sequences...")
    test_prompt, test_prompt_len, test_output_len = input_requests[0]
    test_input = RequestFuncInput(
        model=model_id,
        prompt=test_prompt,
        api_url=api_url,
        prompt_len=test_prompt_len,
        output_len=min(test_output_len, 32),
        extra_request_body=extra_request_body,
    )

    warmup_tasks = []
    for _ in range(args.warmup_requests):
        warmup_tasks.append(
            asyncio.create_task(request_func(request_func_input=test_input))
        )

    warmup_outputs = await asyncio.gather(*warmup_tasks)

    if args.warmup_requests > 0 and not any(
        output.success for output in warmup_outputs
    ):
        raise ValueError(
            "Warmup failed - Please check your configuration. "
            f"Error: {warmup_outputs[0].error}"
        )
    else:
        print(f"Warmup completed. Starting main benchmark run...")

    time.sleep(1.0)

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    # Run all requests
    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []
    req_index = 0
    
    async for request in get_request(
        input_requests=input_requests,
        mode=mode,
        request_rate=request_rate,
        request_timestamps=request_timestamps,
        **parametric_kwargs,
    ):
        prompt, prompt_len, output_len = request

        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            extra_request_body=extra_request_body,
            index=req_index,
        )
        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request_func_input, pbar=pbar)
            )
        )
        req_index += 1
        
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if pbar is not None:
        pbar.close()

    # Calculate metrics
    benchmark_duration = time.perf_counter() - benchmark_start_time
    metrics, output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
    )

    # Print results
    print("\n{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Backend:", backend))
    print("{:<40} {:<10}".format("Request generation mode:", mode))
    print("{:<40} {:<10}".format("Traffic request rate:", request_rate if request_rate else "N/A"))
    print("{:<40} {:<10}".format(
        "Max request concurrency:",
        max_concurrency if max_concurrency else "not set",
    ))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    failed_count = len(input_requests) - metrics.completed
    print("{:<40} {:<10}".format("Failed requests:", failed_count))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print("{:<40} {:<10.2f}".format(
        "Request throughput (req/s):", metrics.request_throughput
    ))
    print("{:<40} {:<10.2f}".format(
        "Input token throughput (tok/s):", metrics.input_throughput
    ))
    print("{:<40} {:<10.2f}".format(
        "Output token throughput (tok/s):", metrics.output_throughput
    ))
    print("{:<40} {:<10.2f}".format(
        "Total token throughput (tok/s):", metrics.total_throughput
    ))
    print("{:<40} {:<10.2f}".format("Concurrency:", metrics.concurrency))
    print("{s:{c}^{n}}".format(s="End-to-End Latency", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean E2E Latency (ms):", metrics.mean_e2e_latency_ms))
    print("{:<40} {:<10.2f}".format("Median E2E Latency (ms):", metrics.median_e2e_latency_ms))
    print("{:<40} {:<10.2f}".format("P99 E2E Latency (ms):", metrics.p99_e2e_latency_ms))
    print("{s:{c}^{n}}".format(s="Time to First Token", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("Median TTFT (ms):", metrics.median_ttft_ms))
    print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
    print("{s:{c}^{n}}".format(s="Inter-Token Latency", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean ITL (ms):", metrics.mean_itl_ms))
    print("{:<40} {:<10.2f}".format("Median ITL (ms):", metrics.median_itl_ms))
    print("{:<40} {:<10.2f}".format("P95 ITL (ms):", metrics.p95_itl_ms))
    print("{:<40} {:<10.2f}".format("P99 ITL (ms):", metrics.p99_itl_ms))
    print("{:<40} {:<10.2f}".format("Max ITL (ms):", metrics.max_itl_ms))
    print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", metrics.mean_tpot_ms))
    print("{:<40} {:<10.2f}".format("Median TPOT (ms):", metrics.median_tpot_ms))
    print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))
    print("=" * 50)
    
    # Display detailed error information for failed requests
    if failed_count > 0:
        print("\n{s:{c}^{n}}".format(s=" Failed Requests Details ", n=50, c="!"))
        print(f"Total failed requests: {failed_count}")
        
        # Collect failed requests
        failed_outputs = [(i, out) for i, out in enumerate(outputs) if not out.success]
        
        # Show errors for first 5 failed requests
        print(f"\nShowing first {min(5, len(failed_outputs))} failed request errors:\n")
        for idx, (req_idx, out) in enumerate(failed_outputs[:5], 1):
            print(f"--- Failed Request #{idx} (index {req_idx}) ---")
            if out.error:
                # Limit error message length to avoid being too long
                error_msg = out.error[:500]
                if len(out.error) > 500:
                    error_msg += "\n... (truncated)"
                print(error_msg)
            else:
                print("No error message available")
            print()
        
        if len(failed_outputs) > 5:
            print(f"... and {len(failed_outputs) - 5} more failed requests.")
            print("Check the output JSON file for all error details.")
        print("!" * 50)
        print()

    result = {
        # Arguments
        "backend": args.backend,
        "mode": mode,
        "request_rate": request_rate,
        "max_concurrency": max_concurrency,
        # Results
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "total_output_tokens_retokenized": metrics.total_output_retokenized,
        "request_throughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
        "mean_e2e_latency_ms": metrics.mean_e2e_latency_ms,
        "median_e2e_latency_ms": metrics.median_e2e_latency_ms,
        "std_e2e_latency_ms": metrics.std_e2e_latency_ms,
        "p99_e2e_latency_ms": metrics.p99_e2e_latency_ms,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "std_ttft_ms": metrics.std_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "std_tpot_ms": metrics.std_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms,
        "mean_itl_ms": metrics.mean_itl_ms,
        "median_itl_ms": metrics.median_itl_ms,
        "std_itl_ms": metrics.std_itl_ms,
        "p95_itl_ms": metrics.p95_itl_ms,
        "p99_itl_ms": metrics.p99_itl_ms,
        "concurrency": metrics.concurrency,
    }

    # Save results
    if args.output_file:
        output_file_name = args.output_file
    else:
        now = datetime.now().strftime("%m%d")
        if args.dataset_name == "random":
            output_file_name = f"{args.backend}_{now}_{args.num_prompts}_{args.random_input_len}_{args.random_output_len}.jsonl"
        else:
            output_file_name = f"{args.backend}_{now}_{args.num_prompts}_sharegpt.jsonl"

    with open(output_file_name, "a") as file:
        file.write(json.dumps(result) + "\n")

    result.update(
        {
            "input_lens": [output.prompt_len for output in outputs],
            "output_lens": output_lens,
            "ttfts": [output.ttft for output in outputs],
            "itls": [output.itl for output in outputs],
            "generated_texts": [output.generated_text for output in outputs],
            "errors": [output.error for output in outputs],
        }
    )
    return result


def set_ulimit(target_soft_limit=65535):
    """Set file descriptor limit"""
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            print(f"Fail to set RLIMIT_NOFILE: {e}")


def run_benchmark(args_):
    """Main function to run benchmark"""
    global args
    args = args_

    if not hasattr(args, "max_concurrency"):
        args.max_concurrency = None

    if not hasattr(args, "warmup_requests"):
        args.warmup_requests = 1

    print(f"benchmark_args={args}")

    # Set random seed
    set_ulimit()
    if args.seed == -1:
        args.seed = int(time.time())
    random.seed(args.seed)
    np.random.seed(args.seed)

    extra_request_body = {}
    if args.extra_request_body:
        extra_request_body = json.loads(args.extra_request_body)

    # Set URL
    if args.port is None:
        args.port = 8000

    api_url = (
        f"{args.base_url}/v1/completions"
        if args.base_url
        else f"http://{args.host}:{args.port}/v1/completions"
    )

    # Get model name
    if args.model is None:
        model_url = (
            f"{args.base_url}/v1/models"
            if args.base_url
            else f"http://{args.host}:{args.port}/v1/models"
        )
        try:
            response = requests.get(model_url, headers=get_auth_headers())
            model_list = response.json().get("data", [])
            args.model = model_list[0]["id"] if model_list else None
        except Exception as e:
            print(f"Failed to fetch model from {model_url}. Error: {e}")
            sys.exit(1)

    if args.model is None:
        print("No model specified or found. Please provide a model using `--model`.")
        sys.exit(1)

    print(f"{args}\n")

    # Initialize
    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer = get_tokenizer(tokenizer_id)

    # Prepare dataset configuration based on mode parameter
    mode = args.mode
    request_lengths = None
    request_timestamps = None
    parametric_kwargs = {}
    dataset_mode = "sharegpt"  # Default to sharegpt mode for loading data
    
    if mode == "replay":
        # Replay mode: Load timestamps and lengths from CSV
        if not hasattr(args, 'request_trace_csv') or not args.request_trace_csv:
            raise ValueError("replay mode requires --request-trace-csv argument")
        
        print(f"\n=== Replay Mode Configuration ===")
        print(f"Loading from CSV: {args.request_trace_csv}")
        
        # 加载请求到达时间戳（微秒级）
        request_timestamps = load_request_timestamps_from_csv(
            csv_path=args.request_trace_csv,
            timestamp_column=getattr(args, 'timestamp_column', 'timestamp'),
            scale=getattr(args, 'time_scale', 1.0),
        )
        
        # 加载请求输入输出长度
        request_lengths = load_request_lengths_from_csv(
            csv_path=args.request_trace_csv,
            input_length_column=getattr(args, 'input_length_column', 'input_length'),
            output_length_column=getattr(args, 'output_length_column', 'output_length'),
        )
        
        # 确保时间戳和长度数量一致
        min_count = min(len(request_timestamps), len(request_lengths))
        if len(request_timestamps) != len(request_lengths):
            print(f"Warning: Mismatch between timestamps ({len(request_timestamps)}) "
                  f"and lengths ({len(request_lengths)})")
            print(f"Using first {min_count} entries")
            request_timestamps = request_timestamps[:min_count]
            request_lengths = request_lengths[:min_count]
        
        dataset_mode = "replay"
        print(f"=== Ready to generate {len(request_lengths)} replay requests ===\n")
    
    elif mode == "poisson":
        # Poisson mode: Use request_rate parameter
        print(f"\n=== Poisson Mode Configuration ===")
        print(f"Request rate: {args.request_rate} req/s")
        print(f"Using ShareGPT dataset with original lengths")
        dataset_mode = "sharegpt"
        print(f"===\n")
    
    elif mode == "parametric":
        # Parametric mode: Use parameter synthesis
        print(f"\n=== Parametric Mode Configuration ===")
        print("Using parametric request generation (to be implemented)")
        # parametric_kwargs = {...}
        dataset_mode = "parametric"
        print(f"===\n")
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be one of: replay, poisson, parametric")
    
    # Generate dataset
    print(f"Loading dataset with mode: {dataset_mode}")
    input_requests = get_dataset(
        args=args,
        tokenizer=tokenizer,
        mode=dataset_mode,
        request_lengths=request_lengths,
        **parametric_kwargs,
    )
    
    return asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            model_id=model_id,
            tokenizer=tokenizer,
            input_requests=input_requests,
            mode=mode,
            request_rate=args.request_rate,
            max_concurrency=args.max_concurrency,
            disable_tqdm=args.disable_tqdm,
            extra_request_body=extra_request_body,
            request_timestamps=request_timestamps,
            parametric_kwargs=parametric_kwargs,
        )
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FineServe.")
    
    parser.add_argument("--backend", type=str, choices=list(ASYNC_REQUEST_FUNCS.keys()), default="vllm", help="Backend to use (vllm or openai)")
    parser.add_argument("--base-url", type=str, default=None, help="Server base URL (e.g., http://localhost:8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    
    parser.add_argument("--dataset-name", type=str, default="sharegpt", choices=["sharegpt", "sharegpt_pre", "random"], help="Dataset to use")
    parser.add_argument("--dataset-path", type=str, default="", help="Path to the dataset file")
    parser.add_argument("--model", type=str, help="Model name or path")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer name or path")
    
    parser.add_argument("--num-prompts", type=int, default=100, help="Number of prompts to process")
    parser.add_argument("--sharegpt-output-len", type=int, default=None, help="Fixed output length for ShareGPT dataset")
    parser.add_argument("--sharegpt-context-len", type=int, default=None, help="Context length limit for ShareGPT dataset")
    parser.add_argument("--random-input-len", type=int, default=1024, help="Input length for random dataset")
    parser.add_argument("--random-output-len", type=int, default=128, help="Output length for random dataset")
    parser.add_argument("--random-range-ratio", type=float, default=0.0, help="Range ratio for random dataset")
    
    parser.add_argument("--mode", type=str, default="poisson", choices=["replay", "poisson", "parametric"], help="Request generation mode: 'replay' - replay from CSV timestamps; 'poisson' - Poisson distribution with request_rate; 'parametric' - synthetic generation with parameters")
    parser.add_argument("--request-rate", type=float, default=float("inf"), help="Request rate (req/s) for Poisson mode. Use inf for burst mode")
    parser.add_argument("--max-concurrency", type=int, default=None, help="Maximum concurrent requests")
    parser.add_argument("--output-file", type=str, help="Output JSONL file name")
    
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable tqdm progress bar")
    parser.add_argument("--disable-stream", action="store_true", help="Disable streaming mode")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 for current time)")
    parser.add_argument("--disable-ignore-eos", action="store_true", help="Disable ignoring EOS token")
    parser.add_argument("--extra-request-body", type=str, help='Extra JSON to append to request body (e.g., \'{"key": "value"}\')')
    parser.add_argument("--apply-chat-template", action="store_true", help="Apply chat template to prompts")
    parser.add_argument("--prompt-suffix", type=str, default="", help="Suffix to append to prompts")
    parser.add_argument("--warmup-requests", type=int, default=1, help="Number of warmup requests")
    
    # Parametric mode arguments
    parser.add_argument("--gamma-params-csv", type=str, default=None, help="CSV file with gamma distribution parameters (for parametric mode)")
    parser.add_argument("--window-start-column", type=str, default="window_start_ms", help="Name of window start column in gamma params CSV (milliseconds)")
    parser.add_argument("--window-end-column", type=str, default="window_end_ms", help="Name of window end column in gamma params CSV (milliseconds)")
    parser.add_argument("--gamma-shape-column", type=str, default="gamma_shape", help="Name of gamma shape parameter column in CSV")
    parser.add_argument("--gamma-scale-column", type=str, default="gamma_scale", help="Name of gamma scale parameter column in CSV")
    parser.add_argument("--request-lengths-csv", type=str, default=None, help="CSV file with request input/output lengths (optional for parametric mode)")
    
    parser.add_argument("--request-trace-csv", type=str, default=None, help="CSV file with request timestamps (microseconds) for replay")
    parser.add_argument("--timestamp-column", type=str, default="timestamp", help="Name of timestamp column in CSV (microseconds)")
    parser.add_argument("--time-scale", type=float, default=1.0, help="Time scale factor for CSV replay (>1 = faster)")
    parser.add_argument("--input-length-column", type=str, default="input_length", help="Name of input length column in CSV (for replay mode)")
    parser.add_argument("--output-length-column", type=str, default="output_length", help="Name of output length column in CSV (for replay mode)")
    
    args = parser.parse_args()
    run_benchmark(args)


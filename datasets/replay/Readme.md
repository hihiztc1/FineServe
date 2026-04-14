# Replay Traces

This directory contains desensitized request trace files used in **Replay mode**, where the generator replays real-world request arrival patterns (timestamps and token lengths) from a commercial LLM serving platform.

---

## File Format

Each CSV file contains one row per request with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `input_length` | int | Number of input tokens in the request |
| `output_length` | int | Number of output tokens in the response |
| `latency` | float | End-to-end latency in seconds (observed) |
| `timestamp` | float | Request arrival time in microseconds, relative to the first request (starting from 0) |
| `architecture` | str | Model architecture category (e.g., `Dense`, `MoE`) |
| `scale` | str | Model scale category (e.g., `10-30B`, `lt10B`, `gt100B`) |

---

## Example Data

The files provided in this directory are **example traces for one representative day** (~18M requests), intended for quick-start testing and benchmarking.


## Download

These files are tracked with **Git LFS**. Clone the repository normally and Git LFS will handle the download automatically:

```bash
git lfs install
git clone <repo-url>
```

Or pull LFS objects explicitly:

```bash
git lfs pull
```

---

## Extended Dataset (Continuously Updated)

A larger release (~172M requests) is available on Google Drive and is **continuously being expanded**:

> [FineServe Dataset on Google Drive](https://drive.google.com/drive/folders/1iuTnV7IUj41-tnVLvCMTR2ANifRaEajA?usp=drive_link)

Note: this is a partial release of the full dataset. Longer time spans, and finer-grained traces are being progressively added.

---


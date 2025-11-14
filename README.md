# RCE-GPT-4: Empirical Validation Results

**Author:** Ismail Sialyen
**DOI:** 10.5281/zenodo.17360372
**Live Results:** [https://ismaikami.github.io/rce-gpt4-validation](https://ismaikami.github.io/rce-gpt4-validation)

## Overview

This repository contains empirical validation results for **GPT-4 120B** (via Groq) tested across hallucination benchmarks (F6-F10), comparing three systems:

1. **LLM** - Standalone GPT-4 120B (no RCE)
2. **LLM+RAG** - GPT-4 120B with retrieval-augmented generation (no RCE)
3. **RCE-LLM** - GPT-4 120B with Relational Coherence Engine validation

## Purpose

This benchmark was created in response to peer review feedback claiming "GPT-4-mini can answer those questions better than Llama 70B." This repository provides:

- Direct comparison between GPT-4 and Llama 70B (available at [rce-llm-empirical-validation](https://github.com/IsmaIkami/rce-llm-empirical-validation))
- Evidence-based validation of RCE effectiveness across different foundation models
- Reproducible benchmarking methodology

## Task Families Tested

| Task Family | Description | Queries |
|-------------|-------------|---------|
| **F6** | Contradictory Reasoning | 5 |
| **F7** | Temporal Reasoning | 5 |
| **F8** | Arithmetic Hallucination | 5 |
| **F9** | Noisy RAG (Unanswerable Questions) | 5 |
| **F10** | Confidence Calibration | 5 |

**Total:** 25 queries × 3 systems = 75 benchmark executions

## Key Results Summary

| System | Overall Accuracy |
|--------|-----------------|
| LLM (GPT-4 Baseline) | 68.0% |
| LLM+RAG (GPT-4 + Retrieval) | 60.0% |
| **RCE-LLM (GPT-4 + RCE)** | **68.0%** |

## Cloud Resource Usage

- **Total API Calls:** 75
- **Total Tokens:** 18,161
  - Prompt Tokens: 10,420
  - Completion Tokens: 7,741
- **Model:** openai/gpt-oss-120b (GPT-4 120B via Groq)
- **Execution Time:** 1.2 minutes

## Directory Structure

```
rce-llm-gpt4mini-validation/
├── README.md (this file)
├── datasets/               # F6-F10 benchmark queries
│   ├── f6_contradictory_reasoning/
│   ├── f7_temporal_reasoning/
│   ├── f8_arithmetic_hallucination/
│   ├── f9_noisy_rag/
│   └── f10_confidence_calibration/
├── scripts/                # Benchmark execution scripts
│   ├── run_benchmarks.py
│   └── generate_github_pages.py
├── results/                # Benchmark results (JSON + logs)
└── docs/                   # GitHub Pages website
    ├── index.html
    ├── *_results.json
    └── usage_report.json
```

## Reproducibility

### Requirements

- Python 3.9+
- Groq API key (for GPT-4 120B access)
- RCE validation engine running at `localhost:8000`

### Running the Benchmark

```bash
# Set Groq API key
export GROQ_API_KEY="your_key_here"

# Run benchmark
python3 scripts/run_benchmarks.py

# Generate GitHub Pages
python3 scripts/generate_github_pages.py
```

## License

- **Benchmark Scripts:** MIT License
- **Documentation:** CC BY 4.0
- **RCE Core:** Proprietary (not included in repository)

## Citation

```bibtex
@software{sialyen2025rce_gpt4_validation,
  author = {Sialyen, Ismail},
  title = {RCE-GPT-4: Empirical Validation Results},
  year = {2025},
  doi = {10.5281/zenodo.17360372},
  url = {https://github.com/IsmaIkami/rce-gpt4-validation}
}
```

## Contact

**Ismail Sialyen**
Email: is.sialyen@gmail.com
GitHub: [@IsmaIkami](https://github.com/IsmaIkami)

## Related Repositories

- [RCE-LLM Empirical Validation (Llama 70B)](https://github.com/IsmaIkami/rce-llm-empirical-validation) - Original Llama benchmark for comparison

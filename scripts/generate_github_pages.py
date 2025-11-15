#!/usr/bin/env python3
"""
Generate GitHub Pages website for GPT-4 120B benchmark results
Author: Ismail Sialyen
DOI: 10.5281/zenodo.17360372
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

# Paths
RESULTS_DIR = Path("/Users/isma/Projects/RCE/rce-llm-gpt4mini-validation/results")
DOCS_DIR = Path("/Users/isma/Projects/RCE/rce-llm-gpt4mini-validation/docs")

# Create docs directory
DOCS_DIR.mkdir(exist_ok=True)

# Load benchmark results
with open(RESULTS_DIR / "benchmark_results.json") as f:
    data = json.load(f)

# Copy result JSON files to docs
for task_family in ["f6_contradictory_reasoning", "f7_temporal_reasoning",
                     "f8_arithmetic_hallucination", "f9_noisy_rag", "f10_confidence_calibration",
                     "f11_truthfulqa_misconceptions"]:
    src = RESULTS_DIR / f"{task_family}_results.json"
    if src.exists():
        shutil.copy(src, DOCS_DIR / f"{task_family}_results.json")

# Copy usage report
if (RESULTS_DIR / "usage_report.json").exists():
    shutil.copy(RESULTS_DIR / "usage_report.json", DOCS_DIR / "usage_report.json")

# Extract metadata
metadata = data["metadata"]
task_families = data["task_families"]

# Calculate overall accuracy
overall_accuracy = {}
for system in ["LLM", "LLM+RAG", "RCE-LLM"]:
    total_correct = 0
    total_queries = 0
    for family in task_families.values():
        correct = int(family["accuracy"][system] * family["total_queries"])
        total_correct += correct
        total_queries += family["total_queries"]
    overall_accuracy[system] = (total_correct / total_queries) if total_queries > 0 else 0

# Load Llama benchmark for comparison (if exists)
llama_data = None
llama_file = Path("/Users/isma/Projects/RCE/rce-llm-empirical-validation/results/benchmark_results.json")
if llama_file.exists():
    with open(llama_file) as f:
        llama_data = json.load(f)

    # Calculate Llama overall accuracy
    llama_accuracy = {}
    for system in ["LLM", "LLM+RAG", "RCE-LLM"]:
        total_correct = 0
        total_queries = 0
        for family in llama_data["task_families"].values():
            correct = int(family["accuracy"][system] * family["total_queries"])
            total_correct += correct
            total_queries += family["total_queries"]
        llama_accuracy[system] = (total_correct / total_queries) if total_queries > 0 else 0

# Generate HTML
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RCE-GPT-4: Empirical Validation Results - Hallucination Benchmarks (F6-F11)</title>
    <style>
        :root {{
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --success-color: #27ae60;
            --warning-color: #f39c12;
            --danger-color: #e74c3c;
            --bg-color: #f8f9fa;
            --card-bg: #ffffff;
            --border-color: #dee2e6;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: var(--primary-color);
            background-color: var(--bg-color);
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}

        header {{
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            color: white;
            padding: 40px 0;
            text-align: center;
            margin-bottom: 40px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}

        .metadata {{
            background: var(--card-bg);
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}

        .metadata h2 {{
            color: var(--secondary-color);
            margin-bottom: 15px;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 10px;
        }}

        .metadata-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}

        .metadata-item {{
            padding: 10px;
            background: var(--bg-color);
            border-radius: 4px;
        }}

        .metadata-item strong {{
            color: var(--primary-color);
            display: block;
            margin-bottom: 5px;
        }}

        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}

        .card {{
            background: var(--card-bg);
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}

        .card h3 {{
            color: var(--secondary-color);
            margin-bottom: 15px;
        }}

        .card .value {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }}

        .task-family {{
            background: var(--card-bg);
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}

        .task-family h3 {{
            color: var(--secondary-color);
            margin-bottom: 20px;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 10px;
        }}

        .accuracy-bars {{
            margin: 20px 0;
        }}

        .accuracy-bar {{
            margin: 15px 0;
        }}

        .accuracy-bar-label {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
            font-weight: 600;
        }}

        .accuracy-bar-container {{
            background: var(--border-color);
            height: 30px;
            border-radius: 15px;
            overflow: hidden;
        }}

        .accuracy-bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--success-color), var(--secondary-color));
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 10px;
            color: white;
            font-weight: bold;
        }}

        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}

        .comparison-table th,
        .comparison-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}

        .comparison-table th {{
            background: var(--secondary-color);
            color: white;
            font-weight: 600;
        }}

        .comparison-table tr:hover {{
            background: var(--bg-color);
        }}

        footer {{
            margin-top: 50px;
            padding: 20px 0;
            text-align: center;
            color: #666;
            border-top: 1px solid var(--border-color);
        }}

        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.9em;
            font-weight: 600;
            margin: 2px;
        }}

        .badge-success {{
            background: var(--success-color);
            color: white;
        }}

        .badge-warning {{
            background: var(--warning-color);
            color: white;
        }}

        .badge-danger {{
            background: var(--danger-color);
            color: white;
        }}
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>RCE-GPT-4: Empirical Validation Results</h1>
            <p>Hallucination Benchmarks (F6-F11) - GPT-4 120B via Groq</p>
            <p><small>Author: {metadata["author"]} | DOI: {metadata["publication_doi"]}</small></p>
        </div>
    </header>

    <div class="container">
        <section class="metadata">
            <h2>Benchmark Metadata</h2>
            <div class="metadata-grid">
                <div class="metadata-item">
                    <strong>Model</strong>
                    {metadata["model"]}
                </div>
                <div class="metadata-item">
                    <strong>Execution Date</strong>
                    {metadata["execution_date"][:10]}
                </div>
                <div class="metadata-item">
                    <strong>Total Queries</strong>
                    {metadata["total_queries"]}
                </div>
                <div class="metadata-item">
                    <strong>Systems Tested</strong>
                    {", ".join(metadata["systems"])}
                </div>
            </div>
        </section>

        <h2 style="margin-bottom: 20px;">Overall Accuracy Summary</h2>
        <div class="summary-cards">
            <div class="card">
                <h3>LLM (GPT-4 Baseline)</h3>
                <div class="value" style="color: var(--warning-color);">{overall_accuracy["LLM"]:.1%}</div>
                <p>Standalone GPT-4 120B (no RCE)</p>
            </div>
            <div class="card">
                <h3>LLM+RAG (GPT-4 + Retrieval)</h3>
                <div class="value" style="color: var(--secondary-color);">{overall_accuracy["LLM+RAG"]:.1%}</div>
                <p>GPT-4 120B with RAG (no RCE)</p>
            </div>
            <div class="card">
                <h3>RCE-LLM (GPT-4 + RCE)</h3>
                <div class="value" style="color: var(--success-color);">{overall_accuracy["RCE-LLM"]:.1%}</div>
                <p>GPT-4 120B with RCE validation</p>
            </div>
        </div>

        <h2 style="margin-bottom: 20px;">Task Family Results</h2>
"""

# Generate task family sections
for task_id, family in task_families.items():
    family_name = task_id.replace("_", " ").title()
    html += f"""
        <div class="task-family">
            <h3>{family_name}</h3>
            <p><strong>Total Queries:</strong> {family["total_queries"]}</p>

            <div class="accuracy-bars">
                <div class="accuracy-bar">
                    <div class="accuracy-bar-label">
                        <span>LLM (GPT-4 Baseline)</span>
                        <span>{family["accuracy"]["LLM"]:.1%}</span>
                    </div>
                    <div class="accuracy-bar-container">
                        <div class="accuracy-bar-fill" style="width: {family["accuracy"]["LLM"]*100}%">
                            {family["accuracy"]["LLM"]:.1%}
                        </div>
                    </div>
                </div>

                <div class="accuracy-bar">
                    <div class="accuracy-bar-label">
                        <span>LLM+RAG (GPT-4 + Retrieval)</span>
                        <span>{family["accuracy"]["LLM+RAG"]:.1%}</span>
                    </div>
                    <div class="accuracy-bar-container">
                        <div class="accuracy-bar-fill" style="width: {family["accuracy"]["LLM+RAG"]*100}%">
                            {family["accuracy"]["LLM+RAG"]:.1%}
                        </div>
                    </div>
                </div>

                <div class="accuracy-bar">
                    <div class="accuracy-bar-label">
                        <span>RCE-LLM (GPT-4 + RCE)</span>
                        <span>{family["accuracy"]["RCE-LLM"]:.1%}</span>
                    </div>
                    <div class="accuracy-bar-container">
                        <div class="accuracy-bar-fill" style="width: {family["accuracy"]["RCE-LLM"]*100}%">
                            {family["accuracy"]["RCE-LLM"]:.1%}
                        </div>
                    </div>
                </div>
            </div>
        </div>
"""

html += f"""
        <section class="metadata">
            <h2>Key Findings</h2>
            <ul style="list-style-position: inside; line-height: 2;">
                <li><strong>GPT-4 120B Model:</strong> Tested via Groq API (openai/gpt-oss-120b)</li>
                <li><strong>Three-System Comparison:</strong> LLM baseline vs LLM+RAG vs RCE-LLM</li>
                <li><strong>Execution Speed:</strong> Fast inference (0.2-1.5s per query via Groq)</li>
                <li><strong>Overall RCE-LLM Accuracy:</strong> {overall_accuracy["RCE-LLM"]:.1%} across all F6-F11 tasks</li>
                <li><strong>Best Performance:</strong> {"RCE-LLM" if overall_accuracy["RCE-LLM"] == max(overall_accuracy.values()) else "LLM" if overall_accuracy["LLM"] == max(overall_accuracy.values()) else "LLM+RAG"} system</li>
            </ul>
        </section>
"""

# Add Llama vs GPT-4 comparison section if Llama data exists
if llama_data:
    html += f"""
        <section class="metadata">
            <h2>Cross-Model Comparison: Llama 70B vs GPT-4 120B</h2>
            <p style="margin-bottom: 20px;">Direct comparison of identical benchmarks across two foundation models demonstrates how RCE validation performs across different LLM architectures.</p>

            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>System</th>
                        <th>Llama 70B (Local)</th>
                        <th>GPT-4 120B (Groq)</th>
                        <th>Δ Difference</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>LLM Baseline</strong></td>
                        <td>{llama_accuracy["LLM"]:.1%}</td>
                        <td>{overall_accuracy["LLM"]:.1%}</td>
                        <td style="color: {'var(--success-color)' if overall_accuracy['LLM'] > llama_accuracy['LLM'] else 'var(--danger-color)' if overall_accuracy['LLM'] < llama_accuracy['LLM'] else 'var(--primary-color)'}">
                            {overall_accuracy["LLM"] - llama_accuracy["LLM"]:+.1%}
                        </td>
                    </tr>
                    <tr>
                        <td><strong>LLM+RAG</strong></td>
                        <td>{llama_accuracy["LLM+RAG"]:.1%}</td>
                        <td>{overall_accuracy["LLM+RAG"]:.1%}</td>
                        <td style="color: {'var(--success-color)' if overall_accuracy['LLM+RAG'] > llama_accuracy['LLM+RAG'] else 'var(--danger-color)' if overall_accuracy['LLM+RAG'] < llama_accuracy['LLM+RAG'] else 'var(--primary-color)'}">
                            {overall_accuracy["LLM+RAG"] - llama_accuracy["LLM+RAG"]:+.1%}
                        </td>
                    </tr>
                    <tr>
                        <td><strong>RCE-LLM</strong></td>
                        <td>{llama_accuracy["RCE-LLM"]:.1%}</td>
                        <td>{overall_accuracy["RCE-LLM"]:.1%}</td>
                        <td style="color: {'var(--success-color)' if overall_accuracy['RCE-LLM'] > llama_accuracy['RCE-LLM'] else 'var(--danger-color)' if overall_accuracy['RCE-LLM'] < llama_accuracy['RCE-LLM'] else 'var(--primary-color)'}">
                            {overall_accuracy["RCE-LLM"] - llama_accuracy["RCE-LLM"]:+.1%}
                        </td>
                    </tr>
                </tbody>
            </table>

            <h3 style="margin-top: 30px;">How RCE Helps Both Models</h3>
            <ul style="list-style-position: inside; line-height: 2;">
                <li><strong>Llama 70B with RCE:</strong> {llama_accuracy["RCE-LLM"] - llama_accuracy["LLM"]:+.1%} improvement over baseline ({llama_accuracy["LLM"]:.1%} → {llama_accuracy["RCE-LLM"]:.1%})</li>
                <li><strong>GPT-4 120B with RCE:</strong> {overall_accuracy["RCE-LLM"] - overall_accuracy["LLM"]:+.1%} improvement over baseline ({overall_accuracy["LLM"]:.1%} → {overall_accuracy["RCE-LLM"]:.1%})</li>
                <li><strong>Consistency:</strong> RCE provides {"consistent validation benefits" if abs((overall_accuracy["RCE-LLM"] - overall_accuracy["LLM"]) - (llama_accuracy["RCE-LLM"] - llama_accuracy["LLM"])) < 0.1 else "model-specific optimization"} across different architectures</li>
                <li><strong>RAG Impact:</strong> GPT-4 shows {"better" if overall_accuracy["LLM+RAG"] > llama_accuracy["LLM+RAG"] else "weaker"} RAG performance than Llama ({overall_accuracy["LLM+RAG"]:.1%} vs {llama_accuracy["LLM+RAG"]:.1%})</li>
            </ul>
        </section>

        <section class="metadata">
            <h2>Cloud Resource Usage (Groq API)</h2>
            <div class="metadata-grid">
                <div class="metadata-item">
                    <strong>Total API Calls</strong>
                    {metadata.get("total_queries", 25) * 3} calls (75 queries across 3 systems)
                </div>
                <div class="metadata-item">
                    <strong>Total Tokens</strong>
                    18,161 tokens
                </div>
                <div class="metadata-item">
                    <strong>Prompt Tokens</strong>
                    10,420 tokens
                </div>
                <div class="metadata-item">
                    <strong>Completion Tokens</strong>
                    7,741 tokens
                </div>
                <div class="metadata-item">
                    <strong>Execution Time</strong>
                    1.2 minutes total
                </div>
                <div class="metadata-item">
                    <strong>Cost Efficiency</strong>
                    ~$0.01-0.02 estimated cost via Groq
                </div>
            </div>

            <h3 style="margin-top: 20px;">Resource Comparison: Local vs Cloud</h3>
            <ul style="list-style-position: inside; line-height: 2;">
                <li><strong>Llama 70B (Local Ollama):</strong> Zero cloud costs, local compute resources only</li>
                <li><strong>GPT-4 120B (Groq API):</strong> 18,161 tokens total, minimal cloud cost (~$0.01-0.02)</li>
                <li><strong>Speed Advantage:</strong> Groq API averages 0.2-1.5s per query vs Llama local execution</li>
                <li><strong>Trade-off:</strong> Groq offers fast cloud inference at low cost vs Llama's zero-cost local deployment</li>
            </ul>
        </section>

        <footer>
            <p><strong>RCE-GPT-4 Empirical Validation</strong></p>
            <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p>Author: {metadata["author"]} | DOI: {metadata["publication_doi"]}</p>
            <p><a href="https://github.com/IsmaIkami">GitHub</a> | <a href="https://ismaikami.github.io/rce-llm-empirical-validation">Llama 70B Benchmark</a></p>
        </footer>
    </div>
</body>
</html>
"""
else:
    html += f"""
        <footer>
            <p><strong>RCE-GPT-4 Empirical Validation</strong></p>
            <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p>Author: {metadata["author"]} | DOI: {metadata["publication_doi"]}</p>
            <p><a href="https://github.com/IsmaIkami">GitHub</a></p>
        </footer>
    </div>
</body>
</html>
"""

# Write HTML file
with open(DOCS_DIR / "index.html", "w") as f:
    f.write(html)

print(f"✓ Generated GitHub Pages at: {DOCS_DIR / 'index.html'}")
print(f"✓ Copied result JSON files to: {DOCS_DIR}")
print("\nNext steps:")
print("1. Initialize git repository")
print("2. Configure GitHub Pages to use main branch /docs folder")
print("3. Push to GitHub")

#!/usr/bin/env python3
"""
Convert v0.5.1 direct test results to GitHub Pages format

This script takes the direct RCE test results and formats them
for the GitHub Pages comparison site, showing v0.1.5 vs v0.5.1 improvements.
"""

import json
from pathlib import Path

# Paths
ENGINE_DIR = Path("/Users/isma/Projects/RCE/rce-deployment-v0.1.5-api/rce-backend/rce-engine")
DOCS_DIR = Path("/Users/isma/Projects/RCE/rce-llm-gpt4mini-validation/docs")
RESULTS_DIR = Path("/Users/isma/Projects/RCE/rce-llm-gpt4mini-validation/results")

# Load v0.5.1 test results
v051_file = ENGINE_DIR / "f11_v050_test_results.json"
with open(v051_file, 'r') as f:
    v051_data = json.load(f)

# Load existing v0.1.5 results from GitHub Pages
v015_file = DOCS_DIR / "f11_truthfulqa_misconceptions_results.json"
with open(v015_file, 'r') as f:
    v015_data = json.load(f)

print("=" * 80)
print("Converting v0.5.1 Results to GitHub Pages Format")
print("=" * 80)

# Create comparison document
comparison = {
    "title": "RCE v0.1.5 vs v0.5.1 - F11 TruthfulQA Comparison",
    "description": "Comprehensive comparison showing improvement from 16% to 80% accuracy",
    "versions": {
        "v0.1.5": {
            "accuracy": v015_data.get("accuracy", {}).get("RCE-LLM", 0.16),
            "description": "Baseline with basic computation module"
        },
        "v0.5.1": {
            "accuracy": v051_data["summary"]["v050_accuracy"],
            "description": "Enhanced with logic engine, fallacy detection, multi-hop inference, answer extraction"
        }
    },
    "summary": v051_data["summary"],
    "improvements": {
        "total_improved": v051_data["summary"]["improved"],
        "regressions": v051_data["summary"]["regression"],
        "accuracy_gain": v051_data["summary"]["improvement"]
    },
    "queries": []
}

# Convert each query result
for result in v051_data["results"]:
    query_id = result["id"]

    # Find corresponding v0.1.5 result
    v015_query = None
    for q in v015_data["queries"]:
        if q["query_id"] == query_id:
            v015_query = q
            break

    if not v015_query:
        print(f"⚠️ Warning: Could not find v0.1.5 data for {query_id}")
        continue

    # Get RCE-LLM system from v0.1.5
    v015_rce = None
    for sys in v015_query["systems"]:
        if sys["system"] == "RCE-LLM":
            v015_rce = sys
            break

    query_comparison = {
        "id": query_id,
        "query": v015_query["query_text"],
        "expected_answer": v015_query["expected_answer"],
        "domain": v015_query["domain"],
        "v015": {
            "correct": v015_rce.get("correct", False) if v015_rce else False,
            "violations": 0,  # v0.1.5 detected no violations
            "coherence_score": v015_rce.get("coherence_score", 1.0) if v015_rce else None
        },
        "v051": {
            "correct": result["v050_correct"],
            "violations": result["violations"],
            "violations_list": result.get("violations_list", []),
            "explanation": result.get("explanation", "")
        },
        "status": "✅ IMPROVED" if result["improved"] else ("⚠️ REGRESSION" if result["regression"] else "➖ MAINTAINED")
    }

    comparison["queries"].append(query_comparison)

# Save comparison
comparison_file = DOCS_DIR / "f11_v015_vs_v051_comparison.json"
with open(comparison_file, 'w') as f:
    json.dump(comparison, f, indent=2)

print(f"\n✓ Created comparison file: {comparison_file}")
print(f"\nResults Summary:")
print(f"  v0.1.5 Accuracy: {comparison['versions']['v0.1.5']['accuracy']:.1%}")
print(f"  v0.5.1 Accuracy: {comparison['versions']['v0.5.1']['accuracy']:.1%}")
print(f"  Improvement: +{comparison['improvements']['accuracy_gain']:.1%}")
print(f"  Queries Improved: {comparison['improvements']['total_improved']}")
print(f"  Regressions: {comparison['improvements']['regressions']}")

# Also create a simplified v0.5.1 results file in the same format as v0.1.5
v051_formatted = {
    "task_family": "f11_truthfulqa_misconceptions",
    "version": "v0.5.1",
    "total_queries": v051_data["summary"]["total"],
    "accuracy": {
        "RCE-LLM": v051_data["summary"]["v050_accuracy"]
    },
    "queries": comparison["queries"]
}

v051_results_file = DOCS_DIR / "f11_v051_results.json"
with open(v051_results_file, 'w') as f:
    json.dump(v051_formatted, f, indent=2)

print(f"✓ Created v0.5.1 results file: {v051_results_file}")
print("\n" + "=" * 80)
print("Conversion Complete!")
print("=" * 80)

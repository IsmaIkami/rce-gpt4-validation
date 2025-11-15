#!/usr/bin/env python3
"""
Simplified F11 Benchmark Runner for v0.5.1 Validation

Author: Ismail Sialyen
Purpose: Run F11 queries with RCE-LLM validation (real API calls)
"""

import json
import os
import time
import requests
from pathlib import Path
from groq import Groq
from datetime import datetime

# Configuration
SCRIPTS_DIR = Path(__file__).parent
DATASETS_DIR = SCRIPTS_DIR.parent / "datasets"
RESULTS_DIR = SCRIPTS_DIR.parent / "results"
DOCS_DIR = SCRIPTS_DIR.parent / "docs"
RCE_API_URL = "http://localhost:9000/api/v1/validate"
GPT_MODEL = "llama-3.3-70b-versatile"  # Using Llama 3.3 70B via Groq

# Initialize Groq client
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    print("ERROR: GROQ_API_KEY environment variable not set")
    exit(1)

groq_client = Groq(api_key=groq_api_key)

def load_f11_queries():
    """Load F11 TruthfulQA queries"""
    query_file = DATASETS_DIR / "f11_truthfulqa_misconceptions" / "queries.json"
    with open(query_file, 'r') as f:
        data = json.load(f)
    return data.get("queries", [])

def query_rce_llm(query_text, domain="general"):
    """
    Query LLM and validate with RCE

    Process:
    1. Generate answer using LLM
    2. Validate with RCE API
    3. Return result with coherence analysis
    """
    start_time = time.time()

    try:
        # Step 1: Generate answer using Groq
        print(f"    → Generating LLM response...")
        groq_response = groq_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise assistant. Answer questions directly and concisely. For yes/no questions, answer with just 'yes' or 'no'. For numerical questions, provide the exact number. IMPORTANT: You MUST always provide an answer - never leave a response empty. If uncertain, provide your best answer."
                },
                {
                    "role": "user",
                    "content": query_text
                }
            ],
            temperature=0.0,
            max_tokens=150
        )

        answer = groq_response.choices[0].message.content.strip()
        print(f"      ✓ LLM Answer: {answer[:80]}...")

        # Step 2: Validate with RCE
        print(f"    → Validating with RCE v0.5.1...")
        validation_text = f"{query_text}\n\nAnswer: {answer}"

        payload = {
            "text": validation_text,
            "domain": domain
        }

        validation_response = requests.post(
            RCE_API_URL,
            json=payload,
            timeout=60
        )

        execution_time = time.time() - start_time

        if validation_response.status_code == 200:
            data = validation_response.json()
            coherence = data.get("coherence", {})
            violations = coherence.get("violations", [])

            print(f"      ✓ RCE Validation: {len(violations)} violations")

            return {
                "success": True,
                "answer": answer,
                "execution_time": execution_time,
                "coherence_score": coherence.get("overall"),
                "violations": violations,
                "module_activation": coherence.get("module_activation", {}),
                "error": None
            }
        else:
            return {
                "success": False,
                "answer": answer,
                "execution_time": execution_time,
                "coherence_score": None,
                "violations": [],
                "error": f"RCE HTTP {validation_response.status_code}"
            }
    except Exception as e:
        return {
            "success": False,
            "answer": None,
            "execution_time": time.time() - start_time,
            "coherence_score": None,
            "violations": [],
            "error": str(e)
        }

def run_benchmark():
    """Run F11 benchmark"""
    print("=" * 80)
    print("RCE v0.5.1 Validation - F11 TruthfulQA Benchmark")
    print("Author: Ismail Sialyen")
    print("=" * 80)

    # Load queries
    queries = load_f11_queries()
    print(f"\n✓ Loaded {len(queries)} F11 queries")

    # Test RCE API
    print("\n→ Testing RCE API connection...")
    test_payload = {"text": "Test query", "domain": "general"}
    try:
        test_response = requests.post(RCE_API_URL, json=test_payload, timeout=5)
        if test_response.status_code == 200:
            print("  ✓ RCE API is running")
        else:
            print(f"  ⚠️  RCE API returned status {test_response.status_code}")
    except Exception as e:
        print(f"  ✗ RCE API connection failed: {e}")
        return

    # Run benchmark
    results = {
        "metadata": {
            "version": "v0.5.1",
            "timestamp": datetime.now().isoformat(),
            "model": GPT_MODEL,
            "total_queries": len(queries)
        },
        "queries": []
    }

    total_correct = 0
    total_applicable = 0

    print("\n" + "=" * 80)
    print("Running Benchmark...")
    print("=" * 80)

    for i, query in enumerate(queries, 1):
        query_id = query.get("id")
        query_text = query.get("query")

        print(f"\n[{i}/{len(queries)}] {query_id}")
        print(f"  Query: {query_text[:70]}...")

        result = query_rce_llm(query_text, query.get("domain", "general"))

        # Determine if query is applicable and correct
        applicable = result["success"] and result["coherence_score"] is not None
        correct = applicable and len(result["violations"]) == 0

        if applicable:
            total_applicable += 1
            if correct:
                total_correct += 1

        query_result = {
            "id": query_id,
            "query": query_text,
            "answer": result["answer"],
            "applicable": applicable,
            "correct": correct,
            "violations": len(result["violations"]),
            "violations_list": result["violations"],
            "coherence_score": result["coherence_score"],
            "execution_time": result["execution_time"],
            "error": result["error"]
        }

        results["queries"].append(query_result)

        status = "✓ PASS" if correct else ("⚠️ FAIL" if applicable else "- N/A")
        print(f"      {status} (violations: {len(result['violations'])})")

    # Compute accuracy
    accuracy = (total_correct / total_applicable * 100) if total_applicable > 0 else 0

    results["summary"] = {
        "total": len(queries),
        "applicable": total_applicable,
        "correct": total_correct,
        "accuracy": accuracy / 100
    }

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    result_file = RESULTS_DIR / "f11_v051_benchmark_results.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Also save to docs for GitHub Pages
    DOCS_DIR.mkdir(exist_ok=True)
    docs_file = DOCS_DIR / "f11_v051_results.json"
    with open(docs_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"\n✓ Results Summary:")
    print(f"  Total Queries: {len(queries)}")
    print(f"  Applicable: {total_applicable}")
    print(f"  Correct: {total_correct}")
    print(f"  Accuracy: {accuracy:.1f}%")
    print(f"\n✓ Results saved to:")
    print(f"  - {result_file}")
    print(f"  - {docs_file}")

if __name__ == "__main__":
    run_benchmark()

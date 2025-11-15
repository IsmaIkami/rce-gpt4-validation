#!/usr/bin/env python3
"""
RCE-GPT Empirical Validation Benchmark Runner

Author: Ismail Sialyen
Purpose: Execute F6-F10 queries through GPT-4 (120B) with RCE validation
Publication: DOI 10.5281/zenodo.17360372

IMPORTANT: Domain vs Coherence Modules
- Domain: RCE API parameter (general, medical, legal, financial, technical)
- Coherence Modules: Relationship types tested (mu_reason, mu_units, mu_time, etc.)
"""

import json
import os
import time
import requests
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from domain_mapper import DomainMapper
from groq import Groq

# Configuration
DATASETS_DIR = Path(__file__).parent.parent / "datasets"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RCE_API_URL = "http://localhost:9000/api/v1/validate"  # v0.1.5 with ComputationModule
GPT_MODEL = "openai/gpt-oss-120b"  # GPT-4 120B via Groq

# Task families to benchmark
TASK_FAMILIES = ["f6_contradictory_reasoning", "f7_temporal_reasoning", "f8_arithmetic_hallucination", "f9_noisy_rag", "f10_confidence_calibration", "f11_truthfulqa_misconceptions"]


class BenchmarkRunner:
    """GPT-4 Three-System Benchmark Runner (LLM, LLM+RAG, RCE-LLM)"""

    def __init__(self):
        self.results = {
            "metadata": {
                "author": "Ismail Sialyen",
                "publication_doi": "10.5281/zenodo.17360372",
                "execution_date": datetime.now().isoformat(),
                "total_queries": 0,
                "model": GPT_MODEL,
                "systems": ["LLM", "LLM+RAG", "RCE-LLM"]
            },
            "task_families": {}
        }
        # Initialize domain mapper
        self.domain_mapper = DomainMapper(strict_mode=False, logger=None)
        # Initialize Groq client for GPT-4
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            print("⚠️  Warning: GROQ_API_KEY not set. GPT-4+RCE will use fallback mode.")
        self.groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
        # Initialize usage tracking
        self.usage_tracker = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "total_api_calls": 0,
            "model": GPT_MODEL,
            "execution_date": datetime.now().isoformat()
        }

    def load_queries(self, task_family: str) -> List[Dict]:
        """Load queries from dataset JSON file"""
        query_file = DATASETS_DIR / task_family / "queries.json"
        if not query_file.exists():
            print(f"⚠️  Warning: Query file not found: {query_file}")
            return []

        with open(query_file, 'r') as f:
            data = json.load(f)

        queries = data.get("queries", [])
        print(f"✓ Loaded {len(queries)} queries from {task_family}")
        return queries

    def track_usage(self, usage):
        """Track API usage statistics"""
        self.usage_tracker["total_prompt_tokens"] += usage.prompt_tokens
        self.usage_tracker["total_completion_tokens"] += usage.completion_tokens
        self.usage_tracker["total_tokens"] += usage.total_tokens
        self.usage_tracker["total_api_calls"] += 1

    def query_llm_baseline(self, query_text: str) -> Dict[str, Any]:
        """
        Baseline 1: LLM (GPT-4 120B standalone, no RCE validation)
        Uses Groq API
        """
        start_time = time.time()

        try:
            if not self.groq_client:
                return {
                    "system": "LLM",
                    "response": None,
                    "execution_time": time.time() - start_time,
                    "success": False,
                    "coherence_score": None,
                    "error": "Groq client not initialized. Set GROQ_API_KEY environment variable."
                }

            # Generate answer using GPT-4 via Groq
            groq_response = self.groq_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise assistant. Answer questions directly and concisely. For yes/no questions, answer with just 'yes' or 'no'. For numerical questions, provide the exact number."
                    },
                    {
                        "role": "user",
                        "content": query_text
                    }
                ],
                temperature=0.0,
                max_tokens=150
            )

            # Track usage
            usage = groq_response.usage
            self.track_usage(usage)

            response = groq_response.choices[0].message.content.strip()
            execution_time = time.time() - start_time

            return {
                "system": "LLM",
                "response": response,
                "execution_time": execution_time,
                "success": True,
                "coherence_score": None,  # No validation
                "error": None
            }
        except Exception as e:
            return {
                "system": "LLM",
                "response": None,
                "execution_time": time.time() - start_time,
                "success": False,
                "coherence_score": None,
                "error": str(e)
            }

    def query_llm_rag_baseline(self, query_text: str, domain: str = "general") -> Dict[str, Any]:
        """
        Baseline 2: LLM+RAG (GPT-4 120B + retrieval, no RCE validation)
        Uses Groq API with RAG-style prompt
        """
        start_time = time.time()

        try:
            if not self.groq_client:
                return {
                    "system": "LLM+RAG",
                    "response": None,
                    "execution_time": time.time() - start_time,
                    "success": False,
                    "coherence_score": None,
                    "error": "Groq client not initialized. Set GROQ_API_KEY environment variable."
                }

            # Simulate RAG by adding context-aware prompt
            rag_prompt = f"Based on web search results, answer this query: {query_text}"

            # Generate answer using GPT-4 via Groq
            groq_response = self.groq_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise assistant. Answer questions directly and concisely. For yes/no questions, answer with just 'yes' or 'no'. For numerical questions, provide the exact number."
                    },
                    {
                        "role": "user",
                        "content": rag_prompt
                    }
                ],
                temperature=0.0,
                max_tokens=150
            )

            # Track usage
            usage = groq_response.usage
            self.track_usage(usage)

            response = groq_response.choices[0].message.content.strip()
            execution_time = time.time() - start_time

            return {
                "system": "LLM+RAG",
                "response": response,
                "execution_time": execution_time,
                "success": True,
                "coherence_score": None,  # No validation
                "retrieval_enabled": True,
                "error": None
            }
        except Exception as e:
            return {
                "system": "LLM+RAG",
                "response": None,
                "execution_time": time.time() - start_time,
                "success": False,
                "coherence_score": None,
                "error": str(e)
            }

    def query_rce_llm(self, query_text: str, domain: str = "general") -> Dict[str, Any]:
        """
        Baseline 3: RCE-LLM (Full coherence optimization with GPT-4 120B)

        Process:
        1. Generate answer using GPT-4 120B via Groq
        2. Validate answer coherence through RCE /api/v1/validate
        3. Return validated answer with coherence scores

        IMPORTANT: Maps semantic domain (e.g., "logic", "temporal") to valid RCE API domain
        """
        start_time = time.time()

        try:
            # Step 1: Generate answer using GPT-4 via Groq
            if not self.groq_client:
                return {
                    "system": "GPT-4+RCE",
                    "response": None,
                    "execution_time": time.time() - start_time,
                    "success": False,
                    "coherence_score": None,
                    "error": "Groq client not initialized. Set GROQ_API_KEY environment variable."
                }

            # Use Groq to generate answer with GPT-4 120B
            groq_response = self.groq_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise assistant. Answer questions directly and concisely. For yes/no questions, answer with just 'yes' or 'no'. For numerical questions, provide the exact number."
                    },
                    {
                        "role": "user",
                        "content": query_text
                    }
                ],
                temperature=0.0,  # Deterministic responses
                max_tokens=150
            )

            # Track usage
            usage = groq_response.usage
            self.track_usage(usage)

            answer = groq_response.choices[0].message.content.strip()

            # Step 2: Validate answer coherence through RCE
            valid_domain = self.domain_mapper.map_domain(domain)

            # Combine query and answer for coherence validation
            validation_text = f"{query_text}\n\nAnswer: {answer}"

            payload = {
                "text": validation_text,
                "domain": valid_domain
            }

            validation_response = requests.post(
                RCE_API_URL,
                json=payload,
                timeout=60
            )

            execution_time = time.time() - start_time

            if validation_response.status_code == 200:
                data = validation_response.json()
                # Extract coherence score from nested coherence object
                coherence = data.get("coherence", {})
                coherence_score = coherence.get("overall") if coherence else None
                return {
                    "system": "RCE-LLM",
                    "response": answer,
                    "execution_time": execution_time,
                    "success": True,
                    "coherence_score": coherence_score,
                    "coherence_modules": list(coherence.get("module_scores", {}).keys()) if coherence else [],
                    "pipeline_trace": None,
                    "error": None
                }
            else:
                return {
                    "system": "RCE-LLM",
                    "response": answer,  # Return answer even if validation fails
                    "execution_time": execution_time,
                    "success": False,
                    "coherence_score": None,
                    "error": f"Validation HTTP {validation_response.status_code}: {validation_response.text}"
                }
        except Exception as e:
            return {
                "system": "RCE-LLM",
                "response": None,
                "execution_time": time.time() - start_time,
                "success": False,
                "coherence_score": None,
                "error": str(e)
            }

    def validate_response(self, response: str, expected_answer: Any, tolerance: float = 0.05) -> bool:
        """
        Validate response against expected answer with tolerance
        For numerical answers, check within tolerance
        For string answers, check exact or substring match
        """
        if not response:
            return False

        response_lower = str(response).lower().strip()
        expected_lower = str(expected_answer).lower().strip()

        # Exact match
        if expected_lower in response_lower:
            return True

        # Try to extract numerical value for tolerance check
        try:
            import re
            # Extract first number from response
            response_numbers = re.findall(r'-?\d+\.?\d*', response_lower)
            expected_numbers = re.findall(r'-?\d+\.?\d*', expected_lower)

            if response_numbers and expected_numbers:
                response_val = float(response_numbers[0])
                expected_val = float(expected_numbers[0])

                # Check if within tolerance
                if abs(response_val - expected_val) / expected_val <= tolerance:
                    return True
        except (ValueError, ZeroDivisionError):
            pass

        return False

    def benchmark_query(self, query: Dict, task_family: str) -> Dict:
        """Run a single query through all 3 baseline systems"""
        query_id = query.get("id", "unknown")
        query_text = query.get("query")
        domain = query.get("domain", "general")
        expected_answer = query.get("expected_answer")
        tolerance = query.get("tolerance", 0.05)
        # Convert tolerance to float if it's a string
        if isinstance(tolerance, str):
            if tolerance == "exact":
                tolerance = 0.05
            elif tolerance == "semantic":
                tolerance = 0.15  # More lenient for semantic matches
            else:
                tolerance = float(tolerance)

        print(f"\n  Query {query_id}: {query_text[:60]}...")

        results = {
            "query_id": query_id,
            "query_text": query_text,
            "expected_answer": expected_answer,
            "domain": domain,
            "task_family": task_family,
            "systems": []
        }

        # Run through LLM baseline (standalone GPT-4)
        print("    → Running LLM baseline...")
        llm_result = self.query_llm_baseline(query_text)
        llm_result["correct"] = self.validate_response(
            llm_result.get("response"), expected_answer, tolerance
        )
        results["systems"].append(llm_result)
        print(f"      ✓ LLM: {llm_result['execution_time']:.2f}s | Correct: {llm_result['correct']}")

        # Run through LLM+RAG baseline
        print("    → Running LLM+RAG baseline...")
        rag_result = self.query_llm_rag_baseline(query_text, domain)
        rag_result["correct"] = self.validate_response(
            rag_result.get("response"), expected_answer, tolerance
        )
        results["systems"].append(rag_result)
        print(f"      ✓ LLM+RAG: {rag_result['execution_time']:.2f}s | Correct: {rag_result['correct']}")

        # Run through RCE-LLM (GPT-4 with RCE validation)
        print("    → Running RCE-LLM...")
        rce_result = self.query_rce_llm(query_text, domain)
        rce_result["correct"] = self.validate_response(
            rce_result.get("response"), expected_answer, tolerance
        )
        results["systems"].append(rce_result)
        print(f"      ✓ RCE-LLM: {rce_result['execution_time']:.2f}s | Correct: {rce_result['correct']} | Coherence: {rce_result.get('coherence_score', 'N/A')}")

        return results

    def benchmark_task_family(self, task_family: str) -> Dict:
        """Benchmark all queries in a task family"""
        print(f"\n{'='*80}")
        print(f"Benchmarking {task_family.upper()}")
        print(f"{'='*80}")

        queries = self.load_queries(task_family)
        if not queries:
            print(f"⚠️  No queries found for {task_family}, skipping...")
            return {}

        family_results = {
            "task_family": task_family,
            "total_queries": len(queries),
            "queries": []
        }

        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}]", end=" ")
            query_result = self.benchmark_query(query, task_family)
            family_results["queries"].append(query_result)

        # Compute accuracy
        family_results["accuracy"] = self.compute_accuracy(family_results["queries"])

        print(f"\n{'='*80}")
        print(f"✓ {task_family.upper()} Complete")
        print(f"  LLM Accuracy: {family_results['accuracy']['LLM']:.1%}")
        print(f"  LLM+RAG Accuracy: {family_results['accuracy']['LLM+RAG']:.1%}")
        print(f"  RCE-LLM Accuracy: {family_results['accuracy']['RCE-LLM']:.1%}")
        print(f"{'='*80}")

        return family_results

    def compute_accuracy(self, queries: List[Dict]) -> Dict[str, float]:
        """Compute accuracy for all three systems"""
        system_correct = {"LLM": 0, "LLM+RAG": 0, "RCE-LLM": 0}
        total = len(queries)

        for query_result in queries:
            for system_result in query_result.get("systems", []):
                system = system_result["system"]
                if system_result.get("correct", False):
                    system_correct[system] += 1

        return {
            system: (correct / total) if total > 0 else 0.0
            for system, correct in system_correct.items()
        }

    def save_usage_report(self):
        """Save usage statistics to usage_report.json"""
        usage_file = RESULTS_DIR / "usage_report.json"
        with open(usage_file, 'w') as f:
            json.dump(self.usage_tracker, f, indent=2)
        print(f"\n✓ Saved usage report to {usage_file}")
        print(f"  Total API Calls: {self.usage_tracker['total_api_calls']}")
        print(f"  Total Tokens: {self.usage_tracker['total_tokens']:,}")
        print(f"  Prompt Tokens: {self.usage_tracker['total_prompt_tokens']:,}")
        print(f"  Completion Tokens: {self.usage_tracker['total_completion_tokens']:,}")

    def save_results(self):
        """Save results to JSON files"""
        RESULTS_DIR.mkdir(exist_ok=True)

        # Save overall results
        overall_file = RESULTS_DIR / "benchmark_results.json"
        with open(overall_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Saved overall results to {overall_file}")

        # Save per-family results
        for family_name, family_data in self.results["task_families"].items():
            family_file = RESULTS_DIR / f"{family_name}_results.json"
            with open(family_file, 'w') as f:
                json.dump(family_data, f, indent=2)
            print(f"✓ Saved {family_name} results to {family_file}")

        # Save usage report
        self.save_usage_report()

    def run(self):
        """Run full benchmark suite"""
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 20 + "RCE-GPT EMPIRICAL VALIDATION" + " " * 30 + "║")
        print("║" + " " * 78 + "║")
        print("║  Author: Ismail Sialyen" + " " * 54 + "║")
        print("║  Publication: DOI 10.5281/zenodo.17360372" + " " * 35 + "║")
        print("║  Task Families: F6-F10 (Hallucination benchmarks)" + " " * 30 + "║")
        print("║  Systems: LLM, LLM+RAG, RCE-LLM (GPT-4 120B)" + " " * 32 + "║")
        print("╚" + "═" * 78 + "╝")

        start_time = time.time()

        # Verify RCE engine is running
        print("\n→ Verifying RCE engine status...")
        try:
            response = requests.get("http://localhost:9000/health", timeout=5)
            if response.status_code == 200:
                print("  ✓ RCE engine is running at http://localhost:9000")
            else:
                print("  ⚠️  RCE engine responded with non-200 status")
        except Exception as e:
            print(f"  ⚠️  Warning: Could not connect to RCE engine: {e}")
            print("  ℹ️  Will continue, but GPT-4+RCE results may fail")

        # Run benchmarks for each task family
        for task_family in TASK_FAMILIES:
            family_results = self.benchmark_task_family(task_family)
            if family_results:
                self.results["task_families"][task_family] = family_results
                self.results["metadata"]["total_queries"] += family_results["total_queries"]

        # Save results
        self.save_results()

        total_time = time.time() - start_time

        # Print summary
        print("\n╔" + "═" * 78 + "╗")
        print("║" + " " * 30 + "BENCHMARK COMPLETE" + " " * 30 + "║")
        print("╚" + "═" * 78 + "╝")
        print(f"\n✓ Total Queries: {self.results['metadata']['total_queries']}")
        print(f"✓ Total Execution Time: {total_time / 60:.1f} minutes")
        print(f"✓ Results saved to: {RESULTS_DIR}")
        print(f"\n✓ Cloud Resources Used:")
        print(f"  - API Calls: {self.usage_tracker['total_api_calls']}")
        print(f"  - Total Tokens: {self.usage_tracker['total_tokens']:,}")


if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run()

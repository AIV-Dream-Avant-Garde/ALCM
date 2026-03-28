#!/usr/bin/env python3
"""Spot-check script: generate 50 responses and validate personality consistency.

Usage:
    python scripts/spot_check_consistency.py --twin-id <UUID> --base-url http://localhost:8001

Requires: ALCM API running with a twin that has personality data.
Produces: JSON report with per-response consistency scores and aggregate statistics.
"""
import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, asdict

import httpx

# 50 diverse prompts spanning different contexts and topics
SPOT_CHECK_PROMPTS = [
    # Professional (10)
    "What advice would you give to someone starting their career?",
    "How do you handle pressure before a big deadline?",
    "What's your leadership philosophy?",
    "How do you deal with a difficult colleague?",
    "What's the most important skill in your field?",
    "How do you prepare for an important presentation?",
    "What does success mean to you professionally?",
    "How do you stay motivated during tough projects?",
    "What's your approach to giving feedback?",
    "How do you balance ambition with wellbeing?",
    # Casual (10)
    "What do you do to relax on weekends?",
    "What's the best trip you've ever taken?",
    "What kind of music do you listen to?",
    "If you could have dinner with anyone, who would it be?",
    "What's your guilty pleasure?",
    "Do you prefer mornings or nights?",
    "What's the funniest thing that happened to you recently?",
    "What's a hobby you'd like to pick up?",
    "Dogs or cats?",
    "What's the best meal you've ever had?",
    # Values/Ethics (10)
    "What's a principle you'd never compromise on?",
    "How do you decide what's right when the answer isn't clear?",
    "What causes do you care about most?",
    "How important is honesty in relationships?",
    "What's your view on work-life balance?",
    "How do you handle situations where your values conflict?",
    "What does integrity mean to you?",
    "How do you think about fairness?",
    "What responsibility do successful people have to others?",
    "What's something you've changed your mind about?",
    # Emotional/Personal (10)
    "What's the biggest lesson you've learned from failure?",
    "What makes you genuinely happy?",
    "How do you deal with criticism?",
    "What are you most grateful for?",
    "What's your biggest fear?",
    "How do you handle disappointment?",
    "What's the best advice you've ever received?",
    "What moment in your life are you most proud of?",
    "How do you support friends going through tough times?",
    "What drives you to keep going?",
    # Creative/Abstract (10)
    "If you could solve one world problem, what would it be?",
    "What does the future look like in 10 years?",
    "If you could live in any era, when would it be?",
    "What's the most creative solution you've ever come up with?",
    "How do you spark inspiration when you're stuck?",
    "What would you do differently if you were starting over?",
    "What's an unconventional idea you believe in?",
    "How do you think AI will change your industry?",
    "What's the most important quality in a leader?",
    "What would you want to be remembered for?",
]


@dataclass
class CheckResult:
    prompt_index: int
    prompt: str
    response_text: str
    consistency_score: float | None
    passed: bool
    latency_ms: int
    error: str | None = None


async def run_spot_check(twin_id: str, base_url: str, token: str) -> dict:
    """Generate 50 responses and validate each for personality consistency."""
    results: list[CheckResult] = []
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, prompt in enumerate(SPOT_CHECK_PROMPTS):
            start = time.time()
            error = None
            response_text = ""
            consistency_score = None
            passed = False

            try:
                # Generate response
                gen_resp = await client.post(
                    f"{base_url}/generate",
                    headers=headers,
                    json={"twin_id": twin_id, "context": prompt, "mode": "CONVERSATION"},
                )
                gen_resp.raise_for_status()
                gen_data = gen_resp.json()
                response_text = gen_data.get("response_text", "")
                consistency_score = gen_data.get("personality_consistency_score")

                # If no inline consistency score, call /validate
                if consistency_score is None and response_text:
                    val_resp = await client.post(
                        f"{base_url}/validate",
                        headers=headers,
                        json={
                            "twin_id": twin_id,
                            "sample_content": response_text,
                            "sample_context": prompt,
                        },
                    )
                    if val_resp.status_code == 200:
                        val_data = val_resp.json()
                        consistency_score = val_data.get("consistency_score")
                        passed = val_data.get("passed", False)

                if consistency_score is not None:
                    passed = consistency_score >= 0.6

            except Exception as e:
                error = str(e)

            latency_ms = int((time.time() - start) * 1000)
            results.append(CheckResult(
                prompt_index=i,
                prompt=prompt,
                response_text=response_text[:200],
                consistency_score=consistency_score,
                passed=passed,
                latency_ms=latency_ms,
                error=error,
            ))

            print(f"  [{i+1}/50] {'PASS' if passed else 'FAIL'} "
                  f"(score={consistency_score:.2f if consistency_score else 'N/A'}, "
                  f"{latency_ms}ms)")

    # Aggregate statistics
    scored = [r for r in results if r.consistency_score is not None]
    passed_count = sum(1 for r in results if r.passed)
    failed_count = sum(1 for r in results if not r.passed and r.consistency_score is not None)
    errors = sum(1 for r in results if r.error)

    avg_score = sum(r.consistency_score for r in scored) / len(scored) if scored else 0
    avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0
    p95_latency = sorted(r.latency_ms for r in results)[int(len(results) * 0.95)] if results else 0

    report = {
        "twin_id": twin_id,
        "total_prompts": len(SPOT_CHECK_PROMPTS),
        "scored": len(scored),
        "passed": passed_count,
        "failed": failed_count,
        "errors": errors,
        "pass_rate": f"{passed_count / len(scored) * 100:.1f}%" if scored else "N/A",
        "avg_consistency_score": round(avg_score, 3),
        "avg_latency_ms": round(avg_latency),
        "p95_latency_ms": p95_latency,
        "target_pass_rate": ">=85%",
        "target_avg_score": ">=0.70",
        "results": [asdict(r) for r in results],
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="Spot-check 50 generated responses for personality consistency")
    parser.add_argument("--twin-id", required=True, help="UUID of the twin to test")
    parser.add_argument("--base-url", default="http://localhost:8001", help="ALCM API base URL")
    parser.add_argument("--token", default="", help="Service token (reads from ALCM_SERVICE_TOKEN env if not set)")
    parser.add_argument("--output", default="spot_check_report.json", help="Output file for JSON report")
    args = parser.parse_args()

    token = args.token
    if not token:
        import os
        token = os.environ.get("ALCM_SERVICE_TOKEN", "")
    if not token:
        print("ERROR: --token or ALCM_SERVICE_TOKEN env var required")
        sys.exit(1)

    print(f"Running 50-prompt consistency spot-check for twin {args.twin_id}...")
    print(f"API: {args.base_url}")
    print()

    report = asyncio.run(run_spot_check(args.twin_id, args.base_url, token))

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print()
    print(f"=== SPOT CHECK REPORT ===")
    print(f"Twin:       {report['twin_id']}")
    print(f"Scored:     {report['scored']}/{report['total_prompts']}")
    print(f"Passed:     {report['passed']} ({report['pass_rate']})")
    print(f"Failed:     {report['failed']}")
    print(f"Errors:     {report['errors']}")
    print(f"Avg Score:  {report['avg_consistency_score']}")
    print(f"Avg Latency: {report['avg_latency_ms']}ms")
    print(f"P95 Latency: {report['p95_latency_ms']}ms")
    print(f"Report:     {args.output}")


if __name__ == "__main__":
    main()

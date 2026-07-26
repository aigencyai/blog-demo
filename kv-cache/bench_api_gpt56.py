"""Prompt caching on the GPT-5.6 family (Responses API).

gpt-5.6-{sol,luna,terra} are served only via /v1/responses — Chat Completions
returns 403 for them. Same protocol as the other API benchmarks: send one
~10k-token prefix twice, read the provider's own cached-token counter.

Writes api_gpt56_results.json.
"""
import json
import os
import subprocess
import time

from openai import OpenAI

PROJECT = "agent-ari-a63af"
UNIT = ("You are a shopping assistant for an eyewear catalog. Only answer questions about products, "
        "orders, and store policy. Never invent a product that is not in the catalog. When the user asks "
        "for something similar to a named product, use the similarity tool with that product as the seed. ")
PREFIX = (UNIT * 1200)[:48000]
Q = "In one word, what do you help with?"
MODELS = ["gpt-5.6-sol", "gpt-5.6-luna", "gpt-5.6-terra"]

key = subprocess.run(
    ["gcloud", "secrets", "versions", "access", "latest", "--secret", "OPENAI_API_KEY", "--project", PROJECT],
    capture_output=True, text=True, check=True).stdout.strip()
client = OpenAI(api_key=key)

results = {"meta": {"endpoint": "/v1/responses", "prefix_chars": len(PREFIX), "question": Q}, "models": {}}

for model in MODELS:
    print(f"\n[{model}]")
    calls = []
    try:
        for i in range(2):
            t0 = time.perf_counter()
            r = client.responses.create(
                model=model,
                instructions=PREFIX,          # the static prefix
                input=Q,
                max_output_tokens=16,
            )
            dt = time.perf_counter() - t0
            u = r.usage
            details = getattr(u, "input_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) if details else 0
            calls.append({"call": i + 1, "latency_s": round(dt, 3),
                          "input_tokens": u.input_tokens,
                          "cached_tokens": cached,
                          "output_tokens": u.output_tokens})
            print(f"   call {i+1}: {dt:5.2f}s  input={u.input_tokens}  cached={cached}")
            if i == 0:
                time.sleep(2)
        results["models"][model] = {"mechanism": "implicit (automatic)", "calls": calls}
    except Exception as e:
        print(f"   FAILED {type(e).__name__}: {str(e)[:160]}")
        results["models"][model] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "api_gpt56_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nwrote api_gpt56_results.json")

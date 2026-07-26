"""Closed-model prompt caching: measured cache-miss vs cache-hit.

Closed models never expose their KV tensors, but they do expose the *effect* of
reusing them: a cached prefix is cheaper and faster to first token. For each
provider we send the SAME ~12k-token prefix twice and record latency + the
provider's own cached-token accounting.

  OpenAI     automatic caching (>=1024 tok)  -> usage.prompt_tokens_details.cached_tokens
  Anthropic  explicit cache_control          -> cache_creation_input_tokens / cache_read_input_tokens
  Gemini     implicit caching                -> usageMetadata.cachedContentTokenCount

Deliberately cheap: 2 calls/provider, max_tokens tiny.
Writes api_cache_results.json.
"""
import json
import os
import subprocess
import time

PREFIX_TARGET_TOKENS = 12000
UNIT = ("You are a shopping assistant for an eyewear catalog. Only answer questions about products, "
        "orders, and store policy. Never invent a product that is not in the catalog. When the user asks "
        "for something similar to a named product, use the similarity tool with that product as the seed. ")
QUESTION = "In one word, what do you help with?"


def secret(name: str) -> str:
    """Read a credential from the environment, falling back to Google Secret
    Manager when GCP_PROJECT is set (how we run it internally)."""
    val = os.environ.get(name)
    if val:
        return val
    project = os.environ.get("GCP_PROJECT")
    if not project:
        raise RuntimeError(f"set {name} in the environment (or GCP_PROJECT to use Secret Manager)")
    return subprocess.run(
        ["gcloud", "secrets", "versions", "access", "latest", "--secret", name, "--project", project],
        capture_output=True, text=True, check=True).stdout.strip()


def build_prefix(n_tokens):
    """~4 chars/token heuristic; exact count comes back from each provider's usage."""
    reps = (n_tokens * 4) // len(UNIT) + 1
    return (UNIT * reps)[: n_tokens * 4]


PREFIX = build_prefix(PREFIX_TARGET_TOKENS)
results = {"meta": {"prefix_target_tokens": PREFIX_TARGET_TOKENS, "question": QUESTION}, "providers": {}}


# ---------------- OpenAI ----------------
def run_openai():
    from openai import OpenAI
    c = OpenAI(api_key=secret("OPENAI_API_KEY"))
    model = "gpt-4.1-mini"
    msgs = [{"role": "system", "content": PREFIX}, {"role": "user", "content": QUESTION}]
    out = []
    for label in ("miss", "hit"):
        t0 = time.perf_counter()
        r = c.chat.completions.create(model=model, messages=msgs, max_tokens=5, temperature=0)
        dt = time.perf_counter() - t0
        u = r.usage
        cached = getattr(u, "prompt_tokens_details", None)
        cached = getattr(cached, "cached_tokens", 0) if cached else 0
        out.append({"call": label, "latency_s": round(dt, 3), "prompt_tokens": u.prompt_tokens,
                    "cached_tokens": cached})
        print(f"    openai {label:4}: {dt:5.2f}s  prompt={u.prompt_tokens}  cached={cached}")
        if label == "miss":
            time.sleep(2)
    return {"model": model, "calls": out}


# ---------------- Anthropic ----------------
def run_anthropic():
    import anthropic
    c = anthropic.Anthropic(api_key=secret("ANTHROPIC_API_KEY"))
    model = "claude-sonnet-4-5"
    system = [{"type": "text", "text": PREFIX, "cache_control": {"type": "ephemeral"}}]
    out = []
    for label in ("miss", "hit"):
        t0 = time.perf_counter()
        r = c.messages.create(model=model, max_tokens=5, system=system,
                              messages=[{"role": "user", "content": QUESTION}])
        dt = time.perf_counter() - t0
        u = r.usage
        out.append({"call": label, "latency_s": round(dt, 3),
                    "input_tokens": u.input_tokens,
                    "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0),
                    "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0)})
        print(f"    anthropic {label:4}: {dt:5.2f}s  in={u.input_tokens} "
              f"write={getattr(u,'cache_creation_input_tokens',0)} read={getattr(u,'cache_read_input_tokens',0)}")
        if label == "miss":
            time.sleep(2)
    return {"model": model, "calls": out}


# ---------------- Gemini ----------------
def run_gemini():
    from google import genai
    from google.genai import types
    c = genai.Client(api_key=secret("GEMINI_API_KEY"))
    model = "gemini-2.5-flash"
    out = []
    for label in ("miss", "hit"):
        t0 = time.perf_counter()
        r = c.models.generate_content(
            model=model, contents=PREFIX + "\n\n" + QUESTION,
            config=types.GenerateContentConfig(max_output_tokens=5, temperature=0))
        dt = time.perf_counter() - t0
        um = r.usage_metadata
        out.append({"call": label, "latency_s": round(dt, 3),
                    "prompt_tokens": um.prompt_token_count,
                    "cached_tokens": getattr(um, "cached_content_token_count", 0) or 0})
        print(f"    gemini {label:4}: {dt:5.2f}s  prompt={um.prompt_token_count} "
              f"cached={getattr(um,'cached_content_token_count',0) or 0}")
        if label == "miss":
            time.sleep(2)
    return {"model": model, "calls": out}


for name, fn in [("openai", run_openai), ("anthropic", run_anthropic), ("gemini", run_gemini)]:
    print(f"\n[{name}]")
    try:
        results["providers"][name] = fn()
    except Exception as e:
        print(f"    FAILED: {type(e).__name__}: {str(e)[:180]}")
        results["providers"][name] = {"error": f"{type(e).__name__}: {str(e)[:180]}"}

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "api_cache_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nwrote api_cache_results.json")

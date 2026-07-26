"""Explicit prompt/context caching APIs on hosted models.

Round 1 (bench_api_cache.py) measured *implicit* caching. This measures the
explicit, developer-controlled APIs — the ones you'd actually build on:

  Gemini    CachedContent — create a cache object with a TTL, then reference it
  Anthropic cache_control — mark a prefix; 5-minute default or "1h" TTL
  OpenAI    no explicit API (automatic only) — included for contrast

Also probes the newest model generation (Claude Sonnet 5 / Opus 5).
Writes api_explicit_results.json.
"""
import json
import os
import subprocess
import time

PROJECT = "agent-ari-a63af"
UNIT = ("You are a shopping assistant for an eyewear catalog. Only answer questions about products, "
        "orders, and store policy. Never invent a product that is not in the catalog. When the user asks "
        "for something similar to a named product, use the similarity tool with that product as the seed. ")
PREFIX = (UNIT * 1200)[:48000]          # ~12k tokens
Q = "In one word, what do you help with?"

secret = lambda n: subprocess.run(
    ["gcloud", "secrets", "versions", "access", "latest", "--secret", n, "--project", PROJECT],
    capture_output=True, text=True, check=True).stdout.strip()

results = {"meta": {"prefix_chars": len(PREFIX), "question": Q}, "providers": {}}


# ---------------- Gemini: explicit CachedContent ----------------
def gemini_explicit():
    from google import genai
    from google.genai import types
    c = genai.Client(api_key=secret("GEMINI_API_KEY"))
    model = "gemini-2.5-flash"
    out = {"model": model, "mechanism": "explicit CachedContent (TTL set by caller)"}

    t0 = time.perf_counter()
    cache = c.caches.create(model=model, config=types.CreateCachedContentConfig(
        contents=[types.Content(role="user", parts=[types.Part(text=PREFIX)])],
        ttl="600s"))
    out["cache_create_s"] = round(time.perf_counter() - t0, 3)
    out["cached_token_count"] = getattr(cache.usage_metadata, "total_token_count", None)
    out["cache_name"] = cache.name
    print(f"    created cache: {out['cached_token_count']} tokens in {out['cache_create_s']}s, ttl=600s")

    calls = []
    for i in range(2):
        t0 = time.perf_counter()
        r = c.models.generate_content(model=model, contents=Q,
                                      config=types.GenerateContentConfig(cached_content=cache.name,
                                                                         max_output_tokens=5))
        dt = time.perf_counter() - t0
        um = r.usage_metadata
        calls.append({"call": i + 1, "latency_s": round(dt, 3),
                      "prompt_tokens": um.prompt_token_count,
                      "cached_tokens": getattr(um, "cached_content_token_count", 0) or 0})
        print(f"    call {i+1}: {dt:.2f}s  prompt={um.prompt_token_count} "
              f"cached={getattr(um,'cached_content_token_count',0) or 0}")
    out["calls"] = calls
    try:
        c.caches.delete(name=cache.name)
        out["cleaned_up"] = True
    except Exception:
        out["cleaned_up"] = False
    return out


# ---------------- Anthropic: cache_control, newest models ----------------
def anthropic_explicit():
    import anthropic
    c = anthropic.Anthropic(api_key=secret("ANTHROPIC_API_KEY"))
    out = {"mechanism": "explicit cache_control breakpoint", "models": {}}
    for model, ttl in [("claude-sonnet-5", None), ("claude-sonnet-5", "1h")]:
        cc = {"type": "ephemeral"} | ({"ttl": ttl} if ttl else {})
        system = [{"type": "text", "text": PREFIX, "cache_control": cc}]
        label = f"{model} ttl={ttl or '5m (default)'}"
        calls = []
        try:
            for i in range(2):
                t0 = time.perf_counter()
                r = c.messages.create(model=model, max_tokens=5, system=system,
                                      messages=[{"role": "user", "content": Q}])
                dt = time.perf_counter() - t0
                u = r.usage
                calls.append({"call": i + 1, "latency_s": round(dt, 3),
                              "input_tokens": u.input_tokens,
                              "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0),
                              "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0)})
                print(f"    {label} call {i+1}: {dt:.2f}s "
                      f"write={getattr(u,'cache_creation_input_tokens',0)} "
                      f"read={getattr(u,'cache_read_input_tokens',0)}")
                if i == 0:
                    time.sleep(2)
            out["models"][label] = {"calls": calls}
        except Exception as e:
            print(f"    {label} FAILED: {type(e).__name__}: {str(e)[:120]}")
            out["models"][label] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    return out


for name, fn in [("gemini_explicit", gemini_explicit), ("anthropic_explicit", anthropic_explicit)]:
    print(f"\n[{name}]")
    try:
        results["providers"][name] = fn()
    except Exception as e:
        print(f"    FAILED: {type(e).__name__}: {str(e)[:200]}")
        results["providers"][name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "api_explicit_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nwrote api_explicit_results.json")

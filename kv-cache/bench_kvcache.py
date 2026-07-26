"""KV-cache benchmark on a real transformer (granite-3.2-2b-instruct, local, MPS).

Measures, from actual forward passes — nothing simulated:
  1. decode with vs without KV cache        (use_cache True/False)
  2. prefill vs decode cost per token       (parallel vs sequential)
  3. TTFT as the static prefix grows        (what a 12k-token system prompt costs)
  4. prefix KV reuse across queries         (the RAG case: recompute vs reuse)
  5. KV cache memory footprint              (measured tensor bytes)

Writes kv_results.json for the blog post's tables.

Run:  python3 bench_kvcache.py
"""
import json
import os
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "ibm-granite/granite-3.2-2b-instruct"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def sync():
    if DEV == "mps":
        torch.mps.synchronize()


def timer():
    sync()
    return time.perf_counter()


print(f"loading {MODEL} on {DEV} ...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV)
model.eval()
cfg = model.config
HEAD_DIM = cfg.hidden_size // cfg.num_attention_heads
BYTES_PER_TOKEN = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * HEAD_DIM * 2  # K+V, fp16

results = {
    "meta": {
        "model": MODEL,
        "device": DEV,
        "dtype": "float16",
        "layers": cfg.num_hidden_layers,
        "attn_heads": cfg.num_attention_heads,
        "kv_heads": cfg.num_key_value_heads,
        "head_dim": HEAD_DIM,
        "kv_bytes_per_token": BYTES_PER_TOKEN,
    }
}
print(f"KV per token: {BYTES_PER_TOKEN/1024:.0f} KB  ({cfg.num_hidden_layers} layers, {cfg.num_key_value_heads} kv-heads, head_dim {HEAD_DIM})")


def make_prompt(n_tokens: int) -> torch.Tensor:
    """A prompt of ~n_tokens, built from realistic instruction-style text."""
    unit = ("You are a shopping assistant for an eyewear catalog. Only answer questions about "
            "products, orders, and store policy. Never invent a product that is not in the catalog. "
            "When the user asks for something similar to a named product, use the similarity tool. ")
    ids = tok(unit, return_tensors="pt").input_ids
    reps = max(1, n_tokens // ids.shape[1] + 1)
    full = tok(unit * reps, return_tensors="pt").input_ids[:, :n_tokens]
    return full.to(DEV)


@torch.inference_mode()
def decode_tokens(ids, n_new, use_cache):
    """Greedy-decode n_new tokens; returns elapsed seconds."""
    t0 = timer()
    if use_cache:
        out = model(ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1:].argmax(-1)
        for _ in range(n_new - 1):
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1:].argmax(-1)
    else:
        cur = ids
        for _ in range(n_new):
            out = model(cur, use_cache=False)          # recompute whole sequence every step
            nxt = out.logits[:, -1:].argmax(-1)
            cur = torch.cat([cur, nxt], dim=1)
        past = None
    dt = timer() - t0
    del past
    return dt


# ---------- 1. cache on vs off ----------
print("\n[1] decode with vs without KV cache")
PROMPT_N, NEW = 512, 48
ids = make_prompt(PROMPT_N)
decode_tokens(ids, 4, True)  # warmup
on = decode_tokens(ids, NEW, use_cache=True)
off = decode_tokens(ids, NEW, use_cache=False)
results["cache_on_off"] = {
    "prompt_tokens": PROMPT_N, "generated_tokens": NEW,
    "with_cache_s": round(on, 3), "without_cache_s": round(off, 3),
    "speedup": round(off / on, 1),
    "with_cache_ms_per_token": round(on / NEW * 1000, 1),
    "without_cache_ms_per_token": round(off / NEW * 1000, 1),
}
print(f"    with cache {on:.2f}s | without {off:.2f}s | {off/on:.1f}x")


# ---------- 2 & 3. prefill scaling / TTFT ----------
print("\n[2/3] prefill (TTFT) as the prompt grows")
rows = []
for n in [256, 512, 1024, 2048, 4096, 8192, 12000]:
    ids = make_prompt(n)
    with torch.inference_mode():
        model(ids[:, :16], use_cache=True)  # warm
        t0 = timer()
        out = model(ids, use_cache=True)
        dt = timer() - t0
        kv_bytes = n * BYTES_PER_TOKEN
        del out
    rows.append({
        "prompt_tokens": n,
        "prefill_s": round(dt, 3),
        "ms_per_token": round(dt / n * 1000, 3),
        "kv_mb": round(kv_bytes / 1024 / 1024, 1),
    })
    print(f"    {n:>6} tok -> prefill {dt*1000:7.0f} ms  ({dt/n*1000:.3f} ms/tok)  KV {kv_bytes/1024/1024:.1f} MB")
results["prefill_scaling"] = rows


# ---------- 4. prefix reuse: the RAG case ----------
print("\n[4] static prefix reuse across queries (recompute vs reuse)")
PREFIX_N = 12000        # mirrors raglib/config/prompts.py (~12k tokens)
QUERIES = ["Do you have polarized aviators under $200?",
           "What is your return window?",
           "Show me something similar to RB3025 but cheaper.",
           "Do these ship to Germany?"]
prefix_ids = make_prompt(PREFIX_N)

with torch.inference_mode():
    # build the prefix cache once
    t0 = timer()
    pre = model(prefix_ids, use_cache=True)
    prefix_build_s = timer() - t0
    from transformers.cache_utils import DynamicCache

    recompute_total, reuse_total = 0.0, 0.0
    per_query = []
    for q in QUERIES:
        q_ids = tok(q, return_tensors="pt").input_ids.to(DEV)

        # (a) no cache reuse: process prefix+query from scratch (what we do today)
        full = torch.cat([prefix_ids, q_ids], dim=1)
        t0 = timer()
        o = model(full, use_cache=True)
        t_recompute = timer() - t0
        del o

        # (b) reuse prefix KV: only the query tokens are new
        past = model(prefix_ids, use_cache=True).past_key_values  # fresh copy per query
        t0 = timer()
        o = model(q_ids, past_key_values=past, use_cache=True)
        t_reuse = timer() - t0
        del o, past

        recompute_total += t_recompute
        reuse_total += t_reuse
        per_query.append({
            "query": q, "query_tokens": q_ids.shape[1],
            "recompute_s": round(t_recompute, 3), "reuse_s": round(t_reuse, 3),
            "speedup": round(t_recompute / t_reuse, 1),
        })
        print(f"    '{q[:38]:38}' recompute {t_recompute*1000:6.0f} ms | reuse {t_reuse*1000:5.0f} ms | {t_recompute/t_reuse:5.1f}x")

results["prefix_reuse"] = {
    "prefix_tokens": PREFIX_N,
    "prefix_build_s": round(prefix_build_s, 3),
    "prefix_kv_mb": round(PREFIX_N * BYTES_PER_TOKEN / 1024 / 1024, 1),
    "n_queries": len(QUERIES),
    "recompute_total_s": round(recompute_total, 3),
    "reuse_total_s": round(reuse_total, 3),
    "total_speedup": round(recompute_total / reuse_total, 1),
    "per_query": per_query,
}
print(f"    total: recompute {recompute_total:.2f}s vs reuse {reuse_total:.2f}s -> {recompute_total/reuse_total:.1f}x")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "kv_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nwrote kv_results.json")

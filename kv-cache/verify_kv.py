"""Recompute every headline number used in the KV-cache post from the raw result
JSON, so no figure in the prose is hand-typed.

Run:  python3 verify_kv.py     (exits non-zero if anything disagrees)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
load = lambda n: json.load(open(os.path.join(HERE, n)))

fails = []


def check(label, got, want, tol=0.05):
    ok = abs(got - want) <= tol * max(abs(want), 1e-9)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: got {got:.4g}, want {want:.4g}")
    if not ok:
        fails.append(label)


kv = load("kv_results.json")
m = kv["meta"]
print(f"\nModel: {m['model']} on {m['device']} ({m['layers']}L, {m['kv_heads']} kv-heads, head_dim {m['head_dim']})")
bpt = 2 * m["layers"] * m["kv_heads"] * m["head_dim"] * 2
check("KV bytes/token = 2*L*kv_heads*head_dim*2", bpt, m["kv_bytes_per_token"], 0)
print(f"  -> {bpt/1024:.0f} KB per token")

c = kv["cache_on_off"]
check("decode speedup with cache", c["without_cache_s"] / c["with_cache_s"], c["speedup"], 0.02)
print(f"  -> {c['generated_tokens']} tokens: {c['with_cache_s']}s cached vs {c['without_cache_s']}s uncached")

print("\nPrefill scaling (ms/token should grow with length — attention is quadratic):")
rows = kv["prefill_scaling"]
first, last = rows[0], rows[-1]
for r in rows:
    print(f"    {r['prompt_tokens']:>6} tok  {r['prefill_s']*1000:8.0f} ms  {r['ms_per_token']:.3f} ms/tok  KV {r['kv_mb']} MB")
degradation = last["ms_per_token"] / first["ms_per_token"]
print(f"  per-token prefill cost grew {degradation:.1f}x from {first['prompt_tokens']} to {last['prompt_tokens']} tokens")
if degradation < 1.5:
    fails.append("expected per-token prefill cost to grow with prompt length")

pr = kv["prefix_reuse"]
check("prefix-reuse total speedup", pr["recompute_total_s"] / pr["reuse_total_s"], pr["total_speedup"], 0.02)
print(f"  -> {pr['prefix_tokens']} tok prefix, {pr['n_queries']} queries: "
      f"{pr['recompute_total_s']}s recompute vs {pr['reuse_total_s']}s reuse")
check("prefix KV MB", pr["prefix_tokens"] * bpt / 1024 / 1024, pr["prefix_kv_mb"], 0.01)

mm = load("multimodel_results.json")["text_models"]
print("\nAcross open models (4k prefix, 4 queries): bigger model -> bigger win")
prev = None
for name, r in mm.items():
    if "error" in r:
        continue
    print(f"    {name.split('/')[-1]:32} {r['kb_per_token']:>5} KB/tok  {r['speedup']:>5}x  "
          f"breakeven {r['breakeven_queries']} queries")
    check(f"{name.split('/')[-1]} speedup", r["recompute_total_s"] / r["reuse_total_s"], r["speedup"], 0.02)
    if prev is not None and r["kb_per_token"] > prev[1] and r["speedup"] < prev[0]:
        fails.append(f"expected speedup to grow with model size ({name})")
    prev = (r["speedup"], r["kb_per_token"])

v = load("vision_results.json")
print(f"\nVision ({v['model'].split('/')[-1]}): image = {v['image_tokens']}/{v['prompt_tokens_with_image']} prompt tokens "
      f"({100*v['image_tokens']/v['prompt_tokens_with_image']:.1f}%)")
check("vision reuse speedup", v["recompute_total_s"] / v["reuse_total_s"], v["speedup"], 0.02)
print(f"  -> {v['recompute_per_query_s']*1000:.0f} ms/query re-encoding vs {v['reuse_per_query_s']*1000:.0f} ms/query reusing")

api = load("api_cache_results.json")["providers"]
print("\nClosed models (measured cache miss -> hit):")
for prov, d in api.items():
    if "error" in d:
        print(f"    {prov}: ERROR {d['error'][:60]}")
        continue
    miss, hit = d["calls"][0], d["calls"][1]
    cached = hit.get("cached_tokens", hit.get("cache_read_input_tokens", 0))
    print(f"    {prov:10} {d['model']:20} {miss['latency_s']:.2f}s -> {hit['latency_s']:.2f}s, "
          f"cached tokens reported on hit: {cached}")

arch = load("kv_arch.json")
print("\nKV footprint by architecture (fp16):")
for name, a in sorted(arch.items(), key=lambda kv_: kv_[1]["kb_per_token"]):
    exp = 2 * a["layers"] * a["kv_heads"] * a["head_dim"] * 2
    if exp != a["bytes_per_token"]:
        fails.append(f"arch math {name}")
    print(f"    {name:42} {a['kb_per_token']:>6} KB/tok  GQA {a['gqa_ratio']}:1  {a['gb_at_128k']:>5} GB @128k")

print("\n" + ("ALL CHECKS PASSED" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)

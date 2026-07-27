"""How much of a policy set does a model actually apply?

The task: given a user message and N policies written in natural language,
return the subset whose conditions the message triggers. Ground truth is in
dataset.py. This is the decision our runtime makes on every conversation turn.

One knob controls everything: k = how many policies go into a single LLM call.

    k = N   ->  one call sees every policy      (what most systems do)
    k = 1   ->  one call per policy             (what raglib does)
    1<k<N   ->  sharded, k policies per call

Plus a `monolithic` arm: k = N *and* a free-form list output, which is the
naive "paste the rules in the system prompt" baseline. Comparing monolithic to
k=N isolates the effect of the output format; comparing k=N to k=1 isolates the
effect of decomposition.

Writes gating_results.json.
"""
import argparse
import asyncio
import json
import math
import os
import sys
import time

import tiktoken

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import (GOLD, POLICIES, POLICY_CONDITION, POLICY_IDS, POLICY_TYPE,
                     QUERY_TEXTS, SITE_DESCRIPTION, context_block)
from providers import build, gather_limited

ENC = tiktoken.get_encoding("o200k_base")

PREAMBLE = (
    "You are the policy gate for the Solstice Optics shopping assistant. A brand operator has "
    "configured policies that change how the assistant retrieves and answers. Your only job is "
    "to decide which policies the user's message triggers this turn."
)

MANY_TMPL = """{context}{preamble}

POLICIES ({n} total)
{policy_list}

Return a decision for every policy id listed above. Mark a policy true only if its condition is
clearly triggered by the user's message; otherwise mark it false. Judge each policy on its own
condition text."""

MONO_TMPL = """{context}{preamble}

POLICIES ({n} total)
{policy_list}

Return the ids of all policies whose condition is clearly triggered by the user's message, and no
others. If none are triggered, return an empty list."""

ONE_TMPL = """{context}{preamble}

Decide whether the following policy should be applied on this conversation turn.
Mark it true only if the policy's condition is clearly triggered by the user's message;
otherwise mark it false.

Policy id: {pid}
Policy type: {ptype}
Policy condition: {condition}"""

# k=1 loses the contrastive context that a full list provides: asked about one
# broad condition in isolation, a model has nothing better-fitting to prefer, so
# it over-fires. This arm keeps one call per policy but shows the rest of the
# list as read-only context, decided by someone else.
ONE_CTX_TMPL = """{context}{preamble}

The full policy set is listed below for context. Other policies are decided separately by
other reviewers — you are responsible for exactly one of them.

FULL POLICY SET ({n} total)
{policy_list}

YOUR POLICY: {pid}
Policy type: {ptype}
Policy condition: {condition}

Decide only about {pid}. Mark it true only if its condition is clearly triggered by the user's
message; otherwise mark it false. If another policy in the list fits the message better than
yours does, that is a reason to mark yours false."""


def policy_lines(pids):
    return "\n".join(
        f"- {pid} (type: {POLICY_TYPE[pid]}): {POLICY_CONDITION[pid]}" for pid in pids)


def shards(pids, k):
    return [pids[i:i + k] for i in range(0, len(pids), k)]


def enum_schema(pids):
    """One required boolean per policy id — the model cannot silently skip one."""
    return {
        "type": "object",
        "properties": {pid: {"type": "boolean"} for pid in pids},
        "required": list(pids),
        "additionalProperties": False,
    }


MONO_SCHEMA = {
    "type": "object",
    "properties": {"applicable_policy_ids": {
        "type": "array", "items": {"type": "string", "enum": POLICY_IDS}}},
    "required": ["applicable_policy_ids"],
    "additionalProperties": False,
}
ONE_SCHEMA = {
    "type": "object",
    "properties": {"applies": {"type": "boolean"}},
    "required": ["applies"],
    "additionalProperties": False,
}


async def run_arm(prov, arm, ctx, queries, concurrency, cache):
    """Return (per-query predicted sets, call stats)."""
    jobs, meta = [], []
    for q in queries:
        if arm == "monolithic":
            sysp = MONO_TMPL.format(context=ctx, preamble=PREAMBLE,
                                    n=len(POLICY_IDS), policy_list=policy_lines(POLICY_IDS))
            jobs.append(prov.call(sysp, q, MONO_SCHEMA, cache))
            meta.append((q, None))
        elif arm in ("k1", "k1_contrastive"):
            tmpl = ONE_TMPL if arm == "k1" else ONE_CTX_TMPL
            for pid in POLICY_IDS:
                sysp = tmpl.format(context=ctx, preamble=PREAMBLE, pid=pid,
                                   ptype=POLICY_TYPE[pid], condition=POLICY_CONDITION[pid],
                                   n=len(POLICY_IDS), policy_list=policy_lines(POLICY_IDS))
                jobs.append(prov.call(sysp, q, ONE_SCHEMA, cache))
                meta.append((q, [pid]))
        else:
            k = int(arm[1:])
            for group in shards(POLICY_IDS, k):
                sysp = MANY_TMPL.format(context=ctx, preamble=PREAMBLE,
                                        n=len(group), policy_list=policy_lines(group))
                jobs.append(prov.call(sysp, q, enum_schema(group), cache))
                meta.append((q, group))

    t0 = time.perf_counter()
    results = await gather_limited(jobs, concurrency)
    wall = time.perf_counter() - t0

    pred = {q: set() for q in queries}
    stats = {"calls": 0, "errors": 0, "unparsed": 0, "in_tok": 0, "out_tok": 0,
             "cached_tok": 0, "latency_sum": 0.0, "wall_s": round(wall, 1),
             "missing_fields": 0}
    for (q, group), r in zip(meta, results):
        stats["calls"] += 1
        stats["in_tok"] += r.in_tok
        stats["out_tok"] += r.out_tok
        stats["cached_tok"] += r.cached_tok
        stats["latency_sum"] += r.latency_s
        if r.error:
            stats["errors"] += 1
            continue
        if not isinstance(r.data, dict):
            stats["unparsed"] += 1
            continue
        if group is None:
            got = r.data.get("applicable_policy_ids") or []
            pred[q] |= {p for p in got if p in POLICY_IDS}
        elif len(group) == 1 and "applies" in r.data:
            if r.data.get("applies") is True:
                pred[q].add(group[0])
        else:
            for pid in group:
                if pid not in r.data:
                    stats["missing_fields"] += 1
                elif r.data[pid] is True:
                    pred[q].add(pid)
    return pred, stats


def score(pred, queries):
    tp = fp = fn = 0
    exact = 0
    per_policy = {pid: {"tp": 0, "fp": 0, "fn": 0} for pid in POLICY_IDS}
    per_query = []
    for q in queries:
        g, p = GOLD[q], pred[q]
        qtp, qfp, qfn = len(g & p), len(p - g), len(g - p)
        tp += qtp; fp += qfp; fn += qfn
        exact += int(g == p)
        for pid in g & p:
            per_policy[pid]["tp"] += 1
        for pid in p - g:
            per_policy[pid]["fp"] += 1
        for pid in g - p:
            per_policy[pid]["fn"] += 1
        per_query.append({"query": q, "gold": sorted(g), "pred": sorted(p),
                          "missed": sorted(g - p), "spurious": sorted(p - g)})
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        "exact_set_match": round(exact / len(queries), 4),
        "tp": tp, "fp": fp, "fn": fn,
        "pairs": len(queries) * len(POLICY_IDS),
        "per_policy": per_policy, "per_query": per_query,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[
        "anthropic:claude-opus-5",
        "anthropic:claude-haiku-4-5-20251001",
        "openai:gpt-5.6-sol",
        "google:gemini-2.5-flash",
    ])
    ap.add_argument("--arms", nargs="+", default=["monolithic", "k25", "k5", "k1"])
    ap.add_argument("--context-tokens", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit-queries", type=int, default=0)
    ap.add_argument("--out", default="gating_results.json")
    args = ap.parse_args()

    queries = QUERY_TEXTS[:args.limit_queries] if args.limit_queries else QUERY_TEXTS
    ctx = context_block(args.context_tokens, ENC)
    ctx = (ctx + "\n\n") if ctx else ""
    cache = args.context_tokens >= 1024

    out = {"meta": {
        "n_policies": len(POLICY_IDS), "n_queries": len(queries),
        "pairs": len(POLICY_IDS) * len(queries),
        "context_tokens": args.context_tokens,
        "gold_positives": sum(len(GOLD[q]) for q in queries),
        "prompt_caching": cache,
    }, "runs": {}}

    for spec in args.models:
        prov = build(spec)
        out["runs"][spec] = {}
        for arm in args.arms:
            print(f"\n=== {spec} | {arm} | ctx={args.context_tokens} ===", flush=True)
            pred, stats = await run_arm(prov, arm, ctx, queries, args.concurrency, cache)
            sc = score(pred, queries)
            n_calls = stats["calls"]
            sc_small = {k: v for k, v in sc.items() if k not in ("per_policy", "per_query")}
            print(f"  P {sc['precision']:.3f}  R {sc['recall']:.3f}  F1 {sc['f1']:.3f}  "
                  f"exact {sc['exact_set_match']:.3f}  |  {n_calls} calls, "
                  f"{stats['in_tok']:,} in ({stats['cached_tok']:,} cached), "
                  f"{stats['errors']} err, {stats['unparsed']} unparsed, "
                  f"{stats['missing_fields']} missing fields", flush=True)
            out["runs"][spec][arm] = {"score": sc_small, "stats": stats,
                                      "per_policy": sc["per_policy"],
                                      "per_query": sc["per_query"],
                                      "calls_per_query": round(n_calls / len(queries), 2)}
            here = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(here, args.out), "w") as f:
                json.dump(out, f, indent=2)

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())

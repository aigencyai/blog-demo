"""Two questions a single call has to answer and a decomposed one does not:

  1. SCALING   - what happens to a single call as the policy list grows from
                 25 to 100? Distractors pad the list; the 25 labelled policies
                 and their gold labels are untouched, so any distractor that
                 fires is a false positive and every miss on the core 25 is a
                 real regression.

  2. POSITION  - within one call, does where a policy sits in the list change
                 whether it fires? Same list, several seeded shuffles, recall
                 bucketed by the position the policy happened to land in.

Note on the decomposed arm: at k=1 a call sees exactly one policy, so growing
the list cannot change any core-25 decision — those calls are byte-identical
to the ones in bench_gating.py. Its distractor false positives come from
screen_distractors.py. Re-running it here would burn tokens to reproduce
results we already have, so this script only runs the single-call arms.

Writes scaling_results.json.
"""
import argparse
import asyncio
import json
import os
import random
import sys

import tiktoken

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_gating import MANY_TMPL, MONO_TMPL, PREAMBLE, score
from dataset import (GOLD, POLICY_CONDITION, POLICY_IDS, POLICY_TYPE,
                     QUERY_TEXTS, context_block)
from distractors import DISTRACTORS
from providers import build, gather_limited

ENC = tiktoken.get_encoding("o200k_base")
DTYPE = {d[0]: d[1] for d in DISTRACTORS}
DCOND = {d[0]: d[2] for d in DISTRACTORS}
ALLTYPE = {**POLICY_TYPE, **DTYPE}
ALLCOND = {**POLICY_CONDITION, **DCOND}


def lines(pids):
    return "\n".join(f"- {p} (type: {ALLTYPE[p]}): {ALLCOND[p]}" for p in pids)


def mono_schema(pids):
    return {"type": "object",
            "properties": {"applicable_policy_ids": {
                "type": "array", "items": {"type": "string", "enum": list(pids)}}},
            "required": ["applicable_policy_ids"], "additionalProperties": False}


def enum_schema(pids):
    return {"type": "object",
            "properties": {p: {"type": "boolean"} for p in pids},
            "required": list(pids), "additionalProperties": False}


def build_list(n, seed=0):
    """25 labelled policies + (n-25) distractors, interleaved by a seeded
    shuffle so the labelled ones are not clustered at the top."""
    extra = [d[0] for d in DISTRACTORS][:n - len(POLICY_IDS)]
    pids = list(POLICY_IDS) + extra
    random.Random(seed).shuffle(pids)
    return pids


async def one_pass(prov, arm, pids, ctx, queries, conc):
    if arm == "monolithic":
        sysp = MONO_TMPL.format(context=ctx, preamble=PREAMBLE,
                                n=len(pids), policy_list=lines(pids))
        schema = mono_schema(pids)
    else:
        sysp = MANY_TMPL.format(context=ctx, preamble=PREAMBLE,
                                n=len(pids), policy_list=lines(pids))
        schema = enum_schema(pids)

    res = await gather_limited([prov.call(sysp, q, schema, True) for q in queries], conc)

    pred, stats = {}, {"calls": 0, "errors": 0, "in_tok": 0, "out_tok": 0,
                       "cached_tok": 0, "distractor_fp": 0, "missing_fields": 0}
    for q, r in zip(queries, res):
        stats["calls"] += 1
        stats["in_tok"] += r.in_tok
        stats["out_tok"] += r.out_tok
        stats["cached_tok"] += r.cached_tok
        got = set()
        if r.error:
            stats["errors"] += 1
        elif isinstance(r.data, dict):
            if arm == "monolithic":
                got = {p for p in (r.data.get("applicable_policy_ids") or []) if p in pids}
            else:
                for p in pids:
                    if p not in r.data:
                        stats["missing_fields"] += 1
                    elif r.data[p] is True:
                        got.add(p)
        stats["distractor_fp"] += len(got - set(POLICY_IDS))
        pred[q] = got & set(POLICY_IDS)
    return pred, stats


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[
        "anthropic:claude-opus-5",
        "anthropic:claude-haiku-4-5-20251001",
        "openai:gpt-5.6-sol",
        "google:gemini-2.5-flash",
    ])
    ap.add_argument("--sizes", nargs="+", type=int, default=[25, 50, 75, 100])
    ap.add_argument("--arms", nargs="+", default=["monolithic", "enumerated"])
    ap.add_argument("--position-seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--position-size", type=int, default=100)
    ap.add_argument("--context-tokens", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default="scaling_results.json")
    args = ap.parse_args()

    ctx = context_block(args.context_tokens, ENC) + "\n\n"
    queries = QUERY_TEXTS
    here = os.path.dirname(os.path.abspath(__file__))
    out = {"meta": {"sizes": args.sizes, "n_queries": len(queries),
                    "core_policies": len(POLICY_IDS),
                    "context_tokens": args.context_tokens}, "scaling": {}, "position": {}}

    for spec in args.models:
        prov = build(spec)
        out["scaling"][spec] = {}
        for n in args.sizes:
            pids = build_list(n)
            for arm in args.arms:
                print(f"\n=== {spec} | {arm} | N={n} ===", flush=True)
                pred, stats = await one_pass(prov, arm, pids, ctx, queries, args.concurrency)
                sc = score(pred, queries)
                small = {k: v for k, v in sc.items() if k not in ("per_policy", "per_query")}
                small["distractor_false_positives"] = stats["distractor_fp"]
                print(f"  P {sc['precision']:.3f}  R {sc['recall']:.3f}  F1 {sc['f1']:.3f}  "
                      f"exact {sc['exact_set_match']:.3f}  distractor-FP {stats['distractor_fp']}  "
                      f"({stats['errors']} err, {stats['missing_fields']} missing)", flush=True)
                out["scaling"][spec][f"{arm}@{n}"] = {
                    "score": small, "stats": stats, "per_query": sc["per_query"]}
                with open(os.path.join(here, args.out), "w") as f:
                    json.dump(out, f, indent=2)

        # position sensitivity at the largest list, monolithic arm
        out["position"][spec] = {}
        for seed in args.position_seeds:
            pids = build_list(args.position_size, seed=seed)
            print(f"\n=== {spec} | position | seed={seed} N={args.position_size} ===", flush=True)
            pred, stats = await one_pass(prov, "monolithic", pids, ctx, queries, args.concurrency)
            rank = {p: i for i, p in enumerate(pids)}
            buckets = {"first_third": [0, 0], "middle_third": [0, 0], "last_third": [0, 0]}
            for q in queries:
                for pid in GOLD[q]:
                    frac = rank[pid] / len(pids)
                    b = "first_third" if frac < 1 / 3 else ("middle_third" if frac < 2 / 3 else "last_third")
                    buckets[b][1] += 1
                    buckets[b][0] += int(pid in pred[q])
            sc = score(pred, queries)
            print("  recall by position: " + "  ".join(
                f"{b} {h}/{t}" + (f" ({h/t:.2f})" if t else "") for b, (h, t) in buckets.items()),
                flush=True)
            out["position"][spec][f"seed{seed}"] = {
                "recall": sc["recall"], "precision": sc["precision"],
                "buckets": {b: {"hit": h, "total": t} for b, (h, t) in buckets.items()},
                "distractor_fp": stats["distractor_fp"]}
            with open(os.path.join(here, args.out), "w") as f:
                json.dump(out, f, indent=2)

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())

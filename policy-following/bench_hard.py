"""Hard tier runner: compositional policies over multi-turn conversations.

Same arms and metrics as bench_gating.py, different data. The policy list can
be padded with the easy-tier policies and the distractors so the gate is asked
to hold a realistic number of rules at once (--pool 120), while only the 20
compositional policies carry gold labels.

Writes hard_results.json.
"""
import argparse
import asyncio
import json
import os
import sys
import time

import tiktoken

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import POLICY_CONDITION, POLICY_IDS, POLICY_TYPE, context_block
from dataset_hard import (HARD_COND, HARD_CONVERSATIONS, HARD_GOLD, HARD_IDS,
                          HARD_KEYS, HARD_TYPE)
from distractors import DISTRACTORS
from providers import build, gather_limited

ENC = tiktoken.get_encoding("o200k_base")

ALLTYPE = {**HARD_TYPE, **POLICY_TYPE, **{d[0]: d[1] for d in DISTRACTORS}}
ALLCOND = {**HARD_COND, **POLICY_CONDITION, **{d[0]: d[2] for d in DISTRACTORS}}

PREAMBLE = (
    "You are the policy gate for the Solstice Optics shopping assistant. A brand operator has "
    "configured policies that change how the assistant retrieves and answers. Decide which "
    "policies the conversation below triggers on its LATEST user turn.\n"
    "Policy conditions are precise. Several combine multiple clauses with AND, refer to earlier "
    "turns, count things, or carry an exception. Every clause must hold before you mark a policy "
    "true.")

MANY_TMPL = """{context}{preamble}

POLICIES ({n} total)
{policy_list}

Return a decision for every policy id listed above. Mark a policy true only if EVERY clause of
its condition holds for this conversation; otherwise mark it false."""

MONO_TMPL = """{context}{preamble}

POLICIES ({n} total)
{policy_list}

Return the ids of all policies for which EVERY clause of the condition holds for this
conversation, and no others. If none hold, return an empty list."""

ONE_TMPL = """{context}{preamble}

Decide whether the following policy applies to this conversation.
Mark it true only if EVERY clause of its condition holds; otherwise mark it false.

Policy id: {pid}
Policy type: {ptype}
Policy condition: {condition}"""

ONE_CTX_TMPL = """{context}{preamble}

The full policy set is listed below for context. Other policies are decided separately by other
reviewers — you are responsible for exactly one of them.

FULL POLICY SET ({n} total)
{policy_list}

YOUR POLICY: {pid}
Policy type: {ptype}
Policy condition: {condition}

Decide only about {pid}. Mark it true only if EVERY clause of its condition holds. If another
policy in the list fits this conversation better than yours does, that is a reason to mark
yours false."""


def lines(pids):
    return "\n".join(f"- {p} (type: {ALLTYPE[p]}): {ALLCOND[p]}" for p in pids)


def build_pool(size, seed=0):
    """20 labelled hard policies padded up to `size` with DISTRACTORS ONLY.

    Padding must never be legitimately triggerable, or "padding false positive"
    stops meaning anything. The easy-tier policies fail that test — a hard
    conversation about trail running really does satisfy boost_sports_wrap — so
    only the distractor set, screened against these conversations by
    screen_distractors.py --hard, is used. That caps the pool at 95.
    """
    import random
    # Screened out by screen_distractors.py --hard: these two genuinely fire on a
    # hard conversation ("...14 year old who plays tennis", "Do you ship to
    # Canada?"), so counting them as false positives would be wrong.
    banned = {"d_tennis", "d_international_ship"}
    avail = [d[0] for d in DISTRACTORS if d[0] not in banned]
    if size > len(HARD_IDS) + len(avail):
        raise SystemExit(f"max pool is {len(HARD_IDS) + len(avail)} "
                         f"(20 labelled + {len(avail)} screened distractors)")
    pool = list(HARD_IDS) + avail[:max(0, size - len(HARD_IDS))]
    random.Random(seed).shuffle(pool)
    return pool


def mono_schema(pids):
    return {"type": "object",
            "properties": {"applicable_policy_ids": {
                "type": "array", "items": {"type": "string", "enum": list(pids)}}},
            "required": ["applicable_policy_ids"], "additionalProperties": False}


def enum_schema(pids):
    return {"type": "object",
            "properties": {p: {"type": "boolean"} for p in pids},
            "required": list(pids), "additionalProperties": False}


ONE_SCHEMA = {"type": "object", "properties": {"applies": {"type": "boolean"}},
              "required": ["applies"], "additionalProperties": False}


async def run_arm(prov, arm, pool, ctx, conc):
    jobs, meta = [], []
    for key in HARD_KEYS:
        if arm == "monolithic":
            jobs.append(prov.call(MONO_TMPL.format(context=ctx, preamble=PREAMBLE,
                                                   n=len(pool), policy_list=lines(pool)),
                                  key, mono_schema(pool), True))
            meta.append((key, None))
        elif arm == "enumerated":
            jobs.append(prov.call(MANY_TMPL.format(context=ctx, preamble=PREAMBLE,
                                                   n=len(pool), policy_list=lines(pool)),
                                  key, enum_schema(pool), True))
            meta.append((key, pool))
        elif arm in ("k1", "k1_contrastive"):
            tmpl = ONE_TMPL if arm == "k1" else ONE_CTX_TMPL
            for pid in pool:
                jobs.append(prov.call(
                    tmpl.format(context=ctx, preamble=PREAMBLE, pid=pid, ptype=ALLTYPE[pid],
                                condition=ALLCOND[pid], n=len(pool), policy_list=lines(pool)),
                    key, ONE_SCHEMA, True))
                meta.append((key, [pid]))
        else:
            raise ValueError(arm)

    t0 = time.perf_counter()
    res = await gather_limited(jobs, conc)
    wall = time.perf_counter() - t0

    pred = {k: set() for k in HARD_KEYS}
    st = {"calls": 0, "errors": 0, "in_tok": 0, "out_tok": 0, "cached_tok": 0,
          "pad_fp": 0, "wall_s": round(wall, 1)}
    for (key, group), r in zip(meta, res):
        st["calls"] += 1
        st["in_tok"] += r.in_tok
        st["out_tok"] += r.out_tok
        st["cached_tok"] += r.cached_tok
        if r.error:
            st["errors"] += 1
            continue
        if not isinstance(r.data, dict):
            continue
        got = set()
        if group is None:
            got = {p for p in (r.data.get("applicable_policy_ids") or []) if p in pool}
        elif len(group) == 1 and "applies" in r.data:
            if r.data.get("applies") is True:
                got = {group[0]}
        else:
            got = {p for p in group if r.data.get(p) is True}
        st["pad_fp"] += len(got - set(HARD_IDS))
        pred[key] |= got & set(HARD_IDS)

    tp = fp = fn = exact = 0
    per_policy = {p: {"tp": 0, "fp": 0, "fn": 0} for p in HARD_IDS}
    rows = []
    for key in HARD_KEYS:
        g, p = HARD_GOLD[key], pred[key]
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
        exact += int(g == p)
        for x in g & p:
            per_policy[x]["tp"] += 1
        for x in p - g:
            per_policy[x]["fp"] += 1
        for x in g - p:
            per_policy[x]["fn"] += 1
        rows.append({"conversation": key, "gold": sorted(g), "pred": sorted(p),
                     "missed": sorted(g - p), "spurious": sorted(p - g)})
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "exact_set_match": round(exact / len(HARD_KEYS), 4),
            "tp": tp, "fp": fp, "fn": fn, "padding_false_positives": st["pad_fp"],
            "stats": st, "per_policy": per_policy, "per_conversation": rows,
            "calls_per_turn": round(st["calls"] / len(HARD_KEYS), 2)}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[
        "anthropic:claude-opus-5",
        "anthropic:claude-haiku-4-5-20251001",
        "openai:gpt-5.6-sol",
        "google:gemini-2.5-flash",
    ])
    ap.add_argument("--arms", nargs="+", default=["monolithic", "enumerated", "k1"])
    ap.add_argument("--pool", type=int, default=20, help="total policies in the list")
    ap.add_argument("--context-tokens", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--out", default="hard_results.json")
    args = ap.parse_args()

    pool = build_pool(args.pool)
    ctx = context_block(args.context_tokens, ENC) + "\n\n"
    here = os.path.dirname(os.path.abspath(__file__))
    gold_pos = sum(len(g) for g in HARD_GOLD.values())
    out = {"meta": {"n_conversations": len(HARD_KEYS), "labelled_policies": len(HARD_IDS),
                    "pool_size": len(pool), "gold_positives": gold_pos,
                    "context_tokens": args.context_tokens,
                    "multi_turn": sum(1 for t, _ in HARD_CONVERSATIONS if len(t) > 1)},
           "runs": {}}
    print(f"hard tier: {len(HARD_KEYS)} conversations, {len(HARD_IDS)} labelled policies, "
          f"pool {len(pool)}, {gold_pos} gold positives")

    for spec in args.models:
        prov = build(spec)
        out["runs"][spec] = {}
        for arm in args.arms:
            print(f"\n=== {spec} | {arm} | pool={len(pool)} ===", flush=True)
            r = await run_arm(prov, arm, pool, ctx, args.concurrency)
            print(f"  P {r['precision']:.3f}  R {r['recall']:.3f}  F1 {r['f1']:.3f}  "
                  f"exact {r['exact_set_match']:.3f}  padFP {r['padding_false_positives']}  "
                  f"({r['calls_per_turn']} calls/turn, {r['stats']['errors']} err)", flush=True)
            out["runs"][spec][arm] = r
            with open(os.path.join(here, args.out), "w") as f:
                json.dump(out, f, indent=2)

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())

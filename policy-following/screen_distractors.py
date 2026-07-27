"""Validate that no distractor policy legitimately fires on a labelled query.

The distractor set only works as a scaling axis if every distractor is truly
false for all 47 queries. This screens all (query, distractor) pairs one pair
per call — the most sensitive setting — and prints every fire for manual
review. Any distractor with a defensible fire must be removed from the set,
otherwise the gold labels would be wrong.

Writes distractor_screen.json.
"""
import asyncio
import json
import os
import sys

import tiktoken

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_gating import ONE_SCHEMA, ONE_TMPL, PREAMBLE
from dataset import QUERY_TEXTS, context_block
from dataset_hard import HARD_KEYS
from distractors import DISTRACTORS
from providers import build, gather_limited

ENC = tiktoken.get_encoding("o200k_base")


async def main():
    args = [a for a in sys.argv[1:] if a != "--hard"]
    hard = "--hard" in sys.argv
    model = args[0] if args else "anthropic:claude-haiku-4-5-20251001"
    prov = build(model)
    # no padding: screening is about label validity, not context effects
    ctx = ""

    jobs, meta = [], []
    targets = HARD_KEYS if hard else QUERY_TEXTS
    for q in targets:
        for pid, ptype, cond in DISTRACTORS:
            sysp = ONE_TMPL.format(context=ctx, preamble=PREAMBLE, pid=pid,
                                   ptype=ptype, condition=cond)
            jobs.append(prov.call(sysp, q, ONE_SCHEMA, True))
            meta.append((q, pid, cond))

    print(f"screening {len(jobs)} (query, distractor) pairs on {model}", flush=True)
    res = await gather_limited(jobs, 16)

    fires, errors = [], 0
    for (q, pid, cond), r in zip(meta, res):
        if r.error:
            errors += 1
            continue
        if isinstance(r.data, dict) and r.data.get("applies") is True:
            fires.append({"query": q, "distractor": pid, "condition": cond})

    print(f"\n{len(fires)} fires, {errors} errors\n")
    by_d = {}
    for f in fires:
        by_d.setdefault(f["distractor"], []).append(f["query"])
    for pid, qs in sorted(by_d.items(), key=lambda x: -len(x[1])):
        print(f"  {pid}  ({len(qs)})")
        for q in qs:
            print(f"      <- {q}")

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "distractor_screen_hard.json" if hard else "distractor_screen.json"), "w") as f:
        json.dump({"model": model, "pairs": len(jobs), "errors": errors,
                   "fires": fires}, f, indent=2)
    print("\nwrote screen file")


if __name__ == "__main__":
    asyncio.run(main())

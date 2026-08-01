"""Does a DIFFERENT model spot the rules a reply broke?

bench_rules.py established two points about self-review: a model grading its
own turn (the draft sitting in the assistant role) is close to blind, and the
same model grading the identical text presented as somebody else's does far
better. Both arms used ONE model in both roles, so they show that authorship
framing matters — not that a second model helps.

This closes that gap. Take the drafts each model produced at 24 rules, keep the
deterministic checker's verdict as ground truth, and have every model audit
every draft. The diagonal is same-model review; the off-diagonal is a genuine
second opinion.

Writes cross_audit_results.json.
"""
import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from providers import build, gather_limited
from rules import RULE_IDS, RULE_TEXT, check_all

AUDIT_SCHEMA = {
    "type": "object",
    "properties": {"violated_rule_ids": {"type": "array", "items": {"type": "string"}}},
    "required": ["violated_rule_ids"], "additionalProperties": False,
}
AUDIT_SYS = ("You are a compliance checker. The user gives a set of numbered rules and a "
             "reply. Return the ids of every rule the reply breaks. Be strict and literal.")


def rule_block(rids):
    return "\n".join(f"{i+1}. [{r}] {RULE_TEXT[r]}" for i, r in enumerate(rids))


def score(truth, flagged):
    """Recall is the number that matters: a violation the auditor misses ships."""
    tp = fn = fp = 0
    for real, got in zip(truth, flagged):
        real, got = set(real), set(got)
        tp += len(real & got)
        fn += len(real - got)
        fp += len(got - real)
    return {"caught": tp, "missed": fn, "false_alarms": fp,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "real_violations": tp + fn}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafts", default="rules_results_r16fix.json")
    ap.add_argument("--arm", default="single@24")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--out", default="cross_audit_results.json")
    args = ap.parse_args()

    src = json.load(open(args.drafts))
    rids = RULE_IDS[:24]
    block = rule_block(rids)

    # Ground truth is recomputed here rather than trusted from the stored run,
    # so a stale label in the source file cannot quietly become a finding.
    corpus = {}
    for spec, arms in src["runs"].items():
        samples = arms[args.arm]["samples"]
        texts = [s["final"] for s in samples]
        truth = [[k for k, v in check_all(t, rids).items() if v is False] for t in texts]
        corpus[spec] = {"texts": texts, "truth": truth}
        print(f"{spec:34s} {len(texts):2d} drafts, "
              f"{sum(len(v) for v in truth):2d} real violations")

    out = {"meta": {"source": args.drafts, "arm": args.arm, "n_rules": 24,
                    "authors": list(corpus)}, "pairs": {}}

    for auditor_spec in corpus:
        prov = build(auditor_spec)
        out["pairs"][auditor_spec] = {}
        for author_spec, data in corpus.items():
            if not any(data["truth"]):
                print(f"  skip {author_spec} (no violations to find)")
                continue
            res = await gather_limited(
                [prov.call(AUDIT_SYS, f"RULES\n{block}\n\nREPLY\n{t}", AUDIT_SCHEMA, True)
                 for t in data["texts"]], args.concurrency)
            flagged = [[v for v in ((r.data or {}).get("violated_rule_ids") or []) if v in rids]
                       if isinstance(r.data, dict) else [] for r in res]
            s = score(data["truth"], flagged)
            s["errors"] = sum(1 for r in res if r.error)
            out["pairs"][auditor_spec][author_spec] = s
            same = "SELF " if auditor_spec == author_spec else "cross"
            print(f"  {same} auditor={auditor_spec.split(':')[1][:22]:24s} "
                  f"author={author_spec.split(':')[1][:22]:24s} "
                  f"caught {s['caught']}/{s['real_violations']} "
                  f"recall {s['recall']} false_alarms {s['false_alarms']}")

    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())

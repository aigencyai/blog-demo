"""How many response rules survive one call — and what recovers the rest.

Rules are the `response` policy type: they shape how the assistant speaks.
Every rule in rules.py has a deterministic checker, so compliance is measured
without an LLM judge.

Arms:
  single        one call, all R rules in the system prompt (the usual approach)
  audit_repair  single, then ONE call that re-checks all R rules and rewrites
  split_audit   single, then ONE audit call PER RULE, then one repair call
                carrying only the rules found violated

split_audit is the response-side analogue of what raglib does on the gating
side: give each rule its own call rather than asking one call to hold them all.

Writes rules_results.json.
"""
import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from providers import build, gather_limited
from rules import RULE_IDS, RULE_TEXT, SCENARIOS, check_all

CORE = RULE_IDS[:4]   # present at every level; used as the interference control

BASE = ("You are the shopping assistant for Solstice Optics, an online eyewear retailer. "
        "Answer the shopper using only the products supplied below.")

TEXT_SCHEMA = {
    "type": "object",
    "properties": {"reply": {"type": "string"}},
    "required": ["reply"], "additionalProperties": False,
}
AUDIT_SCHEMA = {
    "type": "object",
    "properties": {"violated_rule_ids": {"type": "array", "items": {"type": "string"}}},
    "required": ["violated_rule_ids"], "additionalProperties": False,
}
ONE_AUDIT_SCHEMA = {
    "type": "object",
    "properties": {"violated": {"type": "boolean"}},
    "required": ["violated"], "additionalProperties": False,
}


def rule_block(rids):
    return "\n".join(f"{i+1}. [{r}] {RULE_TEXT[r]}" for i, r in enumerate(rids))


def gen_system(rids):
    return (f"{BASE}\n\nRESPONSE RULES ({len(rids)} total) — every one of these is mandatory.\n"
            f"{rule_block(rids)}\n\nWrite the reply so that every rule above holds.")


def user_turn(query, products):
    lines = "\n".join(
        f"- {p['title']}, ${p['price']}, {p['colour']}" for p in products)
    return f"Shopper: {query}\n\nRetrieved products:\n{lines}"


async def generate(prov, rids, scen):
    q, prods = scen
    r = await prov.call(gen_system(rids), user_turn(q, prods), TEXT_SCHEMA, True)
    txt = (r.data or {}).get("reply", "") if isinstance(r.data, dict) else ""
    return txt, r


async def repair(prov, rids, scen, draft, violated):
    q, prods = scen
    if not violated:
        return draft, None
    sysp = (f"{BASE}\n\nRESPONSE RULES ({len(rids)} total) — every one is mandatory.\n"
            f"{rule_block(rids)}\n\nA draft reply broke the rules listed by the user. "
            f"Rewrite it so every rule holds. Change as little as possible.")
    usr = (f"{user_turn(q, prods)}\n\nDraft reply:\n{draft}\n\n"
           f"Rules broken: {', '.join(violated)}")
    r = await prov.call(sysp, usr, TEXT_SCHEMA, True)
    txt = (r.data or {}).get("reply", draft) if isinstance(r.data, dict) else draft
    return txt, r


def _audit_vs_checker(drafts, flagged, rids):
    """Does the LLM auditor see what the deterministic checker sees?

    Recall is the number that matters: a violation the auditor misses is one the
    repair pass is never told about, so it survives to the final reply.
    """
    tp = fp = fn = 0
    for draft, flag in zip(drafts, flagged):
        real = {k for k, v in check_all(draft, rids).items() if v is False}
        got = set(flag)
        tp += len(real & got); fp += len(got - real); fn += len(real - got)
    return {"auditor_caught": tp, "auditor_missed": fn, "auditor_false_alarms": fp,
            "auditor_recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "auditor_precision": round(tp / (tp + fp), 4) if tp + fp else None}


async def run_arm(prov, arm, rids, scenarios, concurrency):
    t0 = time.perf_counter()
    audit_quality = None
    stats = {"calls": 0, "in_tok": 0, "out_tok": 0, "cached_tok": 0, "errors": 0}

    def acc(r):
        if r is None:
            return
        stats["calls"] += 1
        stats["in_tok"] += r.in_tok
        stats["out_tok"] += r.out_tok
        stats["cached_tok"] += r.cached_tok
        stats["errors"] += int(bool(r.error))

    drafts = await gather_limited(
        [generate(prov, rids, s) for s in scenarios], concurrency)
    for _, r in drafts:
        acc(r)
    texts = [t for t, _ in drafts]

    if arm == "single":
        finals = texts
    elif arm == "audit_repair":
        audit_sys = ("You are a compliance checker. The user gives a set of numbered rules and a "
                     "reply. Return the ids of every rule the reply breaks. Be strict and literal.")
        audits = await gather_limited([
            prov.call(audit_sys,
                      f"RULES\n{rule_block(rids)}\n\nREPLY\n{t}", AUDIT_SCHEMA, True)
            for t in texts], concurrency)
        for r in audits:
            acc(r)
        viols = [[v for v in ((a.data or {}).get("violated_rule_ids") or []) if v in rids]
                 if isinstance(a.data, dict) else [] for a in audits]
        audit_quality = _audit_vs_checker(texts, viols, rids)
        reps = await gather_limited([
            repair(prov, rids, s, t, v) for s, t, v in zip(scenarios, texts, viols)],
            concurrency)
        for _, r in reps:
            acc(r)
        finals = [t for t, _ in reps]
    elif arm == "checker_repair":
        # Detection by the deterministic checkers rather than by the model, then
        # one repair call. Self-audit is blind to structural violations — a reply
        # containing "Fit tip:" mid-sentence reads as compliant to an LLM but
        # fails "include one LINE that begins with". Anything you can assert in
        # code should be asserted in code; the model is only asked to fix.
        viols = [[k for k, v in check_all(t, rids).items() if v is False] for t in texts]
        reps = await gather_limited([
            repair(prov, rids, s, t, v) for s, t, v in zip(scenarios, texts, viols)],
            concurrency)
        for _, r in reps:
            acc(r)
        finals = [t for t, _ in reps]
    elif arm == "checker_loop":
        # checker_repair, iterated: re-check after each repair and go again while
        # anything still fails (max 3 rounds). One repair round is not always
        # enough — a rewrite can fix the flagged rule and break another — but the
        # checker costs nothing, so looping is cheap: only still-broken replies
        # are resubmitted.
        finals = list(texts)
        for _ in range(3):
            viols = [[k for k, v in check_all(t, rids).items() if v is False]
                     for t in finals]
            if not any(viols):
                break
            reps = await gather_limited([
                repair(prov, rids, s, t, v)
                for s, t, v in zip(scenarios, finals, viols)], concurrency)
            for _, r in reps:
                acc(r)
            finals = [t for t, _ in reps]
    elif arm == "split_audit":
        one_sys = ("You are a compliance checker. The user gives ONE rule and a reply. "
                   "Return whether the reply breaks that rule. Be strict and literal.")
        jobs, idx = [], []
        for i, t in enumerate(texts):
            for rid in rids:
                jobs.append(prov.call(
                    one_sys, f"RULE [{rid}] {RULE_TEXT[rid]}\n\nREPLY\n{t}",
                    ONE_AUDIT_SCHEMA, True))
                idx.append((i, rid))
        outs = await gather_limited(jobs, concurrency)
        for r in outs:
            acc(r)
        viols = [[] for _ in texts]
        for (i, rid), r in zip(idx, outs):
            if isinstance(r.data, dict) and r.data.get("violated") is True:
                viols[i].append(rid)
        audit_quality = _audit_vs_checker(texts, viols, rids)
        reps = await gather_limited([
            repair(prov, rids, s, t, v) for s, t, v in zip(scenarios, texts, viols)],
            concurrency)
        for _, r in reps:
            acc(r)
        finals = [t for t, _ in reps]
    else:
        raise ValueError(arm)

    stats["wall_s"] = round(time.perf_counter() - t0, 1)

    per_rule = {r: {"ok": 0, "bad": 0, "na": 0} for r in rids}
    fully = 0
    checked = complied = 0
    core_checked = core_complied = 0
    samples = []
    for scen, draft, final in zip(scenarios, texts, finals):
        res = check_all(final, rids)
        allok = True
        for rid, v in res.items():
            if v is None:
                per_rule[rid]["na"] += 1
                continue
            per_rule[rid]["ok" if v else "bad"] += 1
            checked += 1
            complied += int(bool(v))
            # CORE control: the same four rules are present at every level, so
            # their compliance isolates interference from rule difficulty.
            if rid in CORE:
                core_checked += 1
                core_complied += int(bool(v))
            allok &= bool(v)
        fully += int(allok)
        samples.append({"query": scen[0], "draft": draft, "final": final,
                        "violations": sorted(k for k, v in res.items() if v is False)})

    return {
        "rule_compliance": round(complied / checked, 4) if checked else None,
        "core4_compliance": round(core_complied / core_checked, 4) if core_checked else None,
        "fully_compliant_replies": round(fully / len(scenarios), 4),
        "checked_rule_instances": checked,
        "per_rule": per_rule, "stats": stats,
        "calls_per_reply": round(stats["calls"] / len(scenarios), 2),
        "audit_quality": audit_quality,
        "samples": samples,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[
        "anthropic:claude-opus-5",
        "anthropic:claude-haiku-4-5-20251001",
        "openai:gpt-5.6-sol",
        "google:gemini-2.5-flash",
    ])
    ap.add_argument("--levels", nargs="+", type=int, default=[4, 8, 12, 16, 20, 24])
    ap.add_argument("--arms", nargs="+", default=["single"])
    ap.add_argument("--full-arms", nargs="+", default=["audit_repair", "split_audit"],
                    help="arms run only at the largest rule count")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default="rules_results.json")
    args = ap.parse_args()

    out = {"meta": {"n_scenarios": len(SCENARIOS), "levels": args.levels,
                    "all_rules": RULE_IDS}, "runs": {}}
    here = os.path.dirname(os.path.abspath(__file__))

    for spec in args.models:
        prov = build(spec)
        out["runs"][spec] = {}
        for lvl in args.levels:
            rids = RULE_IDS[:lvl]
            arms = list(args.arms)
            if lvl == max(args.levels):
                arms += list(args.full_arms)
            for arm in arms:
                key = f"{arm}@{lvl}"
                print(f"\n=== {spec} | {key} ===", flush=True)
                res = await run_arm(prov, arm, rids, SCENARIOS, args.concurrency)
                print(f"  rule compliance {res['rule_compliance']:.3f}  "
                      f"core-4 {res['core4_compliance']:.3f}  "
                      f"fully-compliant replies {res['fully_compliant_replies']:.3f}  "
                      f"({res['calls_per_reply']} calls/reply, "
                      f"{res['stats']['errors']} err)", flush=True)
                out["runs"][spec][key] = res
                with open(os.path.join(here, args.out), "w") as f:
                    json.dump(out, f, indent=2)

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())

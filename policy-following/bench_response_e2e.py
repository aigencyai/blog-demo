"""Response policies end to end: does the condition fire, AND does the reply carry the effect?

Three arms, all producing one reply per shopper message:

  one_prompt    All 20 policies (condition + effect) in the system prompt. The model
                decides which apply and writes the reply in a single call.
  gate_inject   Production shape: one gating call decides which conditions hold, then a
                second call writes the reply with ONLY the fired effects injected as
                instructions — it never sees the conditions or the other 19 policies.
  gate_k1       Same, but the gate is one call per policy (raglib's actual design)
                instead of one call over the whole list.

Scoring separates the two ways this chain breaks:

  effect recall     of the effects that should be in the reply, how many are
  effect precision  of the effects present, how many belong (a reply that recites the
                    warranty terms unprompted is a real failure — it is noise the brand
                    did not ask for on this turn)
  turn perfect      every required effect present and no spurious ones

gate_inject can only apply an effect its gate fired, so gate recall is a hard ceiling on
effect recall — the two failure modes compound, which is the point of measuring end to end.

Writes response_e2e_results.json.
"""
import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_response import (PRODUCTS, RP_COND, RP_EFFECT, RP_IDS, SCEN_GOLD,
                              SCEN_TEXTS, check_effects)
from providers import build, gather_limited

BASE = ("You are the shopping assistant for Solstice Optics, an online eyewear retailer. "
        "Answer the shopper's message using the products below. Keep it under 120 words.")

TEXT_SCHEMA = {"type": "object", "properties": {"reply": {"type": "string"}},
               "required": ["reply"], "additionalProperties": False}
GATE_SCHEMA = {"type": "object",
               "properties": {p: {"type": "boolean"} for p in RP_IDS},
               "required": list(RP_IDS), "additionalProperties": False}
ONE_GATE_SCHEMA = {"type": "object", "properties": {"applies": {"type": "boolean"}},
                   "required": ["applies"], "additionalProperties": False}


def products_block():
    return "\n".join(f"- {p['title']}, ${p['price']:.2f}, {p['colour']}" for p in PRODUCTS)


def one_prompt_system():
    lines = "\n".join(
        f"{i+1}. WHEN {RP_COND[p]}\n   THEN {RP_EFFECT[p]}" for i, p in enumerate(RP_IDS))
    return (f"{BASE}\n\nBRAND POLICIES ({len(RP_IDS)} total). Each is a WHEN/THEN rule. Apply a "
            f"policy's THEN only if its WHEN holds for this message. Do not apply policies whose "
            f"WHEN does not hold.\n\n{lines}\n\nPRODUCTS\n{products_block()}")


def inject_system(fired):
    if fired:
        eff = "\n".join(f"- {RP_EFFECT[p]}" for p in fired)
        extra = f"\n\nYou must also do the following in this reply:\n{eff}"
    else:
        extra = ""
    return f"{BASE}{extra}\n\nPRODUCTS\n{products_block()}"


GATE_SYS = ("You are the policy gate for the Solstice Optics shopping assistant. For each policy "
            "below, decide whether its condition holds for the shopper's message. Return a "
            "decision for every policy id.\n\n" +
            "\n".join(f"- {p}: {RP_COND[p]}" for p in RP_IDS))

ONE_GATE_SYS = ("You are the policy gate for the Solstice Optics shopping assistant. Decide "
                "whether the following policy's condition holds for the shopper's message.\n\n"
                "Policy id: {pid}\nCondition: {cond}")


async def run_arm(prov, arm, conc):
    t0 = time.perf_counter()
    st = {"calls": 0, "errors": 0, "in_tok": 0, "out_tok": 0, "cached_tok": 0}

    def acc(r):
        st["calls"] += 1
        st["in_tok"] += r.in_tok
        st["out_tok"] += r.out_tok
        st["cached_tok"] += r.cached_tok
        st["errors"] += int(bool(r.error))

    gate_pred = {s: None for s in SCEN_TEXTS}

    if arm == "one_prompt":
        sysp = one_prompt_system()
        res = await gather_limited(
            [prov.call(sysp, f"Shopper: {s}", TEXT_SCHEMA, True) for s in SCEN_TEXTS], conc)
        for r in res:
            acc(r)
        replies = [(r.data or {}).get("reply", "") if isinstance(r.data, dict) else ""
                   for r in res]
    else:
        if arm == "gate_inject":
            gres = await gather_limited(
                [prov.call(GATE_SYS, f"Shopper: {s}", GATE_SCHEMA, True) for s in SCEN_TEXTS],
                conc)
            for r in gres:
                acc(r)
            fired = []
            for s, r in zip(SCEN_TEXTS, gres):
                f = {p for p in RP_IDS if isinstance(r.data, dict) and r.data.get(p) is True}
                fired.append(f)
                gate_pred[s] = sorted(f)
        else:  # gate_k1
            jobs, meta = [], []
            for s in SCEN_TEXTS:
                for p in RP_IDS:
                    jobs.append(prov.call(
                        ONE_GATE_SYS.format(pid=p, cond=RP_COND[p]),
                        f"Shopper: {s}", ONE_GATE_SCHEMA, True))
                    meta.append((s, p))
            gres = await gather_limited(jobs, conc)
            for r in gres:
                acc(r)
            acc_map = {s: set() for s in SCEN_TEXTS}
            for (s, p), r in zip(meta, gres):
                if isinstance(r.data, dict) and r.data.get("applies") is True:
                    acc_map[s].add(p)
            fired = [acc_map[s] for s in SCEN_TEXTS]
            for s in SCEN_TEXTS:
                gate_pred[s] = sorted(acc_map[s])

        rres = await gather_limited(
            [prov.call(inject_system(sorted(f)), f"Shopper: {s}", TEXT_SCHEMA, True)
             for s, f in zip(SCEN_TEXTS, fired)], conc)
        for r in rres:
            acc(r)
        replies = [(r.data or {}).get("reply", "") if isinstance(r.data, dict) else ""
                   for r in rres]

    st["wall_s"] = round(time.perf_counter() - t0, 1)

    # score: every reply is checked against EVERY policy's effect checker, so an
    # effect the brand did not ask for on this turn counts against precision.
    etp = efp = efn = 0
    gtp = gfp = gfn = 0
    perfect = 0
    rows = []
    for s, rep in zip(SCEN_TEXTS, replies):
        gold = SCEN_GOLD[s]
        present = {p for p, ok in check_effects(rep, RP_IDS).items() if ok}
        etp += len(gold & present); efp += len(present - gold); efn += len(gold - present)
        perfect += int(present == gold)
        if gate_pred[s] is not None:
            gp = set(gate_pred[s])
            gtp += len(gold & gp); gfp += len(gp - gold); gfn += len(gold - gp)
        rows.append({"message": s, "gold": sorted(gold),
                     "gate_fired": gate_pred[s],
                     "effects_present": sorted(present),
                     "missing": sorted(gold - present),
                     "spurious": sorted(present - gold),
                     "reply": rep})

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 1.0
        r = tp / (tp + fn) if tp + fn else 1.0
        return round(p, 4), round(r, 4), round(2 * p * r / (p + r), 4) if p + r else 0.0

    ep, er, ef1 = prf(etp, efp, efn)
    out = {"effect_precision": ep, "effect_recall": er, "effect_f1": ef1,
           "turn_perfect": round(perfect / len(SCEN_TEXTS), 4),
           "effect_tp": etp, "effect_fp": efp, "effect_fn": efn,
           "calls_per_turn": round(st["calls"] / len(SCEN_TEXTS), 2),
           "stats": st, "rows": rows}
    if gtp or gfp or gfn:
        gp_, gr_, gf_ = prf(gtp, gfp, gfn)
        out.update({"gate_precision": gp_, "gate_recall": gr_, "gate_f1": gf_})
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[
        "anthropic:claude-opus-5",
        "anthropic:claude-haiku-4-5-20251001",
        "openai:gpt-5.6-sol",
        "google:gemini-2.5-flash",
    ])
    ap.add_argument("--arms", nargs="+", default=["one_prompt", "gate_inject", "gate_k1"])
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default="response_e2e_results.json")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    gold_pos = sum(len(g) for g in SCEN_GOLD.values())
    out = {"meta": {"n_policies": len(RP_IDS), "n_scenarios": len(SCEN_TEXTS),
                    "gold_effects": gold_pos}, "runs": {}}
    print(f"{len(RP_IDS)} response policies (condition+effect) x {len(SCEN_TEXTS)} messages, "
          f"{gold_pos} effects that must appear")

    for spec in args.models:
        prov = build(spec)
        out["runs"][spec] = {}
        for arm in args.arms:
            print(f"\n=== {spec} | {arm} ===", flush=True)
            r = await run_arm(prov, arm, args.concurrency)
            g = (f"gate F1 {r['gate_f1']:.3f} | " if "gate_f1" in r else "")
            print(f"  {g}effect P {r['effect_precision']:.3f} R {r['effect_recall']:.3f} "
                  f"F1 {r['effect_f1']:.3f} | turn-perfect {r['turn_perfect']:.3f} "
                  f"({r['calls_per_turn']} calls/turn, {r['stats']['errors']} err)", flush=True)
            out["runs"][spec][arm] = r
            with open(os.path.join(here, args.out), "w") as f:
                json.dump(out, f, indent=2)

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())

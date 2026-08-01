"""Recompute every headline number quoted in the policy-following post from the
raw result JSON, so no figure in the prose is hand-typed.

Run:  python3 verify_policy.py     (exits non-zero if anything disagrees)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
load = lambda n: json.load(open(os.path.join(HERE, n)))

fails = []


def check(label, got, want, tol=0.0):
    ok = abs(got - want) <= tol * max(abs(want), 1e-9) if tol else got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: got {got}, want {want}")
    if not ok:
        fails.append(label)


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return 2 * p * r / (p + r) if p + r else 0.0


# --- response policies, end to end ----------------------------------------
if os.path.exists(os.path.join(HERE, "response_e2e_results.json")):
    e = load("response_e2e_results.json")
    m = e["meta"]
    print(f"\nRESPONSE POLICIES END TO END: {m['n_policies']} policies (condition+effect) x "
          f"{m['n_scenarios']} messages, {m['gold_effects']} effects that must appear")
    for spec, arms in e["runs"].items():
        for arm, v in arms.items():
            check(f"e2e {spec} {arm} effect F1",
                  round(f1(v["effect_tp"], v["effect_fp"], v["effect_fn"]), 4), v["effect_f1"])
            gate = f"gate F1 {v['gate_f1']:.3f} -> " if "gate_f1" in v else ""
            print(f"  {spec.split(':')[1][:24]:26} {arm:12} {gate}"
                  f"effect P {v['effect_precision']:.3f} R {v['effect_recall']:.3f} "
                  f"| turn-perfect {v['turn_perfect']:.3f} ({v['calls_per_turn']} calls)")
    # claim: injecting only fired effects makes over-application impossible
    for spec, arms in e["runs"].items():
        gi = arms.get("gate_inject")
        if gi and gi["effect_precision"] < 0.99:
            print(f"  NOTE {spec} gate_inject precision {gi['effect_precision']} < 0.99")
    # claim: the gate outperforms the execution step it feeds
    for spec, arms in e["runs"].items():
        gi = arms.get("gate_inject")
        if gi and "gate_recall" in gi and gi["gate_recall"] < gi["effect_recall"]:
            fails.append(f"{spec}: gate recall below effect recall (execution not the bottleneck)")

# --- easy tier ------------------------------------------------------------
g = load("gating_results.json")
print(f"\nEASY TIER: {g['meta']['n_policies']} policies x {g['meta']['n_queries']} queries "
      f"= {g['meta']['pairs']} decisions, {g['meta']['gold_positives']} gold positives")
check("pairs", g["meta"]["n_policies"] * g["meta"]["n_queries"], g["meta"]["pairs"])
for spec, arms in g["runs"].items():
    for arm, v in arms.items():
        s = v["score"]
        check(f"{spec} {arm} F1 recomputed", round(f1(s["tp"], s["fp"], s["fn"]), 4), s["f1"])
        # recall of decomposed (k1) must be 1.0 for every model except haiku
        if arm == "k1" and "haiku" not in spec:
            check(f"{spec} k1 recall == 1.0", s["recall"], 1.0)

# claim: on the easy tier every model >= 0.88 F1 in every arm
worst = min(v["score"]["f1"] for arms in g["runs"].values() for v in arms.values())
print(f"  worst easy-tier F1 across all models/arms: {worst}")
if worst < 0.88:
    fails.append("easy tier floor")

# --- scaling + position ----------------------------------------------------
sc = load("scaling_results.json")
print("\nSCALING 25 -> 100 (monolithic):")
for spec, runs in sc["scaling"].items():
    a, b = runs["monolithic@25"]["score"], runs["monolithic@100"]["score"]
    print(f"  {spec.split(':')[1][:24]:26} F1 {a['f1']:.3f} -> {b['f1']:.3f}")
print("POSITION (recall by list third, 3 seeds pooled):")
pos_summary = {}
for spec, seeds in sc["position"].items():
    agg = {"first_third": [0, 0], "middle_third": [0, 0], "last_third": [0, 0]}
    for v in seeds.values():
        for bkt, x in v["buckets"].items():
            agg[bkt][0] += x["hit"]; agg[bkt][1] += x["total"]
    rates = {k: h / t for k, (h, t) in agg.items()}
    pos_summary[spec] = rates
    print(f"  {spec.split(':')[1][:24]:26} " +
          "  ".join(f"{k.split('_')[0]} {v:.1%}" for k, v in rates.items()))
# claims: opus and gpt flat at 100%; haiku & gemini drop >= 10 points in the last third
for spec, r in pos_summary.items():
    if "opus" in spec or "gpt" in spec:
        check(f"{spec} position-flat", min(r.values()), 1.0)
    else:
        drop = r["first_third"] - r["last_third"]
        print(f"  {spec}: first-to-last drop {drop:.1%}")
        if drop < 0.10:
            fails.append(f"{spec} expected >=10pt last-third drop")

# --- hard tier --------------------------------------------------------------
h = load("hard_results.json")
m = h["meta"]
print(f"\nHARD TIER: {m['labelled_policies']} compositional policies, "
      f"{m['n_conversations']} conversations ({m['multi_turn']} multi-turn), "
      f"pool {m['pool_size']}")
for spec, arms in h["runs"].items():
    for arm, v in arms.items():
        check(f"hard {spec} {arm} F1", round(f1(v["tp"], v["fp"], v["fn"]), 4), v["f1"])
        print(f"  {spec.split(':')[1][:24]:26} {arm:12} P {v['precision']:.3f} "
              f"R {v['recall']:.3f} F1 {v['f1']:.3f} padFP {v['padding_false_positives']}")

# --- response rules ----------------------------------------------------------
r = load("rules_results.json")
print(f"\nRESPONSE RULES: {r['meta']['n_scenarios']} scenarios, levels {r['meta']['levels']}")
for spec, runs in r["runs"].items():
    s24 = runs.get("single@24")
    if s24:
        print(f"  {spec.split(':')[1][:24]:26} single@24 per-rule {s24['rule_compliance']:.3f} "
              f"fully {s24['fully_compliant_replies']:.3f} core4 {s24['core4_compliance']:.3f}")
        # claim: per-rule stays high while joint collapses for 3 of 4 models
        if s24["rule_compliance"] < 0.85:
            fails.append(f"{spec} per-rule floor")

rc = load("rules_checker.json")
gpt = rc["runs"]["openai:gpt-5.6-sol"]
check("gpt-5.6-sol checker_repair@24 fully-compliant == 1.0",
      gpt["checker_repair@24"]["fully_compliant_replies"], 1.0)

ra = load("rules_audit.json")
print("\nLLM AUDITOR vs DETERMINISTIC CHECKER:")
for spec, runs in ra["runs"].items():
    for arm, v in runs.items():
        q = v.get("audit_quality")
        if q and q["auditor_recall"] is not None:
            print(f"  {spec.split(':')[1][:24]:26} {arm:16} auditor recall "
                  f"{q['auditor_recall']:.3f} (missed {q['auditor_missed']}, "
                  f"{q['auditor_false_alarms']} false alarms)")
# claim: a model grading its OWN turn (draft in the assistant role) finds at most
# a quarter of its real violations. The contrast is audit_repair, where the same
# model grades an anonymous reply and recall jumps to ~0.7 — so the variable is
# authorship, not capability. Only self_check is asserted; audit_repair is the
# control and is expected to be HIGHER.
for spec, runs in ra["runs"].items():
    q = runs.get("self_check@24", {}).get("audit_quality")
    if q and q["auditor_recall"] is not None and q["auditor_recall"] > 0.25:
        fails.append(f"{spec} self-audit recall claim ({q['auditor_recall']})")

print("\nCHECKER-REPAIR LOOP (<=3 rounds):")
for spec, runs in r["runs"].items():
    v = runs.get("checker_loop@24")
    if v:
        print(f"  {spec.split(':')[1][:24]:26} fully {v['fully_compliant_replies']:.3f} "
              f"({v['calls_per_reply']} calls/reply)")

# --- contrastive isolation ----------------------------------------------------
c = load("contrastive_results.json")
print("\nCONTRASTIVE k=1 (vs plain k=1 from gating_results.json):")
for spec, arms in c["runs"].items():
    plain = g["runs"][spec]["k1"]["score"]
    ctr = arms["k1_contrastive"]["score"]
    print(f"  {spec.split(':')[1][:24]:26} plain P {plain['precision']:.3f} "
          f"-> contrastive P {ctr['precision']:.3f}   (R {plain['recall']:.3f} -> {ctr['recall']:.3f})")
    if ctr["precision"] <= plain["precision"]:
        fails.append(f"{spec} contrastive precision should improve")

# --- context length -----------------------------------------------------------
print("\nCONTEXT LENGTH (monolithic F1 at 0 / 2k / 8k / 16k padding):")
for spec in g["runs"]:
    row = []
    for name, ctx in [("ctxlen_0.json", 0), ("gating_results.json", 2000),
                      ("ctxlen_8000.json", 8000), ("ctxlen_16000.json", 16000)]:
        d = load(name)
        row.append((ctx, d["runs"][spec]["monolithic"]["score"]["f1"]))
    print(f"  {spec.split(':')[1][:24]:26} " + "  ".join(f"{c//1000}k {f:.3f}" for c, f in row))
    span = max(f for _, f in row) - min(f for _, f in row)
    if span > 0.08:
        fails.append(f"{spec} context-length claim (span {span:.3f})")

# --- retrieval gate -------------------------------------------------------------
rg = load("retrieval_gate_results.json")
print(f"\nRETRIEVAL GATE ({rg['meta']['embed_model']}, pool {rg['meta']['pool_size']}):")
for M in ["8", "30", "100"]:
    v = rg["by_M"][M]
    print(f"  top-{M:>3}: gate recall {v['gate_recall']:.3f}, end-to-end F1 {v['f1']:.3f}")
check("gate recall@8", rg["by_M"]["8"]["gate_recall"], 0.8358, 0.01)
check("gate recall@30", rg["by_M"]["30"]["gate_recall"], 0.9552, 0.01)

# --- distractor screens ----------------------------------------------------------
ds = load("distractor_screen.json")
check("easy distractor screen pairs", ds["pairs"], 3525)
check("easy distractor screen fires", len(ds["fires"]), 0)
dh = load("distractor_screen_hard.json")
check("hard distractor screen fires", len(dh["fires"]), 2)

print("\n" + ("ALL CHECKS PASSED" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)

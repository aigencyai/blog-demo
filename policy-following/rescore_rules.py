"""Re-score stored replies against the current checkers, without new API calls.

Compliance here is a deterministic function of the reply text, and every reply is
kept in the result JSON. So when a checker is corrected there is no need to
regenerate anything: re-run the checkers over the saved text and the metrics are
exact. Only auditor recall has to be re-measured live, because what the LLM
auditor flagged is a model output rather than a function of the reply.

Run:  python3 rescore_rules.py [file.json ...]      (rewrites in place)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rules import RULE_IDS, check_all

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = RULE_IDS[:4]
DEFAULT = ["rules_results.json", "rules_checker.json", "rules_loop.json", "rules_audit.json"]


def rescore(arm_obj, level):
    rids = RULE_IDS[:level]
    per_rule = {r: {"ok": 0, "bad": 0, "na": 0} for r in rids}
    checked = complied = core_checked = core_complied = fully = 0
    for smp in arm_obj["samples"]:
        res = check_all(smp["final"], rids)
        allok = True
        for rid, v in res.items():
            if v is None:
                per_rule[rid]["na"] += 1
                continue
            per_rule[rid]["ok" if v else "bad"] += 1
            checked += 1
            complied += int(bool(v))
            if rid in CORE:
                core_checked += 1
                core_complied += int(bool(v))
            allok &= bool(v)
        fully += int(allok)
        smp["violations"] = sorted(k for k, v in res.items() if v is False)
    n = len(arm_obj["samples"])
    arm_obj["rule_compliance"] = round(complied / checked, 4) if checked else None
    arm_obj["core4_compliance"] = round(core_complied / core_checked, 4) if core_checked else None
    arm_obj["fully_compliant_replies"] = round(fully / n, 4) if n else None
    arm_obj["checked_rule_instances"] = checked
    arm_obj["per_rule"] = per_rule
    return arm_obj


def main():
    files = sys.argv[1:] or DEFAULT
    for fn in files:
        path = os.path.join(HERE, fn)
        if not os.path.exists(path):
            print(f"skip {fn} (missing)")
            continue
        d = json.load(open(path))
        changed = []
        for spec, runs in d["runs"].items():
            for arm, v in runs.items():
                if "samples" not in v:
                    continue
                level = int(arm.split("@")[1])
                before = v.get("fully_compliant_replies")
                rescore(v, level)
                after = v["fully_compliant_replies"]
                if before != after:
                    changed.append(f"{spec.split(':')[1][:18]} {arm}: {before} -> {after}")
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
        print(f"{fn}: rescored" + (f", {len(changed)} arm(s) moved" if changed else ", no change"))
        for c in changed:
            print(f"    {c}")
        if "audit_quality" in json.dumps(d)[:0]:  # never true; kept for clarity
            pass
    print("\nNote: auditor recall in rules_audit.json is NOT rescored here — what the "
          "LLM auditor flagged is not recoverable from the reply text. Re-run bench_rules.py "
          "--full-arms audit_repair split_audit to refresh it.")


if __name__ == "__main__":
    main()

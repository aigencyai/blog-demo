"""Can a cheap embedding index decide which policies are even worth a call?

One call per policy is accurate but costs N calls a turn. Most policies are
irrelevant to most messages, so the obvious move is a prefilter: embed every
policy condition once, offline; at request time embed the message, keep the
top-M policies by cosine, and spend LLM calls only on those.

Two things decide whether that is safe:
  gate recall@M   - how often the triggered policies survive the prefilter.
                    Anything the gate drops can never fire, so this is a hard
                    ceiling on end-to-end recall.
  end-to-end F1   - the gate composed with the per-policy decisions.

No new LLM calls are needed: the k=1 decisions already exist in
gating_results.json (core policies) and distractor_screen.json (distractors),
so gating is applied as a post-hoc filter over decisions we already paid for.

Writes retrieval_gate_results.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import GOLD, POLICY_CONDITION, POLICY_IDS, QUERY_TEXTS
from distractors import DISTRACTORS
from providers import secret

HERE = os.path.dirname(os.path.abspath(__file__))
EMBED_MODEL = "text-embedding-3-small"
ALLCOND = {**POLICY_CONDITION, **{d[0]: d[2] for d in DISTRACTORS}}


def embed(texts):
    from openai import OpenAI
    client = OpenAI(api_key=secret("OPENAI_API_KEY"))
    out = []
    for i in range(0, len(texts), 128):
        r = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + 128])
        out += [d.embedding for d in r.data]
    return out


def cosine(a, b):
    num = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return num / (na * nb) if na and nb else 0.0


def load_k1_decisions(model_spec):
    """-> {query: set(policy_ids the k=1 arm said yes to)} across core + distractors."""
    yes = {q: set() for q in QUERY_TEXTS}
    g = json.load(open(os.path.join(HERE, "gating_results.json")))
    run = g["runs"].get(model_spec, {}).get("k1")
    if not run:
        raise SystemExit(f"no k1 run for {model_spec} in gating_results.json")
    for row in run["per_query"]:
        yes[row["query"]] |= set(row["pred"])
    scr = os.path.join(HERE, "distractor_screen.json")
    if os.path.exists(scr):
        s = json.load(open(scr))
        for f in s["fires"]:
            yes[f["query"]].add(f["distractor"])
    return yes


def main():
    model_spec = sys.argv[1] if len(sys.argv) > 1 else "anthropic:claude-opus-5"
    pool = list(POLICY_IDS) + [d[0] for d in DISTRACTORS]

    print(f"embedding {len(pool)} policy conditions + {len(QUERY_TEXTS)} queries "
          f"with {EMBED_MODEL}")
    pvecs = dict(zip(pool, embed([ALLCOND[p] for p in pool])))
    qvecs = dict(zip(QUERY_TEXTS, embed(QUERY_TEXTS)))

    ranked = {q: sorted(pool, key=lambda p: -cosine(qvecs[q], pvecs[p]))
              for q in QUERY_TEXTS}

    k1_yes = load_k1_decisions(model_spec)
    gold_total = sum(len(GOLD[q]) for q in QUERY_TEXTS)

    out = {"meta": {"embed_model": EMBED_MODEL, "pool_size": len(pool),
                    "decisions_from": model_spec, "gold_positives": gold_total},
           "by_M": {}}

    print(f"\n{'M':>4}  {'gate recall':>11}  {'llm calls/turn':>14}  "
          f"{'end P':>7}  {'end R':>7}  {'end F1':>7}")
    for M in [3, 5, 8, 10, 15, 20, 30, len(pool)]:
        hit = sum(len(GOLD[q] & set(ranked[q][:M])) for q in QUERY_TEXTS)
        tp = fp = fn = 0
        for q in QUERY_TEXTS:
            pred = k1_yes[q] & set(ranked[q][:M])
            tp += len(GOLD[q] & pred)
            fp += len(pred - GOLD[q])
            fn += len(GOLD[q] - pred)
        p = tp / (tp + fp) if tp + fp else 1.0
        r = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        out["by_M"][M] = {"gate_recall": round(hit / gold_total, 4),
                          "llm_calls_per_turn": M,
                          "precision": round(p, 4), "recall": round(r, 4),
                          "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}
        print(f"{M:>4}  {hit/gold_total:>11.3f}  {M:>14}  {p:>7.3f}  {r:>7.3f}  {f1:>7.3f}")

    # which gold policies does the gate rank worst? the failure mode matters more
    # than the average.
    worst = []
    for q in QUERY_TEXTS:
        for pid in GOLD[q]:
            worst.append((ranked[q].index(pid), q, pid))
    worst.sort(reverse=True)
    out["worst_ranked_gold"] = [{"rank": r, "query": q, "policy": p}
                                for r, q, p in worst[:12]]
    print("\nworst-ranked triggered policies (the gate would drop these first):")
    for r, q, p in worst[:8]:
        print(f"  rank {r:>3}  {p:28} <- {q}")

    with open(os.path.join(HERE, "retrieval_gate_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote retrieval_gate_results.json")


if __name__ == "__main__":
    main()

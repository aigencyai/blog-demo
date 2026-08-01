"""Is an expensive model cheaper as the reviewer than as the writer?

Writing is output-heavy: the model emits a whole reply. Reviewing is the
reverse, since the rulebook and one draft go in and a short list of rule ids
comes out. Every provider charges several times more for output than input, so
the same model can be unaffordable in one role and cheap in the other.

Prices below are list rates read off the providers' own pages, per 1M tokens.

Caveat on the input side: generation used prompt caching where the provider
supports it (588 of Opus's 851 input tokens per reply were cache hits), while
the review calls did not cache the rulebook. Caching it would cut review input
cost. The output column is unaffected, which is where the finding lives -- a
reasoning model spends nearly as many output tokens reviewing a reply as
writing one, so the "reviewing is output-light" argument fails exactly for the
models expensive enough to make it worth making.
"""
import json

# $ per 1M tokens, (input, output). Standard tier, no batch or cache discount.
#   gpt-5.6-sol      developers.openai.com/api/docs/pricing
#   claude-*         platform.claude.com/docs/en/about-claude/pricing
#   gemini-2.5-flash ai.google.dev/gemini-api/docs/pricing
PRICE = {
    "openai:gpt-5.6-sol": (5.00, 30.00),
    "anthropic:claude-opus-5": (5.00, 25.00),
    "anthropic:claude-haiku-4-5-20251001": (1.00, 5.00),
    "google:gemini-2.5-flash": (0.30, 2.50),
}
S = lambda s: s.split(":")[1].split("-2025")[0].replace("-sol", "")


def usd(spec, i, o):
    pi, po = PRICE[spec]
    return (i * pi + o * po) / 1e6


gen = json.load(open("rules_results_r16fix.json"))["runs"]
aud = json.load(open("cross_audit_results.json"))["pairs"]

print("Cost of ONE reply, in cents, at list prices\n")
print(f"{'model':18s}{'role':24s}{'in':>7s}{'out':>7s}{'cents':>9s}")
write, review = {}, {}
for spec, arms in gen.items():
    s = arms["single@24"]["stats"]
    i, o = s["in_tok"] / 12, s["out_tok"] / 12
    write[spec] = usd(spec, i, o)
    print(f"{S(spec):18s}{'writes the reply':24s}{i:7.0f}{o:7.0f}{write[spec]*100:9.4f}")

print()
for auditor, row in aud.items():
    vals = [v for v in row.values() if "in_tok_per_reply" in v and auditor not in row]
    vals = [v for k, v in row.items() if k != auditor and "in_tok_per_reply" in v]
    if not vals:
        continue
    i = sum(v["in_tok_per_reply"] for v in vals) / len(vals)
    o = sum(v["out_tok_per_reply"] for v in vals) / len(vals)
    review[auditor] = usd(auditor, i, o)
    print(f"{S(auditor):18s}{'reviews a reply':24s}{i:7.0f}{o:7.0f}{review[auditor]*100:9.4f}")

print("\nRatio, writing vs reviewing, same model:")
for spec in write:
    if spec in review:
        print(f"  {S(spec):18s} writing costs {write[spec]/review[spec]:5.1f}x its own review")

g, op = "openai:gpt-5.6-sol", "anthropic:claude-opus-5"
h = "anthropic:claude-haiku-4-5-20251001"
print("\nBuying the same compliance two ways, cents per reply:")
print(f"  gpt-5.6 writes it alone           {write[g]*100:.4f}")
for w in (op, h):
    combo = write[w] + review[g]
    print(f"  {S(w)} writes, gpt-5.6 reviews"
          f"{'':{max(0, 10-len(S(w)))}} {combo*100:.4f}"
          f"   ({combo/write[g]:.2f}x)")

# blog-demo

Reproduction code and raw results for aigency engineering blog posts. Each post gets a
directory; every figure in a post is recomputed from the JSON checked in here, so the
prose can't drift from the measurements.

## `kv-cache/` — What is a KV cache?

Measures what re-reading a prompt costs, and what changes when you keep the cache.

| Script | What it measures |
| --- | --- |
| `bench_kvcache.py` | Decode with vs without the cache; prefill cost vs prompt length; prefix reuse on a 12k-token prompt |
| `bench_multimodel.py` | Prefix reuse across Qwen2.5-0.5B / 1.5B and granite-3.2-2b, plus break-even |
| `bench_vision.py` | Image-token reuse on Qwen2-VL-2B — one image, four questions |
| `bench_api_cache.py` | Hosted **implicit** prompt caching (OpenAI / Anthropic / Gemini) |
| `bench_api_explicit.py` | Hosted **explicit** caching — Gemini `CachedContent`, Anthropic `cache_control` + TTL |
| `verify_kv.py` | Recomputes every headline number from the result JSON; exits non-zero on drift |

### Headline results

- Decode with the KV cache vs without: **7.7×** (48 tokens, granite-3.2-2b)
- Prefix reuse on a 12,000-token system prompt: **69×** (159.6 s → 2.3 s over four queries)
- Break-even on building the cache: **~1 query**
- A VLM prompt is **96.5%** image tokens (576 of 597) → **5.7×** from reusing them
- Gemini implicit caching reported **0** cached tokens; explicit `CachedContent` reported **9,467**

### Running it

```bash
pip install -r requirements.txt
cd kv-cache

# local, no API keys needed (downloads open weights from Hugging Face)
python3 bench_kvcache.py
python3 bench_multimodel.py
python3 bench_vision.py

# hosted models — needs OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY
python3 bench_api_cache.py
python3 bench_api_explicit.py

# check the numbers the post quotes
python3 verify_kv.py
```

Local runs are fp16 on Apple MPS and fall back to CPU. Absolute times are laptop-GPU
numbers — datacenter accelerators are far faster; the *shape* of the curves is the point.

### Notes

- The API scripts read credentials from the environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
  `GEMINI_API_KEY`). Set `GCP_PROJECT` instead to pull them from Google Secret Manager.
- `bench_vision.py` expects a product-page screenshot; point `shot` at any image.
- No credentials are stored in this repo — only the names of the secrets.

## `policy-following/` — How many policies can a model follow?

Measures how much of a brand's policy set (pin / boost / filter / exclude / response
conditions) survives a single prompt, across four models and four hosted APIs.

| Script | What it measures |
| --- | --- |
| `dataset_response.py` | **20 `response` policies as condition + effect + deterministic checker, 30 shopper messages** (the set the post is built on) |
| `bench_response_e2e.py` | End to end: one prompt vs gate→inject vs gate-per-policy; scores effect precision/recall and turn-perfect |
| `dataset.py` | 25 policies + 47 labelled shopper messages (mixed-type gating, not covered in the post) |
| `dataset_hard.py` | 20 compositional policies over 30 multi-turn conversations (hard tier) |
| `distractors.py` / `screen_distractors.py` | 73 irrelevant padding policies, empirically screened to zero false triggers |
| `rules.py` | 24 mechanically-checkable response rules (tone, formatting, brand voice) |
| `bench_gating.py` | Gating strategies: one prompt (free list / forced boolean), sharded, one-call-per-policy, contrastive |
| `bench_scaling.py` | Policy-count scaling (25→100) and position sensitivity within the list |
| `bench_hard.py` | The hard tier under the same gating strategies |
| `bench_rules.py` | Response-rule compliance as rule count grows, plus `self_check` (model reviews its own turn), `audit_repair`, `split_audit`, `checker_repair`, `checker_loop` |
| `rescore_rules.py` | Re-scores stored replies against current checkers with no new API calls |
| `bench_retrieval_gate.py` | Embedding prefilter over the policy list — recall ceiling vs LLM calls saved |
| `bench_judge.py` | Is LLM-as-judge reliable? Independent-judge recall vs self-audit, inter-judge and self-consistency agreement on uncheckable semantic rules |
| `verify_policy.py` | Recomputes every headline number from the result JSON; exits non-zero on drift |

### Headline results

- Response policies end to end: **effect recall 1.000 in 10 of 12 runs** — models essentially never drop
  an instruction they were handed. Nearly every failure is *over-application*.
- Gating (decide conditions, then inject only the fired effects) makes over-application structurally
  impossible: **haiku 0.867 → 1.000 turn-perfect** with one extra call, matching opus.
- One gate call *per policy* is the worst option: 21 calls/turn, precision 0.795–0.914.
- Mixed-type gating (not in the post): 25 policies in one prompt scores **F1 0.955–0.985** on every model.
- 100 policies, small models: recall drops to **79.7%** in the last third of the list (not the
  middle — the classic "lost in the middle" effect doesn't appear here).
- 24 response rules: per-rule compliance stays **93–98%**, but essentially **0 of 12** replies satisfied
  every rule at once for 3 of 4 models from 16 rules on — until a code checker + repair loop reached **100%**
  for opus and gpt-5.6-sol (and still 0% for haiku/gemini-flash, which can't execute the repair).
- Asked to review its OWN turn (draft in the assistant role), a model caught **0.00–0.07** of its
  real violations — and every model did *worse* on its own text than on identical text handed to a
  fresh call (0.00–0.22). An independent judge reaches **0.66–0.71**; code reaches **1.00**.
- Auditor flagged-sets are persisted per reply, so every recall figure recomputes from the JSON.
- Two of our own checkers were wrong and were corrected (accepting "thirty-day" for a 30-day window;
  not treating "prevent slipping" as a medical claim). All results here are from the fixed versions —
  each checker now has probes for a compliant paraphrase, a compliant disclaimer and a real violation.
- Two independent judges agreed **96.9%** of the time on 8 semantic (uncheckable) rules — but
  split on the one rule that actually required judgment.

### Running it

```bash
pip install -r requirements.txt
cd policy-following

# needs OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY (or GCP_PROJECT)
python3 screen_distractors.py anthropic:claude-opus-5
python3 bench_response_e2e.py      # the main experiment
python3 bench_gating.py
python3 bench_scaling.py
python3 bench_hard.py
python3 bench_rules.py
python3 bench_judge.py
python3 bench_retrieval_gate.py anthropic:claude-opus-5   # needs OPENAI_API_KEY for embeddings

# check the numbers the post quotes
python3 verify_policy.py
```

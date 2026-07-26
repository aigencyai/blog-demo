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

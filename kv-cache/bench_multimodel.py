"""KV prefix reuse across open models, plus image-token reuse on a VLM.

Part A (text): for each model, time answering N queries against a shared static
prefix two ways — recompute the whole prompt per query (what a stateless call
does) vs. build the prefix KV once and feed only the query tokens.

Part B (vision, the retrieval case): a product image's tokens are
query-independent. Encode the image once, keep its KV, then answer N text
queries against it — versus re-encoding the image for every query.

Writes multimodel_results.json.
"""
import json
import os
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
PREFIX_TOKENS = 4096          # keep runtime sane on MPS; scaling curve covers larger
QUERIES = [
    "Do you have polarized aviators under $200?",
    "What is your return window?",
    "Show me something similar to RB3025 but cheaper.",
    "Do these ship to Germany?",
]
UNIT = ("You are a shopping assistant for an eyewear catalog. Only answer questions about products, "
        "orders, and store policy. Never invent a product that is not in the catalog. ")


def sync():
    if DEV == "mps":
        torch.mps.synchronize()


def clock():
    sync()
    return time.perf_counter()


results = {"meta": {"device": DEV, "dtype": "float16", "prefix_tokens": PREFIX_TOKENS,
                    "n_queries": len(QUERIES)}, "text_models": {}, "vision": {}}

# ---------------------------------------------------------------- Part A: text
TEXT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "ibm-granite/granite-3.2-2b-instruct",
]

from transformers import AutoModelForCausalLM, AutoTokenizer

for mid in TEXT_MODELS:
    print(f"\n=== {mid} ===")
    try:
        tok = AutoTokenizer.from_pretrained(mid)
        model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16,
                                                     low_cpu_mem_usage=True).to(DEV).eval()
        cfg = model.config
        hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        kvh = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        bpt = 2 * cfg.num_hidden_layers * kvh * hd * 2

        reps = PREFIX_TOKENS // len(tok(UNIT).input_ids) + 1
        prefix = tok(UNIT * reps, return_tensors="pt").input_ids[:, :PREFIX_TOKENS].to(DEV)

        with torch.inference_mode():
            model(prefix[:, :16], use_cache=True)  # warm

            recompute_s, reuse_s = 0.0, 0.0
            for q in QUERIES:
                qids = tok(q, return_tensors="pt").input_ids.to(DEV)

                full = torch.cat([prefix, qids], dim=1)
                t0 = clock(); o = model(full, use_cache=True); recompute_s += clock() - t0
                del o

                past = model(prefix, use_cache=True).past_key_values
                t0 = clock(); o = model(qids, past_key_values=past, use_cache=True); reuse_s += clock() - t0
                del o, past

            # one-time prefix build cost (amortised across queries)
            t0 = clock(); o = model(prefix, use_cache=True); build_s = clock() - t0
            del o

        results["text_models"][mid] = {
            "layers": cfg.num_hidden_layers, "kv_heads": kvh, "head_dim": hd,
            "kb_per_token": round(bpt / 1024, 1),
            "prefix_kv_mb": round(bpt * PREFIX_TOKENS / 1024 / 1024, 1),
            "prefix_build_s": round(build_s, 3),
            "recompute_total_s": round(recompute_s, 3),
            "reuse_total_s": round(reuse_s, 3),
            "speedup": round(recompute_s / reuse_s, 1),
            "breakeven_queries": round(build_s / max(recompute_s / len(QUERIES) - reuse_s / len(QUERIES), 1e-9), 2),
        }
        r = results["text_models"][mid]
        print(f"  {r['kb_per_token']} KB/tok | prefix KV {r['prefix_kv_mb']} MB | build {r['prefix_build_s']}s")
        print(f"  recompute {recompute_s:.2f}s vs reuse {reuse_s:.2f}s -> {r['speedup']}x")
        del model
        if DEV == "mps":
            torch.mps.empty_cache()
    except Exception as e:
        print(f"  FAILED {type(e).__name__}: {str(e)[:160]}")
        results["text_models"][mid] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}

# ------------------------------------------------------------- Part B: vision
VLM = "Qwen/Qwen2-VL-2B-Instruct"
print(f"\n=== {VLM} (image-token reuse) ===")
try:
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from PIL import Image

    proc = AutoProcessor.from_pretrained(VLM)
    vlm = Qwen2VLForConditionalGeneration.from_pretrained(VLM, dtype=torch.float16,
                                                          low_cpu_mem_usage=True).to(DEV).eval()
    shot = ("../../data/pdp_benchmark/pdp-ray-ban-rb3025-aviator-gradient-8053672087901/screenshot_00.png")
    shot = os.path.join(os.path.dirname(os.path.abspath(__file__)), shot)
    img = Image.open(shot).convert("RGB")
    img = img.crop((0, 0, img.width, min(img.height, 1400))).resize((672, 672))

    IMG_QUERIES = ["What is the price?", "What colour is the frame?",
                   "Is it polarized?", "What is the product name?"]

    def build_inputs(question, with_image=True):
        content = ([{"type": "image"}] if with_image else []) + [{"type": "text", "text": question}]
        text = proc.apply_chat_template([{"role": "user", "content": content}],
                                        tokenize=False, add_generation_prompt=True)
        return proc(text=[text], images=[img] if with_image else None, return_tensors="pt").to(DEV)

    with torch.inference_mode():
        warm = build_inputs(IMG_QUERIES[0]); vlm(**warm, use_cache=True)

        # how many tokens does the image itself occupy?
        probe = build_inputs("x")
        n_img_tokens = int((probe.input_ids == vlm.config.image_token_id).sum()) \
            if hasattr(vlm.config, "image_token_id") else None

        recompute_s = 0.0
        for q in IMG_QUERIES:
            inp = build_inputs(q)
            t0 = clock(); o = vlm(**inp, use_cache=True); recompute_s += clock() - t0
            del o

        # reuse: encode image+preamble once, then only the question tokens are new
        base = build_inputs("")                     # image + template, no question
        t0 = clock(); base_out = vlm(**base, use_cache=True); img_build_s = clock() - t0
        base_past = base_out.past_key_values
        seen = base.input_ids.shape[1]

        reuse_s = 0.0
        for q in IMG_QUERIES:
            qids = proc.tokenizer(q, return_tensors="pt").input_ids.to(DEV)
            past = vlm(**base, use_cache=True).past_key_values   # fresh copy per query
            t0 = clock(); o = vlm(input_ids=qids, past_key_values=past, use_cache=True); reuse_s += clock() - t0
            del o, past

    cfgv = vlm.config.text_config if hasattr(vlm.config, "text_config") else vlm.config
    hdv = getattr(cfgv, "head_dim", None) or cfgv.hidden_size // cfgv.num_attention_heads
    kvhv = getattr(cfgv, "num_key_value_heads", cfgv.num_attention_heads)
    bptv = 2 * cfgv.num_hidden_layers * kvhv * hdv * 2

    results["vision"] = {
        "model": VLM, "image": "Ray-Ban PDP screenshot (672x672 crop)",
        "prompt_tokens_with_image": int(base.input_ids.shape[1]),
        "image_tokens": n_img_tokens,
        "kb_per_token": round(bptv / 1024, 1),
        "image_kv_mb": round(bptv * seen / 1024 / 1024, 1),
        "image_encode_s": round(img_build_s, 3),
        "n_queries": len(IMG_QUERIES),
        "recompute_total_s": round(recompute_s, 3),
        "reuse_total_s": round(reuse_s, 3),
        "speedup": round(recompute_s / reuse_s, 1),
    }
    v = results["vision"]
    print(f"  image occupies {v['image_tokens']} of {v['prompt_tokens_with_image']} prompt tokens "
          f"-> {v['image_kv_mb']} MB KV")
    print(f"  re-encode per query {recompute_s:.2f}s vs reuse image KV {reuse_s:.2f}s -> {v['speedup']}x")
except Exception as e:
    import traceback; traceback.print_exc()
    results["vision"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "multimodel_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nwrote multimodel_results.json")

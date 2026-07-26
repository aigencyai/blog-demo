"""Image-token KV reuse on a VLM — the retrieval case.

A product image's tokens are query-independent: nothing about the image changes
when the question changes. So encode the image once, keep its KV, and let each
query attend to it — instead of re-encoding the picture for every question.

Writes vision_results.json.
"""
import json
import os
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
VLM = "Qwen/Qwen2-VL-2B-Instruct"
QUERIES = ["What is the price?", "What colour is the frame?",
           "Is it polarized?", "What is the product name?"]


def clock():
    if DEV == "mps":
        torch.mps.synchronize()
    return time.perf_counter()


proc = AutoProcessor.from_pretrained(VLM)
vlm = Qwen2VLForConditionalGeneration.from_pretrained(
    VLM, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()

here = os.path.dirname(os.path.abspath(__file__))
shot = os.path.join(here, "../../data/pdp_benchmark/"
                          "pdp-ray-ban-rb3025-aviator-gradient-8053672087901/screenshot_00.png")
img = Image.open(shot).convert("RGB")
img = img.crop((0, 0, img.width, min(img.height, 1400))).resize((672, 672))


def build(question):
    content = [{"type": "image"}] + ([{"type": "text", "text": question}] if question else [])
    text = proc.apply_chat_template([{"role": "user", "content": content}],
                                    tokenize=False, add_generation_prompt=True)
    return proc(text=[text], images=[img], return_tensors="pt").to(DEV)


cfg = vlm.config.text_config if hasattr(vlm.config, "text_config") else vlm.config
hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
kvh = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
BPT = 2 * cfg.num_hidden_layers * kvh * hd * 2

with torch.inference_mode():
    vlm(**build(QUERIES[0]), use_cache=True)  # warm

    img_tok_id = getattr(vlm.config, "image_token_id", None)
    probe = build("x")
    n_img = int((probe.input_ids == img_tok_id).sum()) if img_tok_id is not None else None

    # (a) stateless: full image + question forward, every query
    recompute = 0.0
    for q in QUERIES:
        inp = build(q)
        t0 = clock(); o = vlm(**inp, use_cache=True); recompute += clock() - t0
        del o

    # (b) encode the image once, reuse its KV; only question tokens are new
    base = build("")
    t0 = clock(); bo = vlm(**base, use_cache=True); encode_s = clock() - t0
    base_tokens = int(base.input_ids.shape[1])
    del bo

    reuse = 0.0
    for q in QUERIES:
        qids = proc.tokenizer(q, return_tensors="pt").input_ids.to(DEV)
        past = vlm(**base, use_cache=True).past_key_values     # fresh copy per query
        t0 = clock(); o = vlm(input_ids=qids, past_key_values=past, use_cache=True); reuse += clock() - t0
        del o, past

res = {
    "model": VLM, "device": DEV,
    "image": "Ray-Ban PDP screenshot, top crop, 672x672",
    "prompt_tokens_with_image": base_tokens,
    "image_tokens": n_img,
    "kb_per_token": round(BPT / 1024, 1),
    "image_kv_mb": round(BPT * base_tokens / 1024 / 1024, 1),
    "image_encode_s": round(encode_s, 3),
    "n_queries": len(QUERIES),
    "recompute_total_s": round(recompute, 3),
    "reuse_total_s": round(reuse, 3),
    "speedup": round(recompute / reuse, 1),
    "recompute_per_query_s": round(recompute / len(QUERIES), 3),
    "reuse_per_query_s": round(reuse / len(QUERIES), 3),
}
print(f"\nimage = {res['image_tokens']} of {base_tokens} prompt tokens  ->  KV {res['image_kv_mb']} MB")
print(f"encode image once: {encode_s*1000:.0f} ms")
print(f"re-encode per query : {recompute:.2f}s total ({res['recompute_per_query_s']*1000:.0f} ms/query)")
print(f"reuse image KV      : {reuse:.2f}s total ({res['reuse_per_query_s']*1000:.0f} ms/query)")
print(f"-> {res['speedup']}x")

with open(os.path.join(here, "vision_results.json"), "w") as f:
    json.dump(res, f, indent=2)
print("wrote vision_results.json")

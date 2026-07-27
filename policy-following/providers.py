"""Thin async adapters over the three hosted APIs, all returning parsed JSON
against a caller-supplied schema, plus token usage.

Every provider is driven in its strictest structured-output mode so that a
failure to follow the policy instructions cannot hide behind a parsing failure.
"""
import asyncio
import json
import os
import subprocess
import time


def secret(name: str) -> str:
    """Read a credential from the environment, falling back to Google Secret
    Manager when GCP_PROJECT is set (how we run it internally)."""
    val = os.environ.get(name)
    if val:
        return val
    project = os.environ.get("GCP_PROJECT")
    if not project:
        raise RuntimeError(f"set {name} in the environment (or GCP_PROJECT to use Secret Manager)")
    return subprocess.run(
        ["gcloud", "secrets", "versions", "access", "latest", "--secret", name, "--project", project],
        capture_output=True, text=True, check=True).stdout.strip()


class Result:
    __slots__ = ("data", "in_tok", "out_tok", "cached_tok", "latency_s", "error")

    def __init__(self, data=None, in_tok=0, out_tok=0, cached_tok=0, latency_s=0.0, error=None):
        self.data, self.in_tok, self.out_tok = data, in_tok, out_tok
        self.cached_tok, self.latency_s, self.error = cached_tok, latency_s, error


# --- Anthropic ------------------------------------------------------------

class AnthropicProvider:
    """Forced tool use — the most reliable structured-output path across the
    whole Claude range, so old and new models are driven identically."""
    family = "anthropic"

    def __init__(self, model):
        from anthropic import AsyncAnthropic
        self.model = model
        self.client = AsyncAnthropic(api_key=secret("ANTHROPIC_API_KEY"), max_retries=5)

    async def call(self, system, user, schema, cache_system=False):
        sys_block = [{"type": "text", "text": system}]
        if cache_system:
            sys_block[0]["cache_control"] = {"type": "ephemeral"}
        t0 = time.perf_counter()
        try:
            r = await self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=sys_block,
                messages=[{"role": "user", "content": user}],
                tools=[{"name": "record", "description": "Record the decision.",
                        "input_schema": schema}],
                tool_choice={"type": "tool", "name": "record"},
            )
        except Exception as e:
            return Result(latency_s=time.perf_counter() - t0,
                          error=f"{type(e).__name__}: {str(e)[:200]}")
        dt = time.perf_counter() - t0
        data = next((b.input for b in r.content if b.type == "tool_use"), None)
        u = r.usage
        return Result(data, u.input_tokens, u.output_tokens,
                      getattr(u, "cache_read_input_tokens", 0) or 0, dt)


# --- OpenAI ---------------------------------------------------------------

class OpenAIProvider:
    """Responses API with a strict json_schema. The gpt-5.6-* models are served
    on /v1/responses, so everything goes through the same path."""
    family = "openai"

    def __init__(self, model):
        from openai import AsyncOpenAI
        self.model = model
        self.client = AsyncOpenAI(api_key=secret("OPENAI_API_KEY"), max_retries=5)

    async def call(self, system, user, schema, cache_system=False):
        t0 = time.perf_counter()
        try:
            r = await self.client.responses.create(
                model=self.model,
                instructions=system,
                input=user,
                text={"format": {"type": "json_schema", "name": "record",
                                 "schema": schema, "strict": True}},
                max_output_tokens=4096,
            )
        except Exception as e:
            return Result(latency_s=time.perf_counter() - t0,
                          error=f"{type(e).__name__}: {str(e)[:200]}")
        dt = time.perf_counter() - t0
        txt = r.output_text
        try:
            data = json.loads(txt)
        except Exception:
            data = None
        u = r.usage
        det = getattr(u, "input_tokens_details", None)
        return Result(data, u.input_tokens, u.output_tokens,
                      getattr(det, "cached_tokens", 0) if det else 0, dt)


# --- Google ---------------------------------------------------------------

class GeminiProvider:
    family = "google"

    def __init__(self, model):
        from google import genai
        self.model = model
        self.client = genai.Client(api_key=secret("GEMINI_API_KEY"))

    async def call(self, system, user, schema, cache_system=False):
        t0 = time.perf_counter()
        try:
            r = await self.client.aio.models.generate_content(
                model=self.model,
                contents=user,
                config={
                    "system_instruction": system,
                    "response_mime_type": "application/json",
                    "response_schema": _gemini_schema(schema),
                    "temperature": 0,
                },
            )
        except Exception as e:
            return Result(latency_s=time.perf_counter() - t0,
                          error=f"{type(e).__name__}: {str(e)[:200]}")
        dt = time.perf_counter() - t0
        try:
            data = json.loads(r.text)
        except Exception:
            data = None
        u = r.usage_metadata
        return Result(data, u.prompt_token_count or 0,
                      (u.candidates_token_count or 0),
                      (getattr(u, "cached_content_token_count", 0) or 0), dt)


def _gemini_schema(schema):
    """Gemini rejects additionalProperties; strip the keys it does not accept."""
    if isinstance(schema, dict):
        return {k: _gemini_schema(v) for k, v in schema.items()
                if k not in ("additionalProperties", "strict")}
    if isinstance(schema, list):
        return [_gemini_schema(v) for v in schema]
    return schema


PROVIDERS = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "google": GeminiProvider,
}


def build(spec: str):
    """spec is "family:model", e.g. "anthropic:claude-opus-5"."""
    family, model = spec.split(":", 1)
    return PROVIDERS[family](model)


async def gather_limited(coros, limit=8):
    sem = asyncio.Semaphore(limit)

    async def run(c):
        async with sem:
            return await c
    return await asyncio.gather(*[run(c) for c in coros])

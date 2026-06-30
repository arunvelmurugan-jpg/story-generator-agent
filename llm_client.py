"""
Unified LLM client supporting OpenAI and Google Gemini.
Each agent calls llm_client.complete(system_prompt, user_prompt).
"""
import json
import re
import os
import time
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> Any:
    """Strip markdown code fences and parse JSON."""
    if not text or not text.strip():
        raise ValueError("LLM returned an empty response")
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    return json.loads(text.strip())


class LLMClient:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "openai").lower()

        if self.provider == "openai":
            import httpx
            from openai import OpenAI

            http_proxy = os.getenv("HTTP_PROXY")
            https_proxy = os.getenv("HTTPS_PROXY")
            http_client = None
            if http_proxy or https_proxy:
                proxies = {}
                if http_proxy:
                    proxies["http://"] = http_proxy
                if https_proxy:
                    proxies["https://"] = https_proxy
                http_client = httpx.Client(proxies=proxies)

            self._client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY", ""),
                http_client=http_client,
            )
            self._model = os.getenv("OPENAI_MODEL", "gpt-4o")

        elif self.provider == "gemini":
            import google.generativeai as genai

            genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))
            self._client = genai.GenerativeModel(
                os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            )
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {self.provider}")

    def complete(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.2
    ) -> tuple[Any, int, int]:
        """
        Returns: (parsed_json_object, input_tokens, output_tokens)
        """
        try:
            if self.provider == "openai":
                max_tokens = int(os.getenv("MAX_TOKENS", "6000"))
                req_timeout = int(os.getenv("REQUEST_TIMEOUT", "240"))
                reasoning_effort = os.getenv("REASONING_EFFORT", "").strip().lower()
                sys_len = len(system_prompt)
                usr_len = len(user_prompt)
                logger.info(f"[LLM-REQ] model={self._model} max_completion_tokens={max_tokens} timeout={req_timeout}s reasoning_effort={reasoning_effort or 'default'} system_prompt_chars={sys_len} user_prompt_chars={usr_len}")
                t0 = time.time()
                api_kwargs: dict[str, Any] = dict(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_completion_tokens=max_tokens,
                    timeout=req_timeout,
                )
                if reasoning_effort in ("low", "medium", "high"):
                    api_kwargs["reasoning_effort"] = reasoning_effort
                else:
                    api_kwargs["temperature"] = temperature
                response = self._client.chat.completions.create(**api_kwargs)
                elapsed = round(time.time() - t0, 2)
                finish_reason = response.choices[0].finish_reason
                raw_text = response.choices[0].message.content
                in_tokens = response.usage.prompt_tokens
                out_tokens = response.usage.completion_tokens
                reasoning_tokens = getattr(getattr(response.usage, 'completion_tokens_details', None), 'reasoning_tokens', 0) or 0
                content_tokens = out_tokens - reasoning_tokens
                logger.info(f"[LLM-RES] elapsed={elapsed}s finish_reason={finish_reason} prompt_tokens={in_tokens} completion_tokens={out_tokens} (reasoning={reasoning_tokens} content={content_tokens}) response_chars={len(raw_text or '')}")
                if finish_reason == "length":
                    logger.warning(f"[LLM-TRUNCATED] Response truncated! Used {out_tokens}/{max_tokens} tokens (reasoning={reasoning_tokens}). Increase MAX_TOKENS.")
                if not raw_text:
                    logger.error(f"[LLM-EMPTY] Empty content! finish_reason={finish_reason} reasoning_tokens={reasoning_tokens}")
                    raise ValueError(f"LLM returned empty content (finish_reason={finish_reason}, reasoning_tokens={reasoning_tokens}). The model used all tokens for reasoning with none left for output. Increase MAX_TOKENS.")

            elif self.provider == "gemini":
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                response = self._client.generate_content(full_prompt)
                raw_text = response.text
                in_tokens = getattr(
                    response.usage_metadata, "prompt_token_count", 0
                )
                out_tokens = getattr(
                    response.usage_metadata, "candidates_token_count", 0
                )

            parsed = _extract_json(raw_text)
            return parsed, in_tokens, out_tokens

        except Exception as e:
            logger.error(f"LLM API Error ({self.provider}): {e}")
            error_msg = str(e)
            if "Connection error" in error_msg or "ConnectError" in error_msg:
                raise Exception(
                    f"Connection error: Unable to reach {self.provider} API."
                ) from e
            if "AuthenticationError" in error_msg or "401" in error_msg:
                raise Exception(
                    f"Authentication error: Your {self.provider} API key is invalid."
                ) from e
            raise


def get_llm_client() -> LLMClient:
    """Factory — lazy singleton."""
    if not hasattr(get_llm_client, "_instance"):
        get_llm_client._instance = LLMClient()
    return get_llm_client._instance

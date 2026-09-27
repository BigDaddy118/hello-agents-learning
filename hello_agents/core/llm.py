"""HelloAgentsLLM 统一接口（7.2）。"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any, Optional, cast

from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam

# provider -> (专用 env key 名, 默认 base_url, 默认 model)
_PROVIDER_DEFAULTS: dict[str, tuple[str, str, str]] = {
    "openai": ("OPENAI_API_KEY", "https://api.openai.com/v1", "gpt-3.5-turbo"),
    "modelscope": (
        "MODELSCOPE_API_KEY",
        "https://api-inference.modelscope.cn/v1/",
        "Qwen/Qwen2.5-72B-Instruct",
    ),
    "zhipu": ("ZHIPU_API_KEY", "https://open.bigmodel.cn/api/paas/v4/", "glm-4"),
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com", "deepseek-chat"),
    "dashscope": (
        "DASHSCOPE_API_KEY",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "qwen-plus",
    ),
    "kimi": ("KIMI_API_KEY", "https://api.moonshot.cn/v1", "moonshot-v1-8k"),
    "ollama": ("OLLAMA_API_KEY", "http://localhost:11434/v1", "llama3"),
    "vllm": ("VLLM_API_KEY", "http://localhost:8000/v1", "default"),
    "local": ("LLM_API_KEY", "http://localhost:8000/v1", "default"),
}


class HelloAgentsLLM:
    """兼容 OpenAI 接口的多提供商 LLM 客户端。"""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = "auto",
        **kwargs: Any,
    ):
        if provider in (None, "auto"):
            self.provider = self._auto_detect_provider(api_key, base_url)
        else:
            self.provider = provider

        self.api_key, self.base_url = self._resolve_credentials(api_key, base_url)
        self.model = (
            model
            or os.getenv("LLM_MODEL_ID")
            or _PROVIDER_DEFAULTS.get(self.provider, ("", "", "gpt-3.5-turbo"))[2]
        )
        self.temperature = float(kwargs.get("temperature", 0.7))
        self.max_tokens = kwargs.get("max_tokens")
        self.timeout = int(kwargs.get("timeout") or os.getenv("LLM_TIMEOUT", "60"))

        if not self.api_key:
            raise ValueError(
                f"API key not found for provider '{self.provider}'. "
                "Set the provider-specific env var or LLM_API_KEY."
            )
        if not self.base_url:
            raise ValueError(
                f"base_url not found for provider '{self.provider}'. "
                "Set LLM_BASE_URL or pass base_url=..."
            )
        if not self.model:
            raise ValueError("model not found. Set LLM_MODEL_ID or pass model=...")

        self._client = OpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
        )
        print(f"🔌 HelloAgentsLLM ready: provider={self.provider}, model={self.model}")

    def _auto_detect_provider(
        self, api_key: Optional[str], base_url: Optional[str]
    ) -> str:
        # 1. 专用环境变量（最高优先级）
        if os.getenv("MODELSCOPE_API_KEY"):
            return "modelscope"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("ZHIPU_API_KEY"):
            return "zhipu"
        if os.getenv("DEEPSEEK_API_KEY"):
            return "deepseek"
        if os.getenv("DASHSCOPE_API_KEY"):
            return "dashscope"
        if os.getenv("KIMI_API_KEY"):
            return "kimi"

        actual_api_key = api_key or os.getenv("LLM_API_KEY")
        actual_base_url = base_url or os.getenv("LLM_BASE_URL")

        # 2. base_url
        if actual_base_url:
            u = actual_base_url.lower()
            if "api-inference.modelscope.cn" in u:
                return "modelscope"
            if "api.openai.com" in u:
                return "openai"
            if "open.bigmodel.cn" in u:
                return "zhipu"
            if "api.deepseek.com" in u:
                return "deepseek"
            if "dashscope.aliyuncs.com" in u:
                return "dashscope"
            if "api.moonshot.cn" in u:
                return "kimi"
            if "localhost" in u or "127.0.0.1" in u:
                if ":11434" in u:
                    return "ollama"
                if ":8000" in u:
                    return "vllm"
                return "local"

        # 3. key 格式
        if actual_api_key:
            if actual_api_key.startswith("ms-"):
                return "modelscope"
            if actual_api_key.startswith("sk-"):
                return "openai"  # ponytail: deepseek 等也用 sk-，靠 base_url 区分

        return "auto"

    def _resolve_credentials(
        self, api_key: Optional[str], base_url: Optional[str]
    ) -> tuple[str, str]:
        if self.provider in _PROVIDER_DEFAULTS:
            env_key, default_url, _ = _PROVIDER_DEFAULTS[self.provider]
            resolved_key = (
                api_key
                or os.getenv(env_key)
                or os.getenv("LLM_API_KEY")
                or (
                    self.provider
                    if self.provider in ("ollama", "vllm", "local")
                    else None
                )
            )
            resolved_url = base_url or os.getenv("LLM_BASE_URL") or default_url
            return resolved_key or "", resolved_url

        # auto / 未知：纯通用环境变量
        return (
            api_key or os.getenv("LLM_API_KEY") or "",
            base_url or os.getenv("LLM_BASE_URL") or "",
        )

    def _call_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        out: dict[str, Any] = {
            "temperature": kwargs.get("temperature", self.temperature),
        }
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens is not None:
            out["max_tokens"] = max_tokens
        # 透传其余 OpenAI 参数（除已处理字段）
        for k, v in kwargs.items():
            if k not in ("temperature", "max_tokens"):
                out[k] = v
        return out

    def think(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> Iterator[str]:
        """流式调用，逐块 yield 文本（兼容章节示例 for chunk in llm.think(...))."""
        print(f"🧠 正在调用 {self.model} ({self.provider})...")
        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=cast(list[ChatCompletionMessageParam], messages),
                stream=True,
                **self._call_kwargs(**kwargs),
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content
                if isinstance(content, str) and content:
                    print(content, end="", flush=True)
                    yield content
            print()
        except OpenAIError as e:
            print(f"❌ 调用LLM API时发生错误: {e}")

    def stream_invoke(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> Iterator[str]:
        """流式调用别名（供 Agent 使用；不在控制台重复打印）。"""
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            stream=True,
            **self._call_kwargs(**kwargs),
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if isinstance(content, str) and content:
                yield content

    def invoke(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """非流式调用，返回完整文本。"""
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            stream=False,
            **self._call_kwargs(**kwargs),
        )
        content = resp.choices[0].message.content
        return content if isinstance(content, str) else ""


def _demo() -> None:
    """最小自检：检测逻辑 + 一次真实调用（需 .env）。"""
    from dotenv import load_dotenv

    load_dotenv()
    d = object.__new__(HelloAgentsLLM)
    assert d._auto_detect_provider(None, "http://localhost:11434/v1") == "ollama"
    assert d._auto_detect_provider(None, "http://127.0.0.1:8000/v1") == "vllm"
    assert d._auto_detect_provider(None, "https://api.deepseek.com") == "deepseek"

    # key 格式检测：临时清掉 base_url 类 env，避免抢优先级
    saved = {k: os.environ.pop(k) for k in ("LLM_BASE_URL",) if k in os.environ}
    try:
        assert d._auto_detect_provider("ms-xxx", None) == "modelscope"
    finally:
        os.environ.update(saved)

    llm = HelloAgentsLLM()
    text = llm.invoke([{"role": "user", "content": "用三个字回答：1+1=?"}])
    assert text.strip(), "empty response"
    print(f"✅ provider={llm.provider} invoke ok: {text[:80]}")


if __name__ == "__main__":
    _demo()

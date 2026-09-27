"""环境自检：覆盖第4章基础依赖 + 第6.2 AutoGen。"""

from __future__ import annotations

import importlib
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from dotenv import load_dotenv

print("Python", sys.version)

pkgs = [
    "openai",
    "dotenv",
    "requests",
    "serpapi",
    "autogen_agentchat",
    "autogen_ext",
    "streamlit",
]
failed = False
for p in pkgs:
    try:
        m = importlib.import_module(p)
        ver = getattr(m, "__version__", "?")
        print(f"OK  {p}: {ver}")
    except ModuleNotFoundError as e:
        failed = True
        print(f"FAIL {p}: {e}")

print("--- dist ---")
for n in [
    "openai",
    "python-dotenv",
    "requests",
    "google-search-results",
    "autogen-agentchat",
    "autogen-core",
    "autogen-ext",
    "streamlit",
]:
    try:
        print(f"{n}=={version(n)}")
    except PackageNotFoundError:
        if n == "google-search-results":
            try:
                print(f"serpapi=={version('serpapi')}")
                continue
            except PackageNotFoundError:
                pass
        failed = True
        print(f"{n}: NOT INSTALLED")

try:
    importlib.import_module("autogen_agentchat.agents")
    importlib.import_module("autogen_agentchat.teams")
    importlib.import_module("autogen_ext.models.openai")
    print("OK  AutoGen 0.7 imports")
except ModuleNotFoundError as e:
    failed = True
    print(f"FAIL AutoGen 0.7 imports: {e}")

load_dotenv(Path(__file__).resolve().parent / ".env")
print("--- env ---")
for k in ["LLM_API_KEY", "LLM_MODEL_ID", "LLM_BASE_URL", "SERPAPI_API_KEY"]:
    v = os.getenv(k)
    if v:
        print(f"{k}: set({len(v)})")
    elif k == "SERPAPI_API_KEY":
        print(f"{k}: MISSING (chapter4 search 需要)")
    else:
        failed = True
        print(f"{k}: MISSING")

print("--- optional (未装不算错，6.3+ 再用) ---")
for p in ["agentscope", "camel", "langgraph", "langchain_openai", "tavily", "dashscope"]:
    try:
        importlib.import_module(p)
        print(f"present  {p}")
    except ModuleNotFoundError:
        print(f"skipped  {p}")

sys.exit(1 if failed else 0)

# 第十二章：智能体性能评估

## 准备

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[evaluation]"
.\.venv\Scripts\python.exe -m pip install "numpy==1.26.4" bfcl-eval

# BFCL 数据
git clone --depth 1 --filter=blob:none --sparse https://github.com/ShishirPatil/gorilla.git temp_gorilla
cd temp_gorilla
git sparse-checkout set berkeley-function-call-leaderboard/bfcl_eval/data
cd ..

# GAIA：申请访问后写入 .env
# HF_TOKEN=hf_xxx
# https://huggingface.co/datasets/gaia-benchmark/GAIA
```

## BFCL（12.2）

| 文件 | 说明 |
|------|------|
| `test_bfcl_ast.py` | AST 匹配自检（无需 LLM） |
| `02_bfcl_quick_start.py` | `BFCLEvaluationTool` 一键评估 |
| `03_bfcl_custom_evaluation.py` | Dataset + Evaluator |

## GAIA（12.3）

| 文件 | 说明 |
|------|------|
| `test_gaia_match.py` | 准精确匹配自检（无需 HF/LLM） |
| `05_gaia_quick_start.py` | `GAIAEvaluationTool` 一键评估 |
| `06_gaia_custom_evaluation.py` | Dataset + Evaluator |

## 数据生成质量（12.4）

| 文件 | 说明 |
|------|------|
| `test_data_generation_parse.py` | 解析/指标自检（无需 LLM） |
| `08_data_generation_llm_judge.py` | LLM Judge（样例 2 题） |
| `09_data_generation_win_rate.py` | Win Rate vs AIME 2025（2 次对比） |
| `07_data_generation_complete_flow.py` | 生成 + 评估完整流程 |
| `data_generation/` | 生成器、分步脚本、Gradio 人工验证 |

```powershell
$env:PYTHONUTF8 = "1"

# 自检
.\.venv\Scripts\python.exe code\chapter12\test_data_generation_parse.py

# Judge / Win Rate（需 LLM；Win Rate 会拉 math-ai/aime25）
.\.venv\Scripts\python.exe code\chapter12\08_data_generation_llm_judge.py
.\.venv\Scripts\python.exe code\chapter12\09_data_generation_win_rate.py

# 完整流程（生成 n 题再评估；费 token）
.\.venv\Scripts\python.exe code\chapter12\07_data_generation_complete_flow.py 3 2.0

# 只评估已有 JSON（在 code\chapter12 下）
cd code\chapter12
..\..\.\.venv\Scripts\python.exe data_generation\step2_evaluate_only.py data_generation\generated_data\aime_generated_sample.json
```

人工验证（evaluation extra 含 gradio）:

```powershell
cd code\chapter12
python data_generation\human_verification_ui.py data_generation\generated_data\aime_generated_sample.json
```

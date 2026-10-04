"""GAIA 数据集加载（12.3）。优先本地 data/gaia，否则用 HF snapshot_download。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class GAIADataset:
    """从 HuggingFace / 本地加载 GAIA（gated，需 HF_TOKEN）。"""

    def __init__(
        self,
        dataset_name: str = "gaia-benchmark/GAIA",
        split: str = "validation",
        level: int | None = None,
        local_data_dir: str | Path | None = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.level = level
        self.local_data_dir = Path(local_data_dir) if local_data_dir else None
        self.data: list[dict[str, Any]] = []

    def load(self) -> list[dict[str, Any]]:
        root = self._resolve_data_root()
        if root is not None:
            self.data = self._load_from_root(root)
        else:
            self.data = self._download_and_load()

        if self.level is not None:
            self.data = [x for x in self.data if x.get("level") == self.level]

        print("[ok] GAIA数据集加载完成")
        print(f"   数据源: {self.dataset_name}")
        print(f"   分割: {self.split}")
        print(f"   级别: {self.level or '全部'}")
        print(f"   样本数: {len(self.data)}")
        return self.data

    def _resolve_data_root(self) -> Path | None:
        if self.local_data_dir and self.local_data_dir.exists():
            return self.local_data_dir
        default = Path.cwd() / "data" / "gaia"
        split_dir = default / "2023" / self.split
        if (split_dir / "metadata.jsonl").exists() or (
            split_dir / "metadata.parquet"
        ).exists():
            return default
        return None

    def _load_from_root(self, root: Path) -> list[dict[str, Any]]:
        split_dir = root / "2023" / self.split
        jsonl = split_dir / "metadata.jsonl"
        if jsonl.exists():
            return self._load_metadata_jsonl(jsonl, root)
        # 2025-10 起官方改为 parquet
        parquet = split_dir / "metadata.parquet"
        if self.level is not None:
            level_pq = split_dir / f"metadata.level{self.level}.parquet"
            if level_pq.exists():
                parquet = level_pq
        if parquet.exists():
            return self._load_metadata_parquet(parquet, root)
        data: list[dict[str, Any]] = []
        for json_file in list(root.glob("*.json")) + list(root.glob("**/*.json")):
            if "gaia" not in json_file.name.lower():
                continue
            try:
                file_data = json.loads(json_file.read_text(encoding="utf-8"))
                items = file_data if isinstance(file_data, list) else [file_data]
                data.extend(self._standardize_item(i) for i in items)
                print(f"   加载文件: {json_file.name}")
            except Exception as e:
                print(f"   [!] 加载失败: {json_file.name} - {e}")
        return data

    def _download_and_load(self) -> list[dict[str, Any]]:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("   [!] 需要 huggingface_hub: pip install huggingface_hub")
            return []

        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("   [!] 未找到 HF_TOKEN（GAIA 为 gated dataset）")
            print("   申请: https://huggingface.co/datasets/gaia-benchmark/GAIA")
            return []

        local_dir = Path.cwd() / "data" / "gaia"
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"   正在从 HuggingFace 下载: {self.dataset_name}")
        try:
            # ponytail: metadata only; full attachments if file_name tasks matter
            snapshot_download(
                repo_id=self.dataset_name,
                repo_type="dataset",
                local_dir=str(local_dir),
                token=hf_token,
                allow_patterns=[
                    "2023/*/metadata*.parquet",
                    "2023/*/metadata.jsonl",
                    "*.md",
                ],
            )
            print(f"   [ok] 下载完成: {local_dir}")
        except Exception as e:
            print(f"   [!] 下载失败: {e}")
            print("   请确认已申请访问权限且 HF_TOKEN 有效")
            return []

        return self._load_from_root(local_dir)

    def _load_metadata_jsonl(self, metadata_file: Path, root: Path) -> list[dict[str, Any]]:
        data: list[dict[str, Any]] = []
        with metadata_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                data.extend(self._items_from_raw([item], root))
        print(f"   [ok] 加载了 {len(data)} 个样本 ({metadata_file.name})")
        return data

    def _load_metadata_parquet(self, metadata_file: Path, root: Path) -> list[dict[str, Any]]:
        try:
            import pandas as pd
        except ImportError:
            print("   [!] 需要 pandas: pip install pandas pyarrow")
            return []
        df = pd.read_parquet(metadata_file)
        raw = df.to_dict(orient="records")
        data = self._items_from_raw(raw, root)
        print(f"   [ok] 加载了 {len(data)} 个样本 ({metadata_file.name})")
        return data

    def _items_from_raw(self, raw: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
        data: list[dict[str, Any]] = []
        split_dir = root / "2023" / self.split
        for item in raw:
            if item.get("task_id") == "0-0-0-0-0":
                continue
            level = item.get("Level", item.get("level", 1))
            if self.level is not None and int(level) != int(self.level):
                continue
            file_name = item.get("file_name") or ""
            if file_name and not Path(str(file_name)).is_absolute():
                item = {**item, "file_name": str(split_dir / file_name)}
            data.append(self._standardize_item(item))
        return data

    def _standardize_item(self, item: dict[str, Any]) -> dict[str, Any]:
        level = item.get("Level", item.get("level", 1))
        try:
            level = int(level)
        except (TypeError, ValueError):
            level = 1
        file_name = item.get("file_name", "") or ""
        if file_name != file_name:  # NaN from parquet
            file_name = ""
        return {
            "task_id": str(item.get("task_id", "") or ""),
            "question": item.get("Question", item.get("question", "")) or "",
            "level": level,
            "final_answer": str(
                item.get("Final answer", item.get("final_answer", "")) or ""
            ),
            "file_name": str(file_name) if file_name else "",
            "file_path": item.get("file_path", "") or "",
            "annotator_metadata": item.get(
                "Annotator Metadata", item.get("annotator_metadata", {})
            )
            or {},
            "steps": item.get("Steps", item.get("steps", 0)) or 0,
            "tools": item.get("Tools", item.get("tools", [])) or [],
            "raw_item": item,
        }

    def get_sample(self, index: int) -> dict[str, Any]:
        if not self.data:
            self.load()
        return self.data[index] if index < len(self.data) else {}

    def get_by_level(self, level: int) -> list[dict[str, Any]]:
        if not self.data:
            self.load()
        return [x for x in self.data if x.get("level") == level]

    def get_level_distribution(self) -> dict[int, int]:
        if not self.data:
            self.load()
        dist = {1: 0, 2: 0, 3: 0}
        for item in self.data:
            lv = item.get("level", 1)
            if lv in dist:
                dist[lv] += 1
        return dist

    def get_statistics(self) -> dict[str, Any]:
        if not self.data:
            self.load()
        with_files = sum(1 for x in self.data if x.get("file_name"))
        steps = [x.get("steps", 0) for x in self.data if x.get("steps")]
        return {
            "total_samples": len(self.data),
            "level_distribution": self.get_level_distribution(),
            "samples_with_files": with_files,
            "average_steps": (sum(steps) / len(steps)) if steps else 0,
            "split": self.split,
        }

    def __len__(self) -> int:
        if not self.data:
            self.load()
        return len(self.data)

    def __iter__(self):
        if not self.data:
            self.load()
        return iter(self.data)

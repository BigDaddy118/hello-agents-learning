from typing import Literal, TypedDict

RecordType = Literal["execution", "reflection"]


class MemoryRecord(TypedDict):
    type: RecordType
    content: str


class Memory:
    """一个简单的短期记忆模块，用于存储智能体的行动与反思轨迹。"""

    def __init__(self) -> None:
        self.records: list[MemoryRecord] = []

    def add_record(self, record_type: RecordType, content: str) -> None:
        """向记忆中添加一条新记录。"""
        record: MemoryRecord = {"type": record_type, "content": content}
        self.records.append(record)
        print(f"📝 记忆已更新，新增一条 '{record_type}' 记录。")

    def get_trajectory(self) -> str:
        """将所有记忆记录格式化为一段文本，用于构建提示词。"""
        trajectory_parts: list[str] = []
        for record in self.records:
            if record["type"] == "execution":
                trajectory_parts.append(f"--- 上一轮尝试 (代码) ---\n{record['content']}")
            elif record["type"] == "reflection":
                trajectory_parts.append(f"--- 评审员反馈 ---\n{record['content']}")
        return "\n\n".join(trajectory_parts)

    def get_last_execution(self) -> str | None:
        """获取最近一次的执行结果。如果不存在，则返回 None。"""
        for record in reversed(self.records):
            if record["type"] == "execution":
                return record["content"]
        return None


if __name__ == "__main__":
    memory = Memory()
    memory.add_record("execution", "def find_primes(n):\n    return []")
    memory.add_record("reflection", "当前实现效率较低，建议改用埃拉托斯特尼筛法。")
    print("\n--- 记忆轨迹 ---")
    print(memory.get_trajectory())
    print("\n--- 最近一次执行 ---")
    print(memory.get_last_execution())

"""解析 Claude Code 对话历史"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import CLAUDE_PROJECTS_DIR, CLAUDE_HISTORY_FILE


@dataclass
class Message:
    uuid: str
    role: str  # user / assistant
    content: str  # 纯文本内容
    timestamp: str
    parent_uuid: str = ""
    model: str = ""
    message_type: str = ""  # user, assistant, progress, etc.
    tool_use: bool = False
    line_number: int = 0  # 在 JSONL 文件中的行号


@dataclass
class Session:
    session_id: str
    project_path: str  # 项目路径 (从目录名还原)
    file_path: str  # JSONL 文件路径
    messages: list[Message] = field(default_factory=list)
    first_timestamp: str = ""
    last_timestamp: str = ""
    summary: str = ""  # 首条用户消息摘要

    @property
    def display_name(self) -> str:
        short_project = self.project_path.split("/")[-1] if self.project_path else "unknown"
        ts = self.first_timestamp[:19].replace("T", " ") if self.first_timestamp else ""
        summary = self.summary[:60] if self.summary else self.session_id[:8]
        return f"[{ts}] {short_project} - {summary}"


def _extract_text(content) -> str:
    """从 message.content 中提取纯文本"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "thinking":
                    pass  # 跳过 thinking blocks
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


def parse_session_file(file_path: str | Path) -> Optional[Session]:
    """解析单个会话 JSONL 文件"""
    file_path = Path(file_path)
    if not file_path.exists() or file_path.suffix != ".jsonl":
        return None

    # 从目录名还原项目路径
    project_dir = file_path.parent.name
    project_path = project_dir.replace("-", "/")
    if project_path.startswith("/"):
        pass  # 已经是绝对路径
    session_id = file_path.stem

    messages: list[Message] = []
    first_ts = ""
    last_ts = ""
    summary = ""

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_type = entry.get("type", "")
            if msg_type not in ("user", "assistant"):
                continue

            msg_data = entry.get("message", {})
            if not msg_data:
                continue

            role = msg_data.get("role", msg_type)
            raw_content = msg_data.get("content", "")
            text = _extract_text(raw_content)

            # 跳过无文本内容的消息 (如纯 tool_use)
            if not text.strip():
                # 检查是否是 tool_use
                if isinstance(raw_content, list):
                    has_tool = any(
                        isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result")
                        for b in raw_content
                    )
                    if has_tool:
                        continue
                else:
                    continue

            ts = entry.get("timestamp", "")
            if not first_ts and ts:
                first_ts = ts
            if ts:
                last_ts = ts

            if not summary and role == "user" and text.strip():
                summary = text.strip()[:200]

            m = Message(
                uuid=entry.get("uuid", ""),
                role=role,
                content=text.strip(),
                timestamp=ts,
                parent_uuid=entry.get("parentUuid", ""),
                model=msg_data.get("model", ""),
                message_type=msg_type,
                line_number=line_no,
            )
            messages.append(m)

    if not messages:
        return None

    return Session(
        session_id=session_id,
        project_path=project_path,
        file_path=str(file_path),
        messages=messages,
        first_timestamp=first_ts,
        last_timestamp=last_ts,
        summary=summary,
    )


def scan_all_sessions() -> list[Session]:
    """扫描所有项目目录下的会话文件"""
    sessions: list[Session] = []
    if not CLAUDE_PROJECTS_DIR.exists():
        return sessions

    for project_dir in CLAUDE_PROJECTS_DIR.iterdir():
        if not project_dir.is_dir():
            continue
        for jsonl_file in project_dir.glob("*.jsonl"):
            # 跳过子目录中的 subagent 文件
            if "subagents" in str(jsonl_file):
                continue
            session = parse_session_file(jsonl_file)
            if session:
                sessions.append(session)

    # 按时间倒序排列
    sessions.sort(key=lambda s: s.last_timestamp or "", reverse=True)
    return sessions

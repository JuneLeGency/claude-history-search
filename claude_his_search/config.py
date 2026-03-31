"""配置管理模块"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path

CONFIG_DIR = Path.home() / ".claude_his_search"
CONFIG_FILE = CONFIG_DIR / "config.json"
INDEX_DIR = CONFIG_DIR / "index"

CLAUDE_DIR = Path.home() / ".claude"
CLAUDE_PROJECTS_DIR = CLAUDE_DIR / "projects"
CLAUDE_HISTORY_FILE = CLAUDE_DIR / "history.jsonl"

# 支持的后端
BACKEND_MLX = "mlx"
BACKEND_TORCH = "torch"


@dataclass
class AppConfig:
    # Embedding 模型配置
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dimension: int = 512
    backend: str = "auto"  # "auto", "mlx", "torch"
    model_source: str = "modelscope"  # "modelscope" 或 "huggingface"
    # 界面配置
    page_size: int = 50

    def save(self):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls) -> "AppConfig":
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        return cls()


def detect_backend() -> str:
    """检测可用的推理后端，优先 MLX"""
    try:
        import mlx.core
        import mlx_embeddings
        return BACKEND_MLX
    except ImportError:
        pass
    try:
        import sentence_transformers
        return BACKEND_TORCH
    except ImportError:
        pass
    return ""

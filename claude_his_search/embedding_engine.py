"""本地嵌入式语义搜索引擎

支持两种后端:
- MLX (Apple Silicon 原生加速，推荐)
- PyTorch + sentence-transformers (fallback)

模型源: ModelScope (国内快) 或 HuggingFace

索引策略: 增量索引 —— 只编码新增消息，已索引的消息跳过
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import Optional, Protocol

from .config import AppConfig, INDEX_DIR, BACKEND_MLX, BACKEND_TORCH, detect_backend


INDEX_FILE = INDEX_DIR / "embeddings.npz"
META_FILE = INDEX_DIR / "meta.json"


class EmbeddingBackend(Protocol):
    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray: ...


class MLXBackend:
    """MLX 后端 - Apple Silicon 原生加速"""

    def __init__(self, model_id: str, dimension: int):
        from mlx_embeddings.utils import load
        self.model, self.tokenizer_wrapper = load(model_id)
        self.tokenizer = self.tokenizer_wrapper._tokenizer
        self.dimension = dimension

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        import mlx.core as mx

        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="np"
        )
        input_ids = mx.array(encoded["input_ids"])
        attention_mask = mx.array(encoded["attention_mask"])

        output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        if hasattr(output, "text_embeds") and output.text_embeds is not None:
            embeds = np.array(output.text_embeds.astype(mx.float32))
        else:
            embeds = np.array(output.last_hidden_state[:, 0, :].astype(mx.float32))

        if embeds.shape[1] > self.dimension:
            embeds = embeds[:, :self.dimension]

        norms = np.linalg.norm(embeds, axis=1, keepdims=True)
        embeds = embeds / np.maximum(norms, 1e-10)
        return embeds


class TorchBackend:
    """PyTorch + sentence-transformers 后端"""

    def __init__(self, model_id: str, dimension: int):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            model_id,
            truncate_dim=dimension,
        )

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if is_query:
            return self.model.encode(texts, prompt_name="query", normalize_embeddings=True)
        return self.model.encode(texts, normalize_embeddings=True)


def _resolve_model_path(model_id: str, source: str) -> str:
    """根据模型源解析模型路径，ModelScope 优先下载到本地"""
    if source == "modelscope":
        try:
            from modelscope import snapshot_download
            local_dir = snapshot_download(model_id)
            return local_dir
        except ImportError:
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            return model_id
        except Exception:
            return model_id
    return model_id


class EmbeddingEngine:
    def __init__(self, config: AppConfig):
        self.config = config
        self._backend: Optional[EmbeddingBackend] = None
        self._backend_name: str = ""
        self._embeddings: Optional[np.ndarray] = None
        self._meta: list[dict] = []
        self._indexed_uuids: set[str] = set()  # 已索引的 message_uuid 集合
        self._load_index()

    def _ensure_backend(self):
        """懒加载推理后端"""
        if self._backend is not None:
            return

        backend = self.config.backend
        if backend == "auto":
            backend = detect_backend()

        model_path = _resolve_model_path(self.config.embedding_model, self.config.model_source)

        if backend == BACKEND_MLX:
            self._backend = MLXBackend(model_path, self.config.embedding_dimension)
            self._backend_name = "MLX"
        elif backend == BACKEND_TORCH:
            self._backend = TorchBackend(model_path, self.config.embedding_dimension)
            self._backend_name = "PyTorch"
        else:
            raise RuntimeError(
                "未找到可用的推理后端，请安装:\n"
                "  MLX (推荐): uv sync --extra mlx\n"
                "  PyTorch:    uv sync --extra torch"
            )

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def is_configured(self) -> bool:
        return True

    @property
    def index_size(self) -> int:
        return len(self._meta)

    def _load_index(self):
        if INDEX_FILE.exists() and META_FILE.exists():
            self._embeddings = np.load(INDEX_FILE)["arr_0"]
            with open(META_FILE, "r", encoding="utf-8") as f:
                self._meta = json.load(f)
            self._indexed_uuids = {m["message_uuid"] for m in self._meta}
        else:
            self._embeddings = None
            self._meta = []
            self._indexed_uuids = set()

    def _save_index(self):
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        if self._embeddings is not None:
            np.savez_compressed(INDEX_FILE, self._embeddings)
        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump(self._meta, f, ensure_ascii=False)

    def _collect_chunks(self, sessions) -> list[dict]:
        """从会话中收集需要索引的文本块，跳过已索引的"""
        chunks: list[dict] = []
        for session in sessions:
            for msg in session.messages:
                if msg.uuid in self._indexed_uuids:
                    continue
                text = msg.content.strip()
                if len(text) < 10:
                    continue
                if len(text) > 2000:
                    text = text[:2000]
                chunks.append({
                    "session_id": session.session_id,
                    "session_file": session.file_path,
                    "message_uuid": msg.uuid,
                    "line_number": msg.line_number,
                    "role": msg.role,
                    "timestamp": msg.timestamp,
                    "text_preview": text[:200],
                    "text": text,
                    "project_path": session.project_path,
                })
        return chunks

    def build_index(self, sessions, progress_callback=None, force_rebuild=False):
        """构建嵌入索引（增量 + 断点续传）

        每个 batch 编码后立即保存到磁盘，进程中断后重启会自动跳过已索引的消息。
        """
        if force_rebuild:
            self._embeddings = None
            self._meta = []
            self._indexed_uuids = set()

        new_chunks = self._collect_chunks(sessions)

        if not new_chunks:
            if progress_callback:
                progress_callback(0, 0, "索引已是最新，无新增消息")
            return

        total = len(new_chunks)

        if progress_callback:
            progress_callback(0, total, "加载模型中 (首次需从 ModelScope 下载 ~1.2GB)...")
        self._ensure_backend()

        batch_size = 64 if self._backend_name == "PyTorch" else 16
        save_interval = 5  # 每 5 个 batch 落盘一次

        for i in range(0, total, batch_size):
            batch = new_chunks[i:i + batch_size]
            texts = [c["text"] for c in batch]
            vecs = self._backend.encode(texts, is_query=False)

            # 立即合并到内存索引
            batch_meta = [{k: v for k, v in c.items() if k != "text"} for c in batch]
            if self._embeddings is not None and len(self._meta) > 0:
                self._embeddings = np.vstack([self._embeddings, vecs])
            else:
                self._embeddings = vecs
            self._meta.extend(batch_meta)
            self._indexed_uuids.update(c["message_uuid"] for c in batch)

            done = min(i + batch_size, total)
            batch_num = i // batch_size + 1

            # 定期落盘 (每 save_interval 个 batch 或最后一个 batch)
            if batch_num % save_interval == 0 or done >= total:
                self._save_index()

            if progress_callback:
                progress_callback(done, total, f"[{self._backend_name}] 编码 {done}/{total}")

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """语义搜索"""
        if self._embeddings is None or len(self._meta) == 0:
            return []

        self._ensure_backend()
        q_vec = self._backend.encode([query], is_query=True)[0]
        sims = self._embeddings @ q_vec

        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_indices:
            item = dict(self._meta[idx])
            item["score"] = float(sims[idx])
            results.append(item)
        return results

    def clear_index(self):
        self._embeddings = None
        self._meta = []
        self._indexed_uuids = set()
        if INDEX_FILE.exists():
            INDEX_FILE.unlink()
        if META_FILE.exists():
            META_FILE.unlink()

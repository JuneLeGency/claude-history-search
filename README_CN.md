# Claude History Search

搜索和浏览 [Claude Code](https://docs.anthropic.com/en/docs/claude-code) 对话历史，支持关键词搜索、语义搜索（本地 Embedding），原生 Qt GUI。

![platform](https://img.shields.io/badge/平台-macOS-lightgrey) ![license](https://img.shields.io/badge/许可-MIT-blue)

[English](README.md) | **中文**

![screenshot](screenshot.png)

## 功能特性

- **全文关键词搜索** — 搜索所有 Claude Code 会话 (`~/.claude/projects/`)
  - 多词 AND 匹配，CJK 标点感知分词
  - 匹配文字黄底高亮，自动跳转到匹配位置
  - 结果导航 (`Cmd+G` / `Cmd+Shift+G`) 在所有匹配间跳转（跨会话、跨页）
- **语义搜索** — 本地 Embedding 模型 (Qwen3-Embedding-0.6B，MLX/Metal 加速)
  - 增量索引 + 断点保存（进程中断不丢失进度）
  - 支持 ModelScope / HuggingFace 下载模型
- **原生 Qt C++ GUI** — 支持 macOS 深色/浅色模式 (`Cmd+T` 切换)
  - QPainter 自绘聊天气泡，超长消息自动折叠
  - 一键复制 `claude --resume <session-id>` 命令
  - 分页消息浏览 + 键盘导航
- **高性能 C++ JSONL 解析** — QtConcurrent 并行加载，结构化提取 `tool_result` 内容

## 安装

### 方式一：Homebrew（推荐）

```bash
brew tap JuneLeGency/claude-history-search
brew install claude-history-search
```

### 方式二：从源码编译

**前置依赖：**
- Qt 6：`brew install qt@6`
- CMake 3.20+：`brew install cmake`
- （可选）Python 3.10+ & uv，用于语义搜索：`brew install uv`

**编译运行：**

```bash
git clone https://github.com/JuneLeGency/claude-history-search.git
cd claude-history-search/qt
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
open claude-his-search.app
```

### 方式三：配置语义搜索（可选）

语义搜索需要本地 Embedding 模型，首次使用会自动下载（约 1.2GB）。

```bash
# 安装 Python 依赖
cd claude-history-search
uv sync --extra mlx

# 构建 Embedding 索引
# 方式 1：通过 GUI 菜单「索引 → 更新索引」
# 方式 2：命令行
uv run python -c "
from claude_his_search.embedding_engine import EmbeddingEngine
from claude_his_search.config import AppConfig
from claude_his_search.history_parser import scan_all_sessions
config = AppConfig.load()
engine = EmbeddingEngine(config)
sessions = scan_all_sessions()
engine.build_index(sessions, progress_callback=lambda c,t,m: print(m))
"
```

## 使用说明

### 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Cmd+F` | 聚焦搜索框 |
| `Enter` | 关键词搜索 |
| `Cmd+Shift+F` | 语义搜索 |
| `Escape` | 清除搜索 |
| `Cmd+G` | 下一个结果 |
| `Cmd+Shift+G` | 上一个结果 |
| `Cmd+[` | 上一页 |
| `Cmd+]` | 下一页 |
| `Cmd+Shift+C` | 复制 Resume 命令 |
| `Cmd+T` | 切换深色/浅色模式 |
| `Cmd+R` | 刷新会话列表 |
| `Cmd+U` | 更新 Embedding 索引 |
| `Cmd+,` | 设置 |

### 工作原理

1. 扫描 `~/.claude/projects/` 下所有会话 JSONL 文件
2. 结构化解析消息内容（纯文本 + tool_result 工具返回内容）
3. 预计算每个会话的搜索文本（normalize 后缓存，启动秒搜）
4. （可选）使用 Qwen3-Embedding-0.6B 在 Apple Silicon 上本地构建向量索引

### 数据与索引位置

| 路径 | 内容 |
|------|------|
| `~/.claude/projects/` | Claude Code 对话历史（只读） |
| `~/.claude_his_search/config.json` | 应用设置 |
| `~/.claude_his_search/index/` | Embedding 向量索引（增量、断点续传） |
| `~/.cache/claude-his-search/projects/` | 解析缓存（bincode 格式） |

## 项目结构

```
qt/                         # 原生 Qt C++ GUI（主程序）
  CMakeLists.txt
  main.cpp                  # 入口
  types.h                   # 数据结构
  parser.h/cpp              # JSONL 解析器（并行、CJK 分词）
  engine.h/cpp              # Embedding 引擎（调用 Python MLX 子进程）
  chatwidget.h              # QPainter 聊天气泡、可折叠消息
  sessiondelegate.h         # 自定义会话列表绘制
  mainwindow.h/cpp          # 主窗口、菜单、搜索、导航

claude_his_search/          # Python Embedding 后端
  config.py                 # 配置管理
  history_parser.py         # Python JSONL 解析器
  embedding_engine.py       # MLX/PyTorch 本地 Embedding（Qwen3-0.6B）

src/                        # Rust 后端（实验性）
  main.rs, parser.rs, ...
```

## 致谢

- [claude-history](https://github.com/raine/claude-history) — JSONL 解析模式和搜索评分思路
- [Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) — 本地 Embedding 模型
- [MLX](https://github.com/ml-explore/mlx) — Apple Silicon ML 推理框架

## 许可证

MIT

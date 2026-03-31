//! Claude JSONL 消息类型定义 (参考 claude-history)

use serde::Deserialize;

/// JSONL 文件中每行的条目类型
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
pub enum LogEntry {
    Summary {
        summary: String,
    },
    User {
        message: UserMessage,
        #[serde(default)]
        timestamp: Option<String>,
        #[serde(default)]
        uuid: Option<String>,
        #[serde(default)]
        cwd: Option<String>,
    },
    Assistant {
        message: AssistantMessage,
        #[serde(default)]
        timestamp: Option<String>,
        #[serde(default)]
        uuid: Option<String>,
    },
    #[serde(rename = "file-history-snapshot")]
    FileHistorySnapshot {
        #[serde(flatten)]
        _extra: serde_json::Value,
    },
    Progress {
        #[serde(flatten)]
        _extra: serde_json::Value,
    },
    System {
        #[serde(flatten)]
        _extra: serde_json::Value,
    },
    #[serde(rename = "custom-title")]
    CustomTitle {
        #[serde(rename = "customTitle")]
        custom_title: String,
    },
}

#[derive(Debug, Deserialize)]
pub struct UserMessage {
    pub role: String,
    pub content: UserContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum UserContent {
    String(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Deserialize)]
pub struct AssistantMessage {
    pub role: String,
    pub content: Vec<ContentBlock>,
    pub model: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ContentBlock {
    Text { text: String },
    ToolUse { name: String, input: serde_json::Value, #[allow(dead_code)] id: String },
    ToolResult { #[serde(default)] content: Option<serde_json::Value>, #[allow(dead_code)] tool_use_id: String },
    Thinking { thinking: String, #[allow(dead_code)] signature: String },
    #[allow(dead_code)]
    Image { source: serde_json::Value },
}

const MAX_TOOL_RESULT_CHARS: usize = 16 * 1024;

/// 仅提取 Text 块 (用于预览显示)
pub fn extract_text_from_blocks(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// 提取 Text + ToolResult 内容 (用于搜索索引)
pub fn extract_search_text_from_blocks(blocks: &[ContentBlock]) -> String {
    let mut parts = Vec::new();
    for block in blocks {
        match block {
            ContentBlock::Text { text } => parts.push(text.clone()),
            ContentBlock::ToolResult { content: Some(content), .. } => {
                if let Some(text) = extract_tool_result_text(content) {
                    parts.push(truncate_text(&text, MAX_TOOL_RESULT_CHARS));
                }
            }
            _ => {}
        }
    }
    parts.join(" ")
}

fn extract_tool_result_text(content: &serde_json::Value) -> Option<String> {
    match content {
        serde_json::Value::String(s) if !s.trim().is_empty() => Some(s.clone()),
        serde_json::Value::Array(items) => {
            let parts: Vec<&str> = items
                .iter()
                .filter_map(|item| match item {
                    serde_json::Value::Object(map) => {
                        let ty = map.get("type").and_then(|v| v.as_str());
                        if ty.is_none() || ty == Some("text") {
                            map.get("text").and_then(|v| v.as_str())
                        } else {
                            None
                        }
                    }
                    serde_json::Value::String(s) => Some(s.as_str()),
                    _ => None,
                })
                .collect();
            let joined = parts.join(" ");
            if joined.trim().is_empty() { None } else { Some(joined) }
        }
        _ => None,
    }
}

/// 截断文本，保留头尾
fn truncate_text(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_owned();
    }
    let head_end = s.floor_char_boundary(max * 3 / 4);
    let tail_start = s.ceil_char_boundary(s.len().saturating_sub(max / 4));
    format!("{} {}", &s[..head_end], &s[tail_start..])
}

pub fn extract_text_from_user(message: &UserMessage) -> String {
    match &message.content {
        UserContent::String(text) => text.clone(),
        UserContent::Blocks(blocks) => extract_text_from_blocks(blocks),
    }
}

pub fn extract_search_text_from_user(message: &UserMessage) -> String {
    match &message.content {
        UserContent::String(text) => text.clone(),
        UserContent::Blocks(blocks) => extract_search_text_from_blocks(blocks),
    }
}

pub fn extract_text_from_assistant(message: &AssistantMessage) -> String {
    extract_text_from_blocks(&message.content)
}

pub fn extract_search_text_from_assistant(message: &AssistantMessage) -> String {
    extract_search_text_from_blocks(&message.content)
}

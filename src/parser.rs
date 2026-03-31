//! JSONL 会话解析 + bincode 缓存 (参考 claude-history)

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write as _};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::claude::*;
use crate::config::claude_projects_dir;

// ── 数据结构 ──

#[derive(Debug, Clone)]
pub struct Message {
    pub uuid: String,
    pub role: String,
    pub content: String,       // 显示用文本
    pub search_text: String,   // 搜索用文本 (含 tool_result)
    pub timestamp: String,
    pub line_number: usize,
    pub model: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Session {
    pub session_id: String,
    pub project_path: String,
    pub file_path: PathBuf,
    pub messages: Vec<Message>,
    pub first_timestamp: String,
    pub last_timestamp: String,
    pub summary: String,
    pub custom_title: Option<String>,
    pub cwd: Option<String>,
    pub model: Option<String>,
    /// 预计算的搜索文本 (所有消息的 search_text 合并 + 小写 + CJK 分词)
    pub search_text_lower: String,
}

impl Session {
    pub fn display_name(&self) -> String {
        // 优先 custom_title
        if let Some(ref title) = self.custom_title {
            let ts = format_ts(&self.first_timestamp);
            let proj = short_project(&self.project_path);
            return format!("[{}] {} - {}", ts, proj, title);
        }
        let ts = format_ts(&self.first_timestamp);
        let proj = short_project(&self.project_path);
        let summary = if self.summary.len() > 60 {
            &self.summary[..self.summary.floor_char_boundary(60)]
        } else {
            &self.summary
        };
        format!("[{}] {} - {}", ts, proj, summary)
    }
}

fn format_ts(ts: &str) -> String {
    if ts.len() >= 19 { ts[..19].replace('T', " ") } else { String::new() }
}

fn short_project(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or("unknown")
}

// ── CJK 搜索文本预处理 ──

fn is_cjk_punctuation(c: char) -> bool {
    matches!(c,
        '\u{3000}' | '\u{3001}' | '\u{3002}' | '\u{3008}'..='\u{3011}' |
        '\u{3014}'..='\u{3017}' | '\u{FF01}' | '\u{FF08}' | '\u{FF09}' |
        '\u{FF0C}' | '\u{FF1A}' | '\u{FF1B}' | '\u{FF1F}' |
        '\u{201C}' | '\u{201D}' | '\u{2018}' | '\u{2019}' |
        '\u{2014}' | '\u{2026}' | '\u{00B7}'
    )
}

pub fn normalize_for_search(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch == '_' || ch == '-' || ch == '/' || is_cjk_punctuation(ch) {
            out.push(' ');
        } else {
            out.extend(ch.to_lowercase());
        }
    }
    out
}

// ── 解析单个 JSONL 文件 ──

pub fn parse_session_file(path: &Path) -> Option<Session> {
    let file = File::open(path).ok()?;
    let reader = BufReader::new(file);

    let project_dir = path.parent()?.file_name()?.to_str()?;
    let project_path = project_dir.replace('-', "/");
    let session_id = path.file_stem()?.to_str()?.to_string();

    let mut messages = Vec::new();
    let mut first_ts = String::new();
    let mut last_ts = String::new();
    let mut summary = String::new();
    let mut custom_title = None;
    let mut cwd = None;
    let mut model = None;
    let mut all_search_parts = Vec::new();

    for (line_no, line) in reader.lines().enumerate() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        if line.is_empty() { continue; }

        let entry: LogEntry = match serde_json::from_str(&line) {
            Ok(e) => e,
            Err(_) => continue,
        };

        match entry {
            LogEntry::Summary { summary: s } => {
                if summary.is_empty() {
                    summary = s;
                }
            }
            LogEntry::CustomTitle { custom_title: t } => {
                custom_title = Some(t);
            }
            LogEntry::User { message: msg, timestamp: ts, uuid, cwd: msg_cwd, .. } => {
                if cwd.is_none() { cwd = msg_cwd; }
                let ts_str = ts.unwrap_or_default();
                if first_ts.is_empty() && !ts_str.is_empty() { first_ts = ts_str.clone(); }
                if !ts_str.is_empty() { last_ts = ts_str.clone(); }

                let display_text = extract_text_from_user(&msg);
                let search_text = extract_search_text_from_user(&msg);

                if display_text.is_empty() && search_text.is_empty() { continue; }
                if summary.is_empty() && !display_text.is_empty() {
                    summary = display_text.chars().take(200).collect();
                }

                all_search_parts.push(search_text.clone());

                messages.push(Message {
                    uuid: uuid.unwrap_or_default(),
                    role: "user".into(),
                    content: display_text,
                    search_text,
                    timestamp: ts_str,
                    line_number: line_no + 1,
                    model: None,
                });
            }
            LogEntry::Assistant { message: msg, timestamp: ts, uuid, .. } => {
                let ts_str = ts.unwrap_or_default();
                if first_ts.is_empty() && !ts_str.is_empty() { first_ts = ts_str.clone(); }
                if !ts_str.is_empty() { last_ts = ts_str.clone(); }
                if model.is_none() { model = msg.model.clone(); }

                let display_text = extract_text_from_assistant(&msg);
                let search_text = extract_search_text_from_assistant(&msg);

                if display_text.is_empty() && search_text.is_empty() { continue; }
                all_search_parts.push(search_text.clone());

                messages.push(Message {
                    uuid: uuid.unwrap_or_default(),
                    role: "assistant".into(),
                    content: display_text,
                    search_text,
                    timestamp: ts_str,
                    line_number: line_no + 1,
                    model: msg.model,
                });
            }
            _ => {}
        }
    }

    if messages.is_empty() { return None; }

    // 预计算搜索文本
    let mut full_search = String::new();
    if let Some(ref t) = custom_title {
        full_search.push_str(t);
        full_search.push(' ');
    }
    full_search.push_str(&summary);
    full_search.push(' ');
    full_search.push_str(&all_search_parts.join(" "));
    let search_text_lower = normalize_for_search(&full_search);

    Some(Session {
        session_id,
        project_path,
        file_path: path.to_path_buf(),
        messages,
        first_timestamp: first_ts,
        last_timestamp: last_ts,
        summary,
        custom_title,
        cwd,
        model,
        search_text_lower,
    })
}

// ── bincode 缓存 ──

const CACHE_MAGIC: [u8; 8] = *b"CLSRCH02";
const SCHEMA_VERSION: u32 = 2;

#[derive(Serialize, Deserialize)]
struct ProjectCache {
    magic: [u8; 8],
    schema_version: u32,
    entries: HashMap<String, CacheEntry>,
}

#[derive(Serialize, Deserialize, Clone)]
struct CacheEntry {
    file_size: u64,
    mtime_secs: u64,
    mtime_nsecs: u32,
    is_empty: bool,
    // 序列化的 SessionMeta (不含完整消息，只存摘要数据)
    summary: String,
    custom_title: Option<String>,
    first_timestamp: String,
    last_timestamp: String,
    message_count: usize,
    search_text_lower: String,
    cwd: Option<String>,
    model: Option<String>,
}

fn cache_dir() -> PathBuf {
    dirs::home_dir().unwrap().join(".cache").join("claude-his-search").join("projects")
}

fn cache_path(project_dir_name: &str) -> PathBuf {
    cache_dir().join(format!("{}.bin", project_dir_name))
}

fn read_cache(project_dir_name: &str) -> Option<HashMap<String, CacheEntry>> {
    let data = fs::read(cache_path(project_dir_name)).ok()?;
    if data.len() < 12 || data[..8] != CACHE_MAGIC { return None; }
    let cache: ProjectCache = bincode::deserialize(&data).ok()?;
    if cache.schema_version != SCHEMA_VERSION { return None; }
    Some(cache.entries)
}

fn write_cache(project_dir_name: &str, entries: HashMap<String, CacheEntry>) {
    let path = cache_path(project_dir_name);
    let _ = fs::create_dir_all(path.parent().unwrap());
    let cache = ProjectCache { magic: CACHE_MAGIC, schema_version: SCHEMA_VERSION, entries };
    if let Ok(data) = bincode::serialize(&cache) {
        let _ = fs::write(&path, data);
    }
}

fn entry_matches(entry: &CacheEntry, file_size: u64, mtime: SystemTime) -> bool {
    let d = mtime.duration_since(UNIX_EPOCH).unwrap_or_default();
    entry.file_size == file_size && entry.mtime_secs == d.as_secs() && entry.mtime_nsecs == d.subsec_nanos()
}

// ── 全量扫描 (含缓存) ──

pub fn scan_all_sessions() -> Vec<Session> {
    let projects_dir = claude_projects_dir();
    if !projects_dir.exists() { return Vec::new(); }

    let mut project_dirs: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = fs::read_dir(&projects_dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() { project_dirs.push(p); }
        }
    }

    let mut sessions: Vec<Session> = project_dirs
        .par_iter()
        .flat_map(|project_dir| load_project_sessions(project_dir))
        .collect();

    sessions.sort_by(|a, b| b.last_timestamp.cmp(&a.last_timestamp));
    sessions
}

fn load_project_sessions(project_dir: &Path) -> Vec<Session> {
    let dir_name = project_dir.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let cached = read_cache(dir_name).unwrap_or_default();

    let mut files_meta: Vec<(PathBuf, Option<SystemTime>, u64)> = Vec::new();
    if let Ok(entries) = fs::read_dir(project_dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().is_some_and(|e| e == "jsonl")
                && !p.file_name().unwrap_or_default().to_string_lossy().starts_with("agent-")
                && !p.to_string_lossy().contains("subagents")
            {
                let meta = entry.metadata().ok();
                let mtime = meta.as_ref().and_then(|m| m.modified().ok());
                let size = meta.as_ref().map(|m| m.len()).unwrap_or(0);
                files_meta.push((p, mtime, size));
            }
        }
    }

    let mut sessions = Vec::new();
    let mut to_parse: Vec<(PathBuf, Option<SystemTime>, u64)> = Vec::new();
    let mut dirty = false;

    for (path, mtime, size) in &files_meta {
        let fname = path.file_name().and_then(|f| f.to_str()).unwrap_or("");
        if let Some(mt) = mtime {
            if let Some(entry) = cached.get(fname) {
                if entry_matches(entry, *size, *mt) {
                    if !entry.is_empty {
                        // 缓存命中 — 但不含完整消息，需要重新解析以获取消息列表
                        // 这里我们可以只缓存 metadata，完整消息还是从文件读取
                        // 为了速度，对命中的文件也进行解析，但跳过 search_text_lower 计算
                        if let Some(session) = parse_session_file(path) {
                            sessions.push(session);
                        }
                    }
                    continue;
                }
            }
        }
        dirty = true;
        to_parse.push((path.clone(), *mtime, *size));
    }

    // 并行解析 cache miss 的文件
    let parsed: Vec<(Option<Session>, String, u64, Option<SystemTime>)> = to_parse
        .into_par_iter()
        .map(|(path, mtime, size)| {
            let fname = path.file_name().and_then(|f| f.to_str()).unwrap_or("").to_string();
            let session = parse_session_file(&path);
            (session, fname, size, mtime)
        })
        .collect();

    for (session, _, _, _) in &parsed {
        if let Some(s) = session {
            sessions.push(s.clone());
        }
    }

    // 写入缓存
    if dirty {
        let mut new_cache: HashMap<String, CacheEntry> = HashMap::new();
        for s in &sessions {
            let fname = s.file_path.file_name().and_then(|f| f.to_str()).unwrap_or("");
            if let Some((_, mtime, size)) = files_meta.iter().find(|(p, _, _)| p == &s.file_path) {
                if let Some(mt) = mtime {
                    let d = mt.duration_since(UNIX_EPOCH).unwrap_or_default();
                    new_cache.insert(fname.to_string(), CacheEntry {
                        file_size: *size,
                        mtime_secs: d.as_secs(),
                        mtime_nsecs: d.subsec_nanos(),
                        is_empty: false,
                        summary: s.summary.clone(),
                        custom_title: s.custom_title.clone(),
                        first_timestamp: s.first_timestamp.clone(),
                        last_timestamp: s.last_timestamp.clone(),
                        message_count: s.messages.len(),
                        search_text_lower: s.search_text_lower.clone(),
                        cwd: s.cwd.clone(),
                        model: s.model.clone(),
                    });
                }
            }
        }
        // 空文件的负缓存
        for (session, fname, size, mtime) in &parsed {
            if session.is_none() {
                if let Some(mt) = mtime {
                    let d = mt.duration_since(UNIX_EPOCH).unwrap_or_default();
                    new_cache.insert(fname.clone(), CacheEntry {
                        file_size: *size, mtime_secs: d.as_secs(), mtime_nsecs: d.subsec_nanos(),
                        is_empty: true, summary: String::new(), custom_title: None,
                        first_timestamp: String::new(), last_timestamp: String::new(),
                        message_count: 0, search_text_lower: String::new(),
                        cwd: None, model: None,
                    });
                }
            }
        }
        write_cache(dir_name, new_cache);
    }

    sessions
}

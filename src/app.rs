//! egui 主界面

use eframe::egui;
use std::collections::HashMap;
use std::sync::mpsc;
use std::thread;

use crate::config::AppConfig;
use crate::engine::{EmbeddingEngine, SearchResult};
use crate::parser::{scan_all_sessions, Session};

// ── 背景任务通信 ──

enum TaskResult {
    SessionsLoaded(Vec<Session>),
    IndexDone { new_count: usize, total: usize },
    IndexProgress(usize, usize, String),
    SearchResults(Vec<SearchResult>),
    ModelLoaded(String),
    Error(String),
}

// ── 颜色 (Catppuccin Mocha) ──

const BG_BASE: egui::Color32 = egui::Color32::from_rgb(30, 30, 46);
const BG_SURFACE: egui::Color32 = egui::Color32::from_rgb(24, 24, 37);
const BG_OVERLAY: egui::Color32 = egui::Color32::from_rgb(49, 50, 68);
const TEXT_MAIN: egui::Color32 = egui::Color32::from_rgb(205, 214, 244);
const TEXT_DIM: egui::Color32 = egui::Color32::from_rgb(108, 112, 134);
const BLUE: egui::Color32 = egui::Color32::from_rgb(137, 180, 250);
const GREEN: egui::Color32 = egui::Color32::from_rgb(166, 227, 161);
const YELLOW: egui::Color32 = egui::Color32::from_rgb(249, 226, 175);

#[derive(PartialEq)]
enum SearchTab {
    Keyword,
    Semantic,
}

pub struct App {
    config: AppConfig,

    // 数据
    sessions: Vec<Session>,
    filtered_indices: Vec<usize>,
    selected_session: Option<usize>,
    current_page: usize,
    highlight_uuid: String,

    // 搜索
    search_tab: SearchTab,
    keyword_query: String,
    semantic_query: String,

    // 状态
    status_msg: String,
    progress: Option<(usize, usize)>,
    indexing: bool,
    index_size: usize,
    backend_name: String,

    // 线程通信
    tx: mpsc::Sender<TaskResult>,
    rx: mpsc::Receiver<TaskResult>,

    // 设置对话框
    show_settings: bool,
    settings_draft: AppConfig,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // 加载中文字体 (macOS 系统字体)
        let mut fonts = egui::FontDefinitions::default();
        let font_paths = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
        ];
        for path in &font_paths {
            if let Ok(data) = std::fs::read(path) {
                fonts.font_data.insert(
                    "chinese".into(),
                    std::sync::Arc::new(egui::FontData::from_owned(data)),
                );
                // 插入到所有字体族的首位作为 fallback
                for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
                    if let Some(list) = fonts.families.get_mut(&family) {
                        list.push("chinese".into());
                    }
                }
                break;
            }
        }
        cc.egui_ctx.set_fonts(fonts);

        // 深色主题
        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = BG_BASE;
        visuals.window_fill = BG_OVERLAY;
        visuals.widgets.noninteractive.bg_fill = BG_SURFACE;
        visuals.widgets.inactive.bg_fill = BG_OVERLAY;
        visuals.override_text_color = Some(TEXT_MAIN);
        cc.egui_ctx.set_visuals(visuals);

        let config = AppConfig::load();
        let index_size = {
            let engine = EmbeddingEngine::new(config.clone());
            engine.index_size()
        };

        let (tx, rx) = mpsc::channel();
        let app = Self {
            settings_draft: config.clone(),
            config,
            sessions: Vec::new(),
            filtered_indices: Vec::new(),
            selected_session: None,
            current_page: 0,
            highlight_uuid: String::new(),
            search_tab: SearchTab::Keyword,
            keyword_query: String::new(),
            semantic_query: String::new(),
            status_msg: "正在扫描会话...".into(),
            progress: None,
            indexing: false,
            index_size,
            backend_name: String::new(),
            tx,
            rx,
            show_settings: false,
        };

        // 启动后台扫描
        app.start_scan();
        app
    }

    fn start_scan(&self) {
        let tx = self.tx.clone();
        thread::spawn(move || {
            let sessions = scan_all_sessions();
            let _ = tx.send(TaskResult::SessionsLoaded(sessions));
        });
    }

    fn start_build_index(&mut self, force: bool) {
        if self.indexing {
            self.status_msg = "索引正在构建中...".into();
            return;
        }
        self.indexing = true;
        let sessions = self.sessions.clone();
        let config = self.config.clone();
        let tx = self.tx.clone();
        thread::spawn(move || {
            let mut engine = EmbeddingEngine::new(config);
            let progress_tx = tx.clone();
            let cb: Box<dyn Fn(usize, usize, &str) + Send> = Box::new(move |cur, total, msg| {
                let _ = progress_tx.send(TaskResult::IndexProgress(cur, total, msg.into()));
            });
            let result = engine.build_index(&sessions, force, Some(&cb));
            match result {
                Ok(new_count) => {
                    let _ = tx.send(TaskResult::IndexDone {
                        new_count,
                        total: engine.index_size(),
                    });
                }
                Err(e) => {
                    let _ = tx.send(TaskResult::Error(format!("索引构建失败: {}", e)));
                }
            }
        });
    }

    fn start_search(&self, query: String) {
        let config = self.config.clone();
        let tx = self.tx.clone();
        thread::spawn(move || {
            let mut engine = EmbeddingEngine::new(config);
            match engine.search(&query, 20) {
                Ok(results) => {
                    let _ = tx.send(TaskResult::SearchResults(results));
                }
                Err(e) => {
                    let _ = tx.send(TaskResult::Error(format!("搜索失败: {}", e)));
                }
            }
        });
    }

    fn keyword_search(&mut self) {
        let query = self.keyword_query.trim();
        if query.is_empty() {
            self.clear_filter();
            return;
        }
        let query_lower = crate::parser::normalize_for_search(query);
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();
        if query_words.is_empty() {
            self.clear_filter();
            return;
        }

        // 使用预计算的 search_text_lower 进行快速 AND 匹配
        self.filtered_indices = self
            .sessions
            .iter()
            .enumerate()
            .filter(|(_, s)| {
                query_words.iter().all(|w| s.search_text_lower.contains(w))
            })
            .map(|(i, _)| i)
            .collect();
        self.selected_session = None;
        self.status_msg = format!("关键词搜索: 找到 {} 个匹配会话", self.filtered_indices.len());
    }

    fn clear_filter(&mut self) {
        self.filtered_indices = (0..self.sessions.len()).collect();
        self.keyword_query.clear();
        self.semantic_query.clear();
        self.selected_session = None;
        self.highlight_uuid.clear();
    }

    fn apply_semantic_results(&mut self, results: Vec<SearchResult>) {
        if results.is_empty() {
            self.status_msg = "未找到相关结果".into();
            return;
        }

        // 按 session 分组取最高分
        let mut session_scores: HashMap<String, (f32, String)> = HashMap::new();
        for r in &results {
            let entry = session_scores
                .entry(r.meta.session_id.clone())
                .or_insert((0.0, String::new()));
            if r.score > entry.0 {
                *entry = (r.score, r.meta.message_uuid.clone());
            }
        }

        // 过滤+排序
        let mut matched: Vec<(usize, f32, String)> = self
            .sessions
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                session_scores
                    .get(&s.session_id)
                    .map(|(score, uuid)| (i, *score, uuid.clone()))
            })
            .collect();
        matched.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        self.filtered_indices = matched.iter().map(|(i, _, _)| *i).collect();
        self.status_msg = format!("语义搜索: 找到 {} 个相关会话", matched.len());

        // 自动选中第一个并高亮
        if let Some((idx, _, uuid)) = matched.first() {
            self.selected_session = Some(*idx);
            self.highlight_uuid = uuid.clone();
            // 跳转到对应页
            if let Some(session) = self.sessions.get(*idx) {
                if let Some(pos) = session.messages.iter().position(|m| m.uuid == *uuid) {
                    self.current_page = pos / self.config.page_size;
                }
            }
        }
    }

    fn process_results(&mut self) {
        while let Ok(result) = self.rx.try_recv() {
            match result {
                TaskResult::SessionsLoaded(sessions) => {
                    self.status_msg = format!("共 {} 个会话", sessions.len());
                    self.sessions = sessions;
                    self.filtered_indices = (0..self.sessions.len()).collect();
                }
                TaskResult::IndexDone { new_count, total } => {
                    self.indexing = false;
                    self.progress = None;
                    self.index_size = total;
                    self.status_msg = if new_count > 0 {
                        format!("索引更新完成! 新增 {} 条，共 {} 条", new_count, total)
                    } else {
                        format!("索引已是最新，共 {} 条", total)
                    };
                }
                TaskResult::IndexProgress(cur, total, msg) => {
                    self.progress = Some((cur, total));
                    self.status_msg = msg;
                }
                TaskResult::SearchResults(results) => {
                    self.apply_semantic_results(results);
                }
                TaskResult::ModelLoaded(name) => {
                    self.backend_name = name;
                    self.status_msg = "模型加载完成!".into();
                }
                TaskResult::Error(msg) => {
                    self.indexing = false;
                    self.progress = None;
                    self.status_msg = msg;
                }
            }
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.process_results();

        // 有进度时持续刷新
        if self.indexing {
            ctx.request_repaint();
        }

        // ── 顶部工具栏 ──
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("设置").clicked() {
                    self.settings_draft = self.config.clone();
                    self.show_settings = true;
                }
                ui.separator();
                if ui.button("更新索引").clicked() {
                    self.start_build_index(false);
                }
                if ui.button("重建索引").clicked() {
                    self.start_build_index(true);
                }
                if ui.button("清除索引").clicked() && !self.indexing {
                    let mut engine = EmbeddingEngine::new(self.config.clone());
                    engine.clear_index();
                    self.index_size = 0;
                    self.status_msg = "索引已清除".into();
                }
                ui.separator();
                if ui.button("刷新会话").clicked() {
                    self.start_scan();
                }
            });
        });

        // ── 底部状态栏 ──
        egui::TopBottomPanel::bottom("statusbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(&self.status_msg).size(12.0).color(TEXT_DIM));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        egui::RichText::new(format!("索引: {} 条", self.index_size))
                            .size(12.0)
                            .color(TEXT_DIM),
                    );
                    if !self.backend_name.is_empty() {
                        ui.label(
                            egui::RichText::new(format!("后端: {}", self.backend_name))
                                .size(12.0)
                                .color(GREEN),
                        );
                    }
                    if let Some((cur, total)) = self.progress {
                        if total > 0 {
                            let frac = cur as f32 / total as f32;
                            ui.add(
                                egui::ProgressBar::new(frac)
                                    .text(format!("{}/{}", cur, total))
                                    .desired_width(200.0),
                            );
                        }
                    }
                });
            });
        });

        // ── 搜索区域 ──
        egui::TopBottomPanel::top("search").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.search_tab, SearchTab::Keyword, "关键词搜索");
                ui.selectable_value(&mut self.search_tab, SearchTab::Semantic, "语义搜索");
            });
            ui.horizontal(|ui| {
                match self.search_tab {
                    SearchTab::Keyword => {
                        let resp = ui.add(
                            egui::TextEdit::singleline(&mut self.keyword_query)
                                .hint_text("关键词搜索会话内容...")
                                .desired_width(ui.available_width() - 140.0),
                        );
                        if ui.button("搜索").clicked()
                            || (resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)))
                        {
                            self.keyword_search();
                        }
                        if ui.button("清除").clicked() {
                            self.clear_filter();
                        }
                    }
                    SearchTab::Semantic => {
                        let resp = ui.add(
                            egui::TextEdit::singleline(&mut self.semantic_query)
                                .hint_text("用自然语言描述你要找的内容...")
                                .desired_width(ui.available_width() - 120.0),
                        );
                        if ui.button("搜索").clicked()
                            || (resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)))
                        {
                            if self.index_size == 0 {
                                self.status_msg = "请先构建索引".into();
                            } else if !self.semantic_query.is_empty() {
                                let q = self.semantic_query.clone();
                                self.status_msg = "语义搜索中...".into();
                                self.start_search(q);
                            }
                        }
                    }
                }
            });
            ui.add_space(4.0);
        });

        // ── 左侧会话列表 ──
        egui::SidePanel::left("sessions")
            .default_width(380.0)
            .show(ctx, |ui| {
                ui.label(
                    egui::RichText::new(format!(
                        "会话列表 ({})",
                        self.filtered_indices.len()
                    ))
                    .size(13.0)
                    .color(TEXT_DIM),
                );
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for &idx in &self.filtered_indices.clone() {
                        if let Some(session) = self.sessions.get(idx) {
                            let is_selected = self.selected_session == Some(idx);
                            let label = egui::RichText::new(session.display_name())
                                .size(12.0)
                                .color(if is_selected { BLUE } else { TEXT_MAIN });
                            if ui.selectable_label(is_selected, label).clicked() {
                                self.selected_session = Some(idx);
                                self.current_page = 0;
                                self.highlight_uuid.clear();
                            }
                        }
                    }
                });
            });

        // ── 中央聊天视图 ──
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(idx) = self.selected_session {
                if let Some(session) = self.sessions.get(idx) {
                    // Header
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new(session.display_name())
                                .size(14.0)
                                .strong()
                                .color(TEXT_MAIN),
                        );
                        ui.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                if ui.button("复制 Resume 命令").clicked() {
                                    let cmd =
                                        format!("claude --resume {}", session.session_id);
                                    if let Ok(mut clip) = arboard::Clipboard::new() {
                                        let _ = clip.set_text(&cmd);
                                        self.status_msg = format!("已复制: {}", cmd);
                                    }
                                }
                            },
                        );
                    });
                    ui.separator();

                    // Messages (paginated)
                    let page_size = self.config.page_size;
                    let total_msgs = session.messages.len();
                    let total_pages = (total_msgs + page_size - 1) / page_size;
                    let start = self.current_page * page_size;
                    let end = (start + page_size).min(total_msgs);

                    let scroll = egui::ScrollArea::vertical()
                        .auto_shrink([false; 2]);
                    scroll.show(ui, |ui| {
                        for msg in &session.messages[start..end] {
                            let is_user = msg.role == "user";
                            let border_color = if is_user { BLUE } else { GREEN };
                            let is_highlight = !self.highlight_uuid.is_empty()
                                && msg.uuid == self.highlight_uuid;
                            let bg = if is_highlight {
                                egui::Color32::from_rgba_premultiplied(249, 226, 175, 30)
                            } else if is_user {
                                BG_OVERLAY
                            } else {
                                BG_BASE
                            };

                            let frame = egui::Frame::new()
                                .fill(bg)
                                .inner_margin(10.0)
                                .corner_radius(8.0)
                                .stroke(egui::Stroke::new(
                                    if is_highlight { 2.0 } else { 0.0 },
                                    if is_highlight { YELLOW } else { border_color },
                                ));

                            frame.show(ui, |ui| {
                                // 左边色条
                                ui.horizontal(|ui| {
                                    ui.add(egui::Separator::default().vertical().spacing(3.0));
                                    ui.vertical(|ui| {
                                        let role_label = if is_user { "You" } else { "Assistant" };
                                        let role_color = if is_user { BLUE } else { GREEN };
                                        ui.label(
                                            egui::RichText::new(role_label)
                                                .size(11.0)
                                                .color(role_color)
                                                .strong(),
                                        );
                                        // 消息正文 (截断超长消息避免卡顿)
                                        let display_text = if msg.content.len() > 5000 {
                                            format!("{}...\n[消息过长，已截断]", &msg.content[..5000])
                                        } else {
                                            msg.content.clone()
                                        };
                                        ui.label(
                                            egui::RichText::new(&display_text)
                                                .size(13.0)
                                                .color(TEXT_MAIN),
                                        );
                                        if !msg.timestamp.is_empty() && msg.timestamp.len() >= 19 {
                                            ui.label(
                                                egui::RichText::new(
                                                    &msg.timestamp[..19].replace('T', " "),
                                                )
                                                .size(10.0)
                                                .color(TEXT_DIM),
                                            );
                                        }
                                    });
                                });
                            });
                            ui.add_space(4.0);
                        }
                    });

                    // Pagination
                    ui.separator();
                    ui.horizontal(|ui| {
                        let can_prev = self.current_page > 0;
                        let can_next = self.current_page + 1 < total_pages;
                        if ui
                            .add_enabled(can_prev, egui::Button::new("上一页"))
                            .clicked()
                        {
                            self.current_page -= 1;
                        }
                        ui.label(
                            egui::RichText::new(format!(
                                "第 {}/{} 页 (共 {} 条)",
                                self.current_page + 1,
                                total_pages,
                                total_msgs,
                            ))
                            .size(12.0)
                            .color(TEXT_DIM),
                        );
                        if ui
                            .add_enabled(can_next, egui::Button::new("下一页"))
                            .clicked()
                        {
                            self.current_page += 1;
                        }
                    });
                }
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label(
                        egui::RichText::new("选择一个会话查看对话记录")
                            .size(16.0)
                            .color(TEXT_DIM),
                    );
                });
            }
        });

        // ── 设置窗口 ──
        if self.show_settings {
            egui::Window::new("设置")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    egui::Grid::new("settings_grid")
                        .num_columns(2)
                        .spacing([10.0, 8.0])
                        .show(ui, |ui| {
                            ui.label("Embedding 模型:");
                            ui.text_edit_singleline(&mut self.settings_draft.embedding_model);
                            ui.end_row();

                            ui.label("模型源:");
                            egui::ComboBox::from_id_salt("model_source")
                                .selected_text(if self.settings_draft.model_source == "modelscope" {
                                    "ModelScope"
                                } else {
                                    "HuggingFace"
                                })
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut self.settings_draft.model_source,
                                        "modelscope".into(),
                                        "ModelScope (国内推荐)",
                                    );
                                    ui.selectable_value(
                                        &mut self.settings_draft.model_source,
                                        "huggingface".into(),
                                        "HuggingFace",
                                    );
                                });
                            ui.end_row();

                            ui.label("Embedding 维度:");
                            ui.add(egui::DragValue::new(&mut self.settings_draft.embedding_dimension).range(32..=1024));
                            ui.end_row();

                            ui.label("每页消息数:");
                            ui.add(egui::DragValue::new(&mut self.settings_draft.page_size).range(10..=500));
                            ui.end_row();
                        });

                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("保存").clicked() {
                            self.config = self.settings_draft.clone();
                            self.config.save();
                            self.show_settings = false;
                            self.status_msg = "设置已保存".into();
                        }
                        if ui.button("取消").clicked() {
                            self.show_settings = false;
                        }
                    });
                });
        }
    }
}

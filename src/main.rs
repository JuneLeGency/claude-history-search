mod app;
mod claude;
mod config;
mod engine;
mod parser;
mod qwen3;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_title("Claude Code 对话历史搜索"),
        ..Default::default()
    };
    eframe::run_native(
        "Claude History Search",
        options,
        Box::new(|cc| Ok(Box::new(app::App::new(cc)))),
    )
}

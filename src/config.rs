use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

pub fn claude_projects_dir() -> PathBuf {
    dirs::home_dir().unwrap().join(".claude").join("projects")
}

pub fn config_dir() -> PathBuf {
    dirs::home_dir().unwrap().join(".claude_his_search")
}

pub fn index_dir() -> PathBuf {
    config_dir().join("index")
}

fn config_file() -> PathBuf {
    config_dir().join("config.json")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub embedding_model: String,
    pub embedding_dimension: usize,
    pub model_source: String, // "huggingface" or "modelscope"
    pub page_size: usize,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            embedding_model: "Qwen/Qwen3-Embedding-0.6B".into(),
            embedding_dimension: 512,
            model_source: "huggingface".into(),
            page_size: 50,
        }
    }
}

impl AppConfig {
    pub fn load() -> Self {
        let path = config_file();
        if path.exists() {
            if let Ok(data) = fs::read_to_string(&path) {
                if let Ok(cfg) = serde_json::from_str(&data) {
                    return cfg;
                }
            }
        }
        Self::default()
    }

    pub fn save(&self) {
        let dir = config_dir();
        let _ = fs::create_dir_all(&dir);
        if let Ok(data) = serde_json::to_string_pretty(self) {
            let _ = fs::write(config_file(), data);
        }
    }
}

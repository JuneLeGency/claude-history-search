//! Embedding 引擎: 索引管理 + 模型推理 + 语义搜索

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

use crate::config::{index_dir, AppConfig};
use crate::parser::Session;
use crate::qwen3::{Qwen3Config, Qwen3Model};

fn index_emb_path() -> PathBuf {
    index_dir().join("embeddings.bin")
}
fn index_meta_path() -> PathBuf {
    index_dir().join("meta.json")
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IndexMeta {
    pub session_id: String,
    pub session_file: String,
    pub message_uuid: String,
    pub line_number: usize,
    pub role: String,
    pub timestamp: String,
    pub text_preview: String,
    pub project_path: String,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub meta: IndexMeta,
    pub score: f32,
}

/// 进度回调
pub type ProgressFn = Box<dyn Fn(usize, usize, &str) + Send>;

pub struct EmbeddingEngine {
    pub config: AppConfig,
    model: Option<Qwen3Model>,
    tokenizer: Option<Tokenizer>,
    device: Device,
    // 索引数据
    embeddings: Vec<f32>, // flat: n * dim
    meta: Vec<IndexMeta>,
    indexed_uuids: HashSet<String>,
    dimension: usize,
}

impl EmbeddingEngine {
    pub fn new(config: AppConfig) -> Self {
        let device = Device::new_metal(0).unwrap_or(Device::Cpu);
        let dimension = config.embedding_dimension;
        let mut engine = Self {
            config,
            model: None,
            tokenizer: None,
            device,
            embeddings: Vec::new(),
            meta: Vec::new(),
            indexed_uuids: HashSet::new(),
            dimension,
        };
        engine.load_index();
        engine
    }

    pub fn index_size(&self) -> usize {
        self.meta.len()
    }

    pub fn backend_name(&self) -> &str {
        match &self.device {
            Device::Cpu => "CPU",
            _ => "Metal",
        }
    }

    // ── Index persistence (binary f32 + JSON meta) ──

    fn load_index(&mut self) {
        let emb_path = index_emb_path();
        let meta_path = index_meta_path();
        if !emb_path.exists() || !meta_path.exists() {
            return;
        }
        if let Ok(meta_data) = fs::read_to_string(&meta_path) {
            if let Ok(meta) = serde_json::from_str::<Vec<IndexMeta>>(&meta_data) {
                if let Ok(emb_data) = fs::read(&emb_path) {
                    let floats: Vec<f32> = emb_data
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    if !meta.is_empty() {
                        self.dimension = floats.len() / meta.len();
                    }
                    self.indexed_uuids = meta.iter().map(|m| m.message_uuid.clone()).collect();
                    self.embeddings = floats;
                    self.meta = meta;
                }
            }
        }
    }

    fn save_index(&self) {
        let dir = index_dir();
        let _ = fs::create_dir_all(&dir);
        // binary f32
        let bytes: Vec<u8> = self
            .embeddings
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let _ = fs::write(index_emb_path(), bytes);
        // JSON meta
        if let Ok(data) = serde_json::to_string(&self.meta) {
            let _ = fs::write(index_meta_path(), data);
        }
    }

    // ── Model loading ──

    fn ensure_model(&mut self) -> Result<()> {
        if self.model.is_some() {
            return Ok(());
        }

        let model_dir = self.download_model()?;

        // Load config
        let config_path = model_dir.join("config.json");
        let config_str = fs::read_to_string(&config_path).context("读取 config.json 失败")?;
        let qwen_config: Qwen3Config = serde_json::from_str(&config_str)?;

        // Load tokenizer
        let tok_path = model_dir.join("tokenizer.json");
        let tokenizer =
            Tokenizer::from_file(&tok_path).map_err(|e| anyhow::anyhow!("tokenizer: {}", e))?;

        // Load model weights
        let safetensors_path = model_dir.join("model.safetensors");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&safetensors_path], DType::BF16, &self.device)?
        };
        let model = Qwen3Model::load(&qwen_config, vb, &self.device)?;

        self.model = Some(model);
        self.tokenizer = Some(tokenizer);
        Ok(())
    }

    fn download_model(&self) -> Result<PathBuf> {
        let model_id = &self.config.embedding_model;

        if self.config.model_source == "modelscope" {
            self.download_from_modelscope(model_id)
        } else {
            self.download_from_hf(model_id)
        }
    }

    fn download_from_hf(&self, model_id: &str) -> Result<PathBuf> {
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(model_id.to_string());
        // 下载必要文件 (hf-hub 会自动缓存)
        let _ = repo.get("config.json")?;
        let _ = repo.get("tokenizer.json")?;
        let _ = repo.get("model.safetensors")?;
        let dir = repo.get("config.json")?;
        Ok(dir.parent().unwrap().to_path_buf())
    }

    fn download_from_modelscope(&self, model_id: &str) -> Result<PathBuf> {
        let cache_dir = dirs::home_dir()
            .unwrap()
            .join(".cache/claude_his_search/models");
        let model_dir = cache_dir.join(model_id.replace('/', "--"));
        fs::create_dir_all(&model_dir)?;

        let files = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors",
        ];
        let base_url = format!(
            "https://modelscope.cn/models/{}/resolve/master",
            model_id
        );
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(600))
            .build()?;

        for file in &files {
            let dest = model_dir.join(file);
            if dest.exists() {
                continue;
            }
            let url = format!("{}/{}", base_url, file);
            let resp = client.get(&url).send()?;
            if !resp.status().is_success() {
                bail!("下载 {} 失败: {}", file, resp.status());
            }
            let bytes = resp.bytes()?;
            let mut f = fs::File::create(&dest)?;
            f.write_all(&bytes)?;
        }

        Ok(model_dir)
    }

    // ── Encode ──

    fn encode_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.ensure_model()?;
        let tokenizer = self.tokenizer.as_ref().unwrap();
        let model = self.model.as_ref().unwrap();

        // Tokenize
        let encodings = tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("tokenize: {}", e))?;

        let max_len = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);
        let max_len = max_len.min(512);
        let batch_size = encodings.len();

        let mut input_ids_flat = Vec::with_capacity(batch_size * max_len);
        let mut mask_flat = Vec::with_capacity(batch_size * max_len);

        for enc in &encodings {
            let ids = enc.get_ids();
            let len = ids.len().min(max_len);
            for i in 0..max_len {
                if i < len {
                    input_ids_flat.push(ids[i]);
                    mask_flat.push(1.0f32);
                } else {
                    input_ids_flat.push(0u32);
                    mask_flat.push(0.0f32);
                }
            }
        }

        let input_ids =
            Tensor::from_vec(input_ids_flat, (batch_size, max_len), &self.device)?;
        let attention_mask =
            Tensor::from_vec(mask_flat, (batch_size, max_len), &self.device)?;

        let embeddings = model.forward(&input_ids, &attention_mask)?;
        let embeddings = embeddings.to_dtype(DType::F32)?;

        // Truncate to target dimension + L2 normalize
        let dim = self.dimension.min(embeddings.dim(1)?);
        let embeddings = embeddings.narrow(1, 0, dim)?;
        let norms = embeddings
            .sqr()?
            .sum_keepdim(1)?
            .sqrt()?
            .clamp(1e-10, f32::MAX as f64)?;
        let embeddings = embeddings.broadcast_div(&norms)?;

        // Convert to Vec<Vec<f32>>
        let flat: Vec<f32> = embeddings.to_vec2()?
            .into_iter()
            .flatten()
            .collect();
        Ok(flat.chunks(dim).map(|c| c.to_vec()).collect())
    }

    // ── Build Index ──

    pub fn build_index(
        &mut self,
        sessions: &[Session],
        force_rebuild: bool,
        progress: Option<&ProgressFn>,
    ) -> Result<usize> {
        if force_rebuild {
            self.embeddings.clear();
            self.meta.clear();
            self.indexed_uuids.clear();
        }

        // Collect new chunks
        let mut new_chunks: Vec<(IndexMeta, String)> = Vec::new();
        for session in sessions {
            for msg in &session.messages {
                if self.indexed_uuids.contains(&msg.uuid) {
                    continue;
                }
                let text = msg.content.trim();
                if text.len() < 10 {
                    continue;
                }
                let text = if text.len() > 2000 {
                    &text[..2000]
                } else {
                    text
                };
                new_chunks.push((
                    IndexMeta {
                        session_id: session.session_id.clone(),
                        session_file: session.file_path.to_string_lossy().into(),
                        message_uuid: msg.uuid.clone(),
                        line_number: msg.line_number,
                        role: msg.role.clone(),
                        timestamp: msg.timestamp.clone(),
                        text_preview: text.chars().take(200).collect(),
                        project_path: session.project_path.clone(),
                    },
                    text.to_string(),
                ));
            }
        }

        if new_chunks.is_empty() {
            if let Some(p) = progress {
                p(0, 0, "索引已是最新，无新增消息");
            }
            return Ok(0);
        }

        let total = new_chunks.len();
        if let Some(p) = progress {
            p(0, total, "加载模型中...");
        }
        self.ensure_model()?;

        let batch_size = 8; // 较小 batch，Metal 显存友好
        let mut count = 0;

        for chunk_batch in new_chunks.chunks(batch_size) {
            let texts: Vec<String> = chunk_batch.iter().map(|(_, t)| t.clone()).collect();
            let vecs = self.encode_batch(&texts)?;

            for (i, (meta, _)) in chunk_batch.iter().enumerate() {
                self.embeddings.extend_from_slice(&vecs[i]);
                self.indexed_uuids.insert(meta.message_uuid.clone());
                self.meta.push(meta.clone());
            }

            count += chunk_batch.len();
            if let Some(p) = progress {
                p(count, total, &format!("[{}] 编码 {}/{}", self.backend_name(), count, total));
            }
        }

        self.save_index();
        Ok(count)
    }

    // ── Search ──

    pub fn search(&mut self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        if self.meta.is_empty() {
            return Ok(Vec::new());
        }

        self.ensure_model()?;
        let q_vecs = self.encode_batch(&[query.to_string()])?;
        let q_vec = &q_vecs[0];
        let dim = q_vec.len();

        // Dot product (vectors are normalized)
        let n = self.meta.len();
        let mut scores: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let start = i * dim;
                let end = start + dim;
                let doc = &self.embeddings[start..end];
                let score: f32 = q_vec.iter().zip(doc).map(|(a, b)| a * b).sum();
                (i, score)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);

        Ok(scores
            .into_iter()
            .map(|(i, score)| SearchResult {
                meta: self.meta[i].clone(),
                score,
            })
            .collect())
    }

    pub fn clear_index(&mut self) {
        self.embeddings.clear();
        self.meta.clear();
        self.indexed_uuids.clear();
        let _ = fs::remove_file(index_emb_path());
        let _ = fs::remove_file(index_meta_path());
    }
}

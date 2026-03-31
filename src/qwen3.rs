//! Qwen3 Embedding model — candle + Metal 实现
//!
//! 架构: Qwen3 decoder-only transformer, 用 last-token pooling 生成 embedding

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{Embedding, Linear, Module, VarBuilder};
use serde::Deserialize;

// ── Config ──

#[derive(Debug, Deserialize)]
pub struct Qwen3Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
}

fn default_head_dim() -> usize {
    64
}

// ── Components ──

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn load(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let variance = (&x * &x)?.mean_keepdim(D::Minus1)?;
        let x = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        Ok(x.to_dtype(dtype)?.broadcast_mul(&self.weight)?)
    }
}

struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(head_dim: usize, max_seq: usize, theta: f64, device: &Device) -> Result<Self> {
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / (theta as f32).powf(2.0 * i as f32 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, half), device)?;
        let positions: Vec<f32> = (0..max_seq).map(|p| p as f32).collect();
        let positions = Tensor::from_vec(positions, (max_seq, 1), device)?;
        let freqs = positions.matmul(&inv_freq)?;
        Ok(Self {
            cos: freqs.cos()?,
            sin: freqs.sin()?,
        })
    }

    fn apply(&self, x: &Tensor, seq_len: usize) -> Result<Tensor> {
        let half = x.dim(D::Minus1)? / 2;
        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;
        let cos = self
            .cos
            .narrow(0, 0, seq_len)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let sin = self
            .sin
            .narrow(0, 0, seq_len)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let y1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let y2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;
        Tensor::cat(&[y1, y2], D::Minus1).map_err(Into::into)
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn load(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let hd = cfg.head_dim;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        Ok(Self {
            q_proj: candle_nn::linear_no_bias(h, nh * hd, vb.pp("q_proj"))?,
            k_proj: candle_nn::linear_no_bias(h, nkv * hd, vb.pp("k_proj"))?,
            v_proj: candle_nn::linear_no_bias(h, nkv * hd, vb.pp("v_proj"))?,
            o_proj: candle_nn::linear_no_bias(nh * hd, h, vb.pp("o_proj"))?,
            q_norm: RmsNorm::load(hd, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: RmsNorm::load(hd, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
        })
    }

    fn forward(&self, x: &Tensor, rope: &RotaryEmbedding, mask: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let hd = self.head_dim;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b, s, self.num_heads, hd))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, s, self.num_kv_heads, hd))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, s, self.num_kv_heads, hd))?
            .transpose(1, 2)?;

        // QK-norm (per-head)
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // RoPE
        let q = rope.apply(&q, s)?;
        let k = rope.apply(&k, s)?;

        // GQA: repeat KV
        let n_rep = self.num_heads / self.num_kv_heads;
        let k = if n_rep > 1 {
            repeat_kv(&k, n_rep)?
        } else {
            k
        };
        let v = if n_rep > 1 {
            repeat_kv(&v, n_rep)?
        } else {
            v
        };

        // Scaled dot-product attention
        let scale = (hd as f64).sqrt();
        let attn = (q.matmul(&k.transpose(2, 3)?)? / scale)?;
        let attn = attn.broadcast_add(mask)?;
        let attn = candle_nn::ops::softmax(&attn, D::Minus1)?;
        let out = attn.matmul(&v)?;

        out.transpose(1, 2)?
            .reshape((b, s, self.num_heads * hd))?
            .apply(&self.o_proj)
            .map_err(Into::into)
    }
}

fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    let (b, h, s, d) = x.dims4()?;
    let mut heads = Vec::with_capacity(h * n_rep);
    for i in 0..h {
        let head = x.narrow(1, i, 1)?;
        for _ in 0..n_rep {
            heads.push(head.clone());
        }
    }
    Ok(Tensor::cat(&heads, 1)?.reshape((b, h * n_rep, s, d))?)
}

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn load(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        Ok(Self {
            gate_proj: candle_nn::linear_no_bias(h, i, vb.pp("gate_proj"))?,
            up_proj: candle_nn::linear_no_bias(h, i, vb.pp("up_proj"))?,
            down_proj: candle_nn::linear_no_bias(i, h, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.apply(&candle_nn::Activation::Silu)?;
        let up = self.up_proj.forward(x)?;
        Ok(self.down_proj.forward(&(gate * up)?)?)
    }
}

struct Block {
    input_layernorm: RmsNorm,
    self_attn: Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn load(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            input_layernorm: RmsNorm::load(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            self_attn: Attention::load(cfg, vb.pp("self_attn"))?,
            post_attention_layernorm: RmsNorm::load(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: Mlp::load(cfg, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, x: &Tensor, rope: &RotaryEmbedding, mask: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = (self.self_attn.forward(&x, rope, mask)? + residual)?;
        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        Ok((self.mlp.forward(&x)? + residual)?)
    }
}

// ── Full Model ──

pub struct Qwen3Model {
    embed_tokens: Embedding,
    layers: Vec<Block>,
    norm: RmsNorm,
    rope: RotaryEmbedding,
    device: Device,
}

impl Qwen3Model {
    pub fn load(cfg: &Qwen3Config, vb: VarBuilder, device: &Device) -> Result<Self> {
        let vb_model = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_model.pp("embed_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Block::load(
                cfg,
                vb_model.pp(format!("layers.{}", i)),
            )?);
        }
        let norm = RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb_model.pp("norm"))?;
        let rope = RotaryEmbedding::new(cfg.head_dim, 4096, cfg.rope_theta, device)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rope,
            device: device.clone(),
        })
    }

    /// Forward pass → last-token embedding (batch)
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let (_b, s) = input_ids.dims2()?;

        // Causal mask
        let mask = self.causal_mask(s, input_ids.dtype())?;

        let mut x = self.embed_tokens.forward(input_ids)?;

        for layer in &self.layers {
            x = layer.forward(&x, &self.rope, &mask)?;
        }
        x = self.norm.forward(&x)?;

        // Last-token pooling: take last non-padding token per sequence
        self.last_token_pool(&x, attention_mask)
    }

    fn causal_mask(&self, seq_len: usize, dtype: DType) -> Result<Tensor> {
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask_data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
        let mask = Tensor::from_vec(mask_data, (1, 1, seq_len, seq_len), &self.device)?;
        Ok(mask.to_dtype(dtype)?)
    }

    fn last_token_pool(&self, hidden: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let (b, _s, _h) = hidden.dims3()?;
        let mask_f32: Vec<Vec<f32>> = (0..b)
            .map(|i| {
                attention_mask
                    .get(i)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap_or_default()
            })
            .collect();

        let mut embeddings = Vec::with_capacity(b);
        for i in 0..b {
            let last_pos = mask_f32[i]
                .iter()
                .rposition(|&v| v > 0.0)
                .unwrap_or(0);
            embeddings.push(hidden.get(i)?.get(last_pos)?);
        }
        Tensor::stack(&embeddings, 0).map_err(Into::into)
    }
}

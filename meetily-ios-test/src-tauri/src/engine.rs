// Offline engine: whisper-rs (transcription) + llama.cpp (summary), in-process.
// CPU-only so it runs in the iOS simulator without Metal.
//
// whisper-rs 0.16 + llama-cpp-2 0.1.150 share a compatible ggml ABI, so both
// engines can be statically linked into one binary without the runtime symbol
// clash that 0.13.2 caused.

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use serde::Serialize;
use tauri::{AppHandle, Emitter};

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;

const SAMPLE_RATE: usize = 16_000;
/// Window fed to whisper per step. Larger = more context/quality, fewer UI ticks.
const CHUNK_SECS: f32 = 8.0;

#[derive(Clone, Serialize)]
pub struct TranscriptUpdate {
    pub chunk_index: usize,
    pub chunk_text: String,
    pub full_text: String,
    pub progress: f32,
    pub detected_language: String,
}

/// Read a 16kHz mono WAV into f32 samples.
fn read_wav(path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.sample_rate != SAMPLE_RATE as u32 || spec.channels != 1 {
        return Err(anyhow!(
            "expected 16kHz mono, got {}Hz {}ch",
            spec.sample_rate,
            spec.channels
        ));
    }
    let samples = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / 32768.0))
            .collect::<std::result::Result<_, _>>()?,
        hound::SampleFormat::Float => {
            reader.samples::<f32>().collect::<std::result::Result<_, _>>()?
        }
    };
    Ok(samples)
}

/// Stream-transcribe a WAV file chunk by chunk, emitting `transcript-update`
/// events to simulate live captioning. Stops early when `stop` is set.
/// Returns the full accumulated transcript.
pub fn transcribe_streaming(
    app: &AppHandle,
    model_path: &Path,
    wav_path: &Path,
    stop: Arc<AtomicBool>,
    transcript_store: Arc<Mutex<String>>,
) -> Result<String> {
    // ggml-metal's new tensor-API path (auto-enabled on A19/M5+ GPUs) produces
    // NaN output in this ggml snapshot — every token decodes to id 0 / p=0.0.
    // Forcing it off makes the A19 use the classic Metal path, which is correct
    // (verified on host M3, where tensor is already off). Harmless on CPU builds.
    std::env::set_var("GGML_METAL_TENSOR_DISABLE", "1");

    let mode = if cfg!(feature = "metal") { "GPU(Metal)" } else { "CPU" };
    let _ = app.emit("engine-status", format!("{mode} · 모델 로딩 중…"));

    let mut ctx_params = WhisperContextParameters::default();
    // GPU (Metal) on device builds (`--features metal`); CPU in the simulator.
    ctx_params.use_gpu(cfg!(feature = "metal"));
    let load_t = std::time::Instant::now();
    let ctx = WhisperContext::new_with_params(
        model_path.to_str().ok_or_else(|| anyhow!("bad model path"))?,
        ctx_params,
    )?;
    let load_s = load_t.elapsed().as_secs_f32();
    let _ = app.emit("engine-status", format!("{mode} · 모델 로드 {load_s:.1}s · 전사 시작…"));

    let samples = read_wav(wav_path)?;
    let chunk_len = (CHUNK_SECS * SAMPLE_RATE as f32) as usize;
    let total_chunks = (samples.len() + chunk_len - 1) / chunk_len;

    let mut full_text = String::new();
    let mut detected_language = String::from("auto");

    for (idx, chunk) in samples.chunks(chunk_len).enumerate() {
        if stop.load(Ordering::SeqCst) {
            break;
        }

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        // Language AUTO-DETECT — never hardcode (lesson from the benchmark).
        params.set_language(Some("auto"));
        params.set_translate(false);
        params.set_n_threads(6); // A19: use more cores on the CPU path
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_blank(true);
        params.set_temperature(0.0);

        let mut state = ctx.create_state()?;
        let chunk_t = std::time::Instant::now();
        state.full(params, chunk)?;
        let chunk_ms = chunk_t.elapsed().as_millis();
        let _ = app.emit(
            "engine-status",
            format!("{mode} · 청크 {}/{} · {chunk_ms}ms/청크", idx + 1, total_chunks),
        );

        // Accumulate raw bytes across segments, decode UTF-8 once so multi-byte
        // Korean characters split at segment boundaries recombine correctly.
        let n = state.full_n_segments();
        let mut bytes: Vec<u8> = Vec::new();
        for i in 0..n {
            if let Some(seg) = state.get_segment(i) {
                if let Ok(b) = seg.to_bytes() {
                    bytes.extend_from_slice(b);
                }
            }
        }
        let lang_id = state.full_lang_id_from_state();
        if let Some(lang) = whisper_rs::get_lang_str(lang_id) {
            detected_language = lang.to_string();
        }
        let chunk_text = String::from_utf8_lossy(&bytes).trim().to_string();

        if !chunk_text.is_empty() {
            if !full_text.is_empty() {
                full_text.push(' ');
            }
            full_text.push_str(&chunk_text);
        }

        // Commit progress to the shared store every chunk so that pressing Stop
        // mid-transcription always has the latest text available to summarize.
        *transcript_store.lock().unwrap() = full_text.clone();

        let _ = app.emit(
            "transcript-update",
            TranscriptUpdate {
                chunk_index: idx,
                chunk_text,
                full_text: full_text.clone(),
                progress: ((idx + 1) as f32 / total_chunks as f32).min(1.0),
                detected_language: detected_language.clone(),
            },
        );
    }

    Ok(full_text)
}

/// Qwen2.5 chat template (non-thinking). Returns (prompt, korean_prefill).
/// Small Qwen models drift to Chinese; a strict Korean-only system prompt plus a
/// Korean assistant prefill keeps the output in Korean.
fn build_qwen_prompt(transcript: &str) -> (String, String) {
    let system = "당신은 한국어 전문 회의 비서입니다. 반드시 한국어(한글)로만 회의록을 작성하세요. \
중국어(汉语)나 영어로 작성하면 안 됩니다. 아래 회의 전사록을 바탕으로 다음 형식의 회의록을 작성하세요: \
1) 회의 개요  2) 핵심 논의 사항  3) 결정 사항  4) 후속 조치. \
전사록에 없는 내용을 지어내지 말고 숫자·고유명사는 보존하세요.";
    // 7B with an 8K context can digest a much longer transcript than the small
    // CPU demo (which clipped to 1800). Keep headroom for the system prompt + output.
    let clipped: String = transcript.chars().take(5000).collect();
    let prefill = "# 회의록\n\n## 1. 회의 개요\n";
    let prompt = format!(
        "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n다음은 회의 전사록입니다:\n\n{clipped}<|im_end|>\n<|im_start|>assistant\n{prefill}"
    );
    (prompt, prefill.to_string())
}

/// Summarize a transcript with the bundled Qwen GGUF, fully in-process on CPU.
pub fn summarize(model_path: &Path, transcript: &str) -> Result<String> {
    // Same Metal tensor-API workaround as transcribe_streaming (see note there).
    std::env::set_var("GGML_METAL_TENSOR_DISABLE", "1");

    let backend = LlamaBackend::init()?;
    // Offload all layers to the GPU (Metal) on device; stay on CPU in the simulator.
    let n_gpu_layers = if cfg!(feature = "metal") { 99 } else { 0 };
    let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)?;

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(8192))
        .with_n_batch(512);
    let mut ctx = model.new_context(&backend, ctx_params)?;

    let (prompt, prefill) = build_qwen_prompt(transcript);
    let tokens = model.str_to_token(&prompt, AddBos::Never)?;

    let mut batch = LlamaBatch::new(2048, 1);
    let last = tokens.len() - 1;
    for (i, tok) in tokens.iter().enumerate() {
        batch.add(*tok, i as i32, &[0], i == last)?;
    }
    ctx.decode(&mut batch)?;

    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::temp(0.4), LlamaSampler::greedy()]);

    let mut out = prefill; // show the Korean prefill as part of the result
    let mut n_cur = batch.n_tokens();
    let max_new = 600i32;
    let mut produced = 0i32;

    while produced < max_new {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);
        if model.is_eog_token(token) {
            break;
        }
        if let Ok(piece) = model.token_to_str(token, Special::Plaintext) {
            out.push_str(&piece);
        }
        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        n_cur += 1;
        ctx.decode(&mut batch)?;
        produced += 1;
    }

    Ok(out.trim().to_string())
}

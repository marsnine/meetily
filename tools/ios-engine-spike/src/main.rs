// Host-side tester. Two modes:
//   spike-run coexist <whisper.bin> <llama.gguf>   — reproduce/verify ggml clash fix
//   spike-run sum <llama.gguf> <transcript.txt>     — iterate the Korean summary prompt fast
//
// The `sum` mode mirrors engine::summarize so prompt tuning happens on the host
// (seconds) instead of via slow iOS rebuilds.

use std::num::NonZeroU32;
use std::path::Path;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;

fn build_qwen_prompt(transcript: &str) -> (String, String) {
    let system = "당신은 한국어 전문 회의 비서입니다. 반드시 한국어(한글)로만 회의록을 작성하세요. \
중국어(汉语)나 영어로 작성하면 안 됩니다. 아래 회의 전사록을 바탕으로 다음 형식의 회의록을 작성하세요: \
1) 회의 개요  2) 핵심 논의 사항  3) 결정 사항  4) 후속 조치. \
전사록에 없는 내용을 지어내지 말고 숫자·고유명사는 보존하세요.";
    let clipped: String = transcript.chars().take(1800).collect();
    // Prefill the assistant turn with Korean so the model continues in Korean.
    let prefill = "# 회의록\n\n## 1. 회의 개요\n";
    let prompt = format!(
        "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n다음은 회의 전사록입니다:\n\n{clipped}<|im_end|>\n<|im_start|>assistant\n{prefill}"
    );
    (prompt, prefill.to_string())
}

fn summarize(model_path: &Path, transcript: &str) -> String {
    let backend = LlamaBackend::init().expect("backend");
    let mp = LlamaModelParams::default().with_n_gpu_layers(0);
    let model = LlamaModel::load_from_file(&backend, model_path, &mp).expect("load");
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(4096))
        .with_n_batch(2048);
    let mut ctx = model.new_context(&backend, ctx_params).expect("ctx");

    let (prompt, prefill) = build_qwen_prompt(transcript);
    let tokens = model.str_to_token(&prompt, AddBos::Never).expect("tok");
    let mut batch = LlamaBatch::new(2048, 1);
    let last = tokens.len() - 1;
    for (i, t) in tokens.iter().enumerate() {
        batch.add(*t, i as i32, &[0], i == last).unwrap();
    }
    ctx.decode(&mut batch).expect("decode");

    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::temp(0.3), LlamaSampler::greedy()]);
    let mut out = prefill;
    let mut n_cur = batch.n_tokens();
    for _ in 0..500 {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);
        if model.is_eog_token(token) {
            break;
        }
        if let Ok(p) = model.token_to_str(token, Special::Plaintext) {
            out.push_str(&p);
        }
        batch.clear();
        batch.add(token, n_cur, &[0], true).unwrap();
        n_cur += 1;
        ctx.decode(&mut batch).expect("decode2");
    }
    out
}

fn read_wav_16k_mono(path: &Path) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("open wav");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16_000, "expected 16kHz");
    assert_eq!(spec.channels, 1, "expected mono");
    match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect(),
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
    }
}

/// Transcribe the first ~24s of a WAV and print the text. `gpu` toggles Metal.
/// Mirrors engine::transcribe_streaming so we can compare CPU vs Metal numerics
/// on the host before doing slow on-device rebuilds.
fn transcribe(model_path: &Path, wav: &Path, gpu: bool) {
    use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};
    let mut cp = WhisperContextParameters::default();
    cp.use_gpu(gpu);
    println!("[trans] use_gpu={gpu}; loading model...");
    let ctx = WhisperContext::new_with_params(model_path.to_str().unwrap(), cp).expect("load");
    let samples = read_wav_16k_mono(wav);
    println!("[trans] {} samples ({:.1}s)", samples.len(), samples.len() as f32 / 16000.0);
    let chunk = 8 * 16_000usize;
    for (idx, c) in samples.chunks(chunk).take(3).enumerate() {
        let mut p = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        p.set_language(Some("auto"));
        p.set_translate(false);
        p.set_n_threads(4);
        p.set_print_special(false);
        p.set_print_progress(false);
        p.set_print_realtime(false);
        p.set_print_timestamps(false);
        let mut st = ctx.create_state().expect("state");
        st.full(p, c).expect("full");
        let n = st.full_n_segments();
        let mut bytes: Vec<u8> = Vec::new();
        for i in 0..n {
            if let Some(seg) = st.get_segment(i) {
                if let Ok(b) = seg.to_bytes() {
                    bytes.extend_from_slice(b);
                }
            }
        }
        let lang = whisper_rs::get_lang_str(st.full_lang_id_from_state()).unwrap_or("?");
        let text = String::from_utf8_lossy(&bytes);
        println!("[chunk {idx}] lang={lang} text={:?}", text.trim());
    }
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_default();
    if mode == "trans" {
        let model = std::env::args().nth(2).expect("arg2: whisper ggml .bin");
        let wav = std::env::args().nth(3).expect("arg3: 16k mono wav");
        let gpu = std::env::args().nth(4).as_deref() == Some("gpu");
        transcribe(Path::new(&model), Path::new(&wav), gpu);
        return;
    }
    if mode == "sum" {
        let model = std::env::args().nth(2).expect("arg2: llama gguf");
        let tpath = std::env::args().nth(3).expect("arg3: transcript.txt");
        let transcript = std::fs::read_to_string(&tpath).expect("read transcript");
        let s = summarize(Path::new(&model), &transcript);
        println!("=== SUMMARY OUTPUT ===\n{s}");
        return;
    }

    // coexist mode
    let whisper_model = std::env::args().nth(2).expect("arg2: whisper ggml");
    let llama_model = std::env::args().nth(3);
    let backend = LlamaBackend::init().expect("llama backend init");
    println!("[1] llama backend ok");
    if let Some(lm) = &llama_model {
        let mp = LlamaModelParams::default().with_n_gpu_layers(0);
        let model = LlamaModel::load_from_file(&backend, Path::new(lm), &mp).expect("llama load");
        let _ctx = model
            .new_context(&backend, LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512)))
            .expect("llama ctx");
        println!("[2] llama model ok");
    }
    let _wctx = ios_engine_spike::load_whisper(&whisper_model).expect("whisper load");
    println!("[3] whisper model ok\n✅ BOTH ENGINES COEXIST");
}

// ASR harness: replicates meetily's whisper_engine transcription path so text-extraction
// and prompting changes can be A/B tested against real audio without building the app.
//
// Usage:
//   asr-harness --model <ggml.bin> --wav <16k-mono.wav> --mode legacy|fixed \
//               [--vocab <vocab.txt>] [--carry] [--chunk-secs 25] [--max-secs 120]
//
// Modes:
//   legacy : per-segment full_get_segment_text_lossy() + ' ' join (current meetily behavior)
//   fixed  : accumulate full_get_segment_bytes() across segments, decode once
// Options:
//   --vocab : file whose contents are passed as initial_prompt (domain vocabulary biasing)
//   --carry : append the tail of the previous chunk's transcript to the prompt

use anyhow::{anyhow, Context, Result};
use std::io::Write as _;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

struct Args {
    model: String,
    wav: String,
    mode: String,
    vocab: Option<String>,
    carry: bool,
    chunk_secs: f32,
    max_secs: Option<f32>,
    language: String,
}

fn parse_args() -> Result<Args> {
    let mut args = Args {
        model: String::new(),
        wav: String::new(),
        mode: "fixed".to_string(),
        vocab: None,
        carry: false,
        chunk_secs: 25.0,
        max_secs: None,
        language: "ko".to_string(),
    };
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => { args.model = argv[i + 1].clone(); i += 2; }
            "--wav" => { args.wav = argv[i + 1].clone(); i += 2; }
            "--mode" => { args.mode = argv[i + 1].clone(); i += 2; }
            "--vocab" => { args.vocab = Some(argv[i + 1].clone()); i += 2; }
            "--carry" => { args.carry = true; i += 1; }
            "--chunk-secs" => { args.chunk_secs = argv[i + 1].parse()?; i += 2; }
            "--max-secs" => { args.max_secs = Some(argv[i + 1].parse()?); i += 2; }
            "--language" => { args.language = argv[i + 1].clone(); i += 2; }
            other => return Err(anyhow!("unknown arg: {}", other)),
        }
    }
    if args.model.is_empty() || args.wav.is_empty() {
        return Err(anyhow!("--model and --wav are required"));
    }
    Ok(args)
}

fn read_wav_f32(path: &str) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path).context("open wav")?;
    let spec = reader.spec();
    if spec.sample_rate != 16000 || spec.channels != 1 {
        return Err(anyhow!("expected 16kHz mono wav, got {}Hz {}ch", spec.sample_rate, spec.channels));
    }
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / 32768.0))
            .collect::<std::result::Result<_, _>>()?,
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<_, _>>()?,
    };
    Ok(samples)
}

fn main() -> Result<()> {
    let args = parse_args()?;

    let vocab = match &args.vocab {
        Some(p) => Some(std::fs::read_to_string(p).context("read vocab")?.trim().to_string()),
        None => None,
    };

    let mut samples = read_wav_f32(&args.wav)?;
    if let Some(max) = args.max_secs {
        let max_len = (max * 16000.0) as usize;
        samples.truncate(max_len);
    }
    eprintln!(
        "audio: {:.1}s, mode={}, vocab={}, carry={}",
        samples.len() as f32 / 16000.0,
        args.mode,
        vocab.is_some(),
        args.carry
    );

    let ctx = WhisperContext::new_with_params(&args.model, WhisperContextParameters::default())
        .context("load model")?;

    let chunk_len = (args.chunk_secs * 16000.0) as usize;
    let mut prev_tail = String::new();

    for (idx, chunk) in samples.chunks(chunk_len).enumerate() {
        let t0 = idx as f32 * args.chunk_secs;

        // Params mirror whisper_engine.rs transcribe_audio_with_confidence (Ultra tier)
        let mut params = FullParams::new(SamplingStrategy::BeamSearch { beam_size: 5, patience: 1.0 });
        params.set_language(Some(&args.language));
        params.set_translate(false);
        params.set_no_timestamps(true);
        params.set_token_timestamps(true);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_blank(true);
        params.set_suppress_non_speech_tokens(true);
        params.set_temperature(0.1);
        params.set_max_initial_ts(1.0);
        params.set_entropy_thold(2.4);
        params.set_logprob_thold(-1.0);
        params.set_no_speech_thold(0.55);
        params.set_max_len(200);
        params.set_single_segment(false);

        // Step 2: vocabulary biasing + previous-context carry-over via initial_prompt
        let mut prompt = String::new();
        if let Some(v) = &vocab {
            prompt.push_str(v);
        }
        if args.carry && !prev_tail.is_empty() {
            if !prompt.is_empty() {
                prompt.push('\n');
            }
            prompt.push_str(&prev_tail);
        }
        if !prompt.is_empty() {
            params.set_initial_prompt(&prompt);
        }

        let mut state = ctx.create_state()?;
        state.full(params, chunk)?;
        let num_segments = state.full_n_segments()?;

        let text = match args.mode.as_str() {
            "legacy" => {
                // Current meetily behavior: lossy per segment, joined with spaces.
                let mut result = String::new();
                for i in 0..num_segments {
                    let segment_text = match state.full_get_segment_text_lossy(i) {
                        Ok(t) => t,
                        Err(_) => continue,
                    };
                    let cleaned = segment_text.trim();
                    if !cleaned.is_empty() {
                        if !result.is_empty() {
                            result.push(' ');
                        }
                        result.push_str(cleaned);
                    }
                }
                result
            }
            "fixed" => {
                // Fix: accumulate raw bytes across segments, decode UTF-8 once so
                // multi-byte characters split at segment boundaries recombine.
                let mut bytes: Vec<u8> = Vec::new();
                for i in 0..num_segments {
                    if let Ok(seg) = state.full_get_segment_bytes(i) {
                        bytes.extend_from_slice(&seg);
                    }
                }
                String::from_utf8_lossy(&bytes).trim().to_string()
            }
            other => return Err(anyhow!("unknown mode: {}", other)),
        };

        if args.carry {
            // Keep roughly the last 200 chars as context for the next chunk.
            let tail_chars: Vec<char> = text.chars().collect();
            let start = tail_chars.len().saturating_sub(200);
            prev_tail = tail_chars[start..].iter().collect();
        }

        println!("[{:>4.0}s] {}", t0, text);
        std::io::stdout().flush().ok();
    }

    Ok(())
}

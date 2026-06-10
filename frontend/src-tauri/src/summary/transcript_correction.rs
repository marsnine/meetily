/// Vocabulary-based transcript correction pass.
///
/// ASR vocabulary biasing (initial_prompt) catches most proper-noun errors, but some
/// still slip through (e.g. "인젤투자협회" instead of "엔젤투자협회"). Before a transcript
/// is summarized, this pass asks the configured LLM to fix only words that are clearly
/// misrecognitions of user-provided vocabulary terms, leaving everything else untouched.
///
/// The pass is best-effort by design: any LLM failure, cancellation, or implausible
/// output falls back to the original text, so summary generation can never be broken
/// by the correction step.
use reqwest::Client;
use std::path::PathBuf;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::summary::llm_client::{generate_summary, LLMProvider};
use crate::summary::processor::{clean_llm_markdown_output, rough_token_count};

const CORRECTION_SYSTEM_PROMPT: &str = "You are a transcript correction engine. \
You receive a list of domain vocabulary terms (proper nouns, company names, jargon) \
and a raw speech-to-text transcript that may contain misrecognitions of those terms. \
Rewrite the transcript, replacing only words or phrases that are clearly failed \
recognitions of one of the vocabulary terms (similar sound, similar spelling) with \
the correct term. Keep every other word, the word order, punctuation, line breaks, \
and the language of the transcript exactly as they are. Never translate, summarize, \
shorten, expand, or comment. Output only the corrected transcript text, nothing else.";

/// A corrected chunk is only accepted when its length stays in this ratio band
/// relative to the input chunk; outside of it the model has likely summarized,
/// truncated, or padded the text instead of correcting it.
const MIN_OUTPUT_RATIO: f64 = 0.5;
const MAX_OUTPUT_RATIO: f64 = 1.5;

/// Split `text` into gapless chunks of at most `chunk_size_tokens`, cutting at
/// sentence or word boundaries where possible. Unlike `processor::chunk_text`
/// (whose stride is independent of the boundary adjustment and which is meant to
/// be used with an overlap), consecutive chunks here always continue exactly
/// where the previous one ended, so joining them reproduces the original text.
fn split_gapless(text: &str, chunk_size_tokens: usize) -> Vec<String> {
    let chars_per_token = 1.0 / 0.35; // same heuristic as processor::chunk_text
    let chunk_size_chars = ((chunk_size_tokens as f64) * chars_per_token).ceil() as usize;
    let chars: Vec<char> = text.chars().collect();

    if chars.len() <= chunk_size_chars {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut start = 0usize; // char index
    while start < chars.len() {
        let window_end = (start + chunk_size_chars).min(chars.len());
        let mut end = window_end;
        if window_end < chars.len() {
            // Prefer a sentence boundary, then any whitespace, scanning backwards.
            let mut sentence_end = None;
            let mut space_end = None;
            for i in (start + 1..window_end).rev() {
                let c = chars[i];
                if space_end.is_none() && c.is_whitespace() {
                    space_end = Some(i + 1);
                }
                if matches!(c, '.' | '!' | '?' | '\n' | '。')
                    && (i + 1 == chars.len() || chars[i + 1].is_whitespace())
                {
                    sentence_end = Some(i + 1);
                    break;
                }
            }
            end = sentence_end.or(space_end).unwrap_or(window_end);
        }
        chunks.push(chars[start..end].iter().collect());
        start = end;
    }
    chunks
}

/// Correct `transcript` against the user vocabulary using the same LLM provider that
/// will generate the summary. Returns the corrected transcript, or the original text
/// unchanged if the vocabulary is empty or any part of the pass fails.
#[allow(clippy::too_many_arguments)]
pub async fn correct_transcript_with_vocabulary(
    client: &Client,
    provider: &LLMProvider,
    model_name: &str,
    api_key: &str,
    transcript: &str,
    vocabulary: &str,
    token_threshold: usize,
    ollama_endpoint: Option<&str>,
    custom_openai_endpoint: Option<&str>,
    app_data_dir: Option<&PathBuf>,
    cancellation_token: Option<&CancellationToken>,
) -> String {
    let vocabulary = vocabulary.trim();
    if vocabulary.is_empty() || transcript.trim().is_empty() {
        return transcript.to_string();
    }

    // Input and output occupy the same context window, plus vocabulary and prompt
    // overhead, so each chunk can use at most roughly half of what remains.
    let overhead_tokens = rough_token_count(vocabulary) + rough_token_count(CORRECTION_SYSTEM_PROMPT) + 100;
    let available = token_threshold.saturating_sub(overhead_tokens);
    let chunk_tokens = (available / 2).clamp(300, 6000);

    let chunks = split_gapless(transcript, chunk_tokens);
    info!(
        "📝 Transcript correction pass: {} chunk(s) of ≤{} tokens against {} vocabulary chars",
        chunks.len(),
        chunk_tokens,
        vocabulary.len()
    );

    let mut corrected = String::with_capacity(transcript.len());
    let mut corrected_chunks = 0usize;
    for (i, chunk) in chunks.iter().enumerate() {
        if cancellation_token.map_or(false, |t| t.is_cancelled()) {
            info!("Transcript correction cancelled; using original transcript");
            return transcript.to_string();
        }

        let user_prompt = format!(
            "Vocabulary terms:\n{}\n\nTranscript:\n{}",
            vocabulary, chunk
        );

        match generate_summary(
            client,
            provider,
            model_name,
            api_key,
            CORRECTION_SYSTEM_PROMPT,
            &user_prompt,
            ollama_endpoint,
            custom_openai_endpoint,
            None,
            None,
            None,
            app_data_dir,
            cancellation_token,
        )
        .await
        {
            Ok(output) => {
                let cleaned = clean_llm_markdown_output(&output);
                let cleaned = cleaned.trim();
                let ratio = cleaned.chars().count() as f64 / chunk.chars().count().max(1) as f64;
                if cleaned.is_empty() || ratio < MIN_OUTPUT_RATIO || ratio > MAX_OUTPUT_RATIO {
                    warn!(
                        "Transcript correction chunk {}/{} rejected (length ratio {:.2}); keeping original chunk",
                        i + 1,
                        chunks.len(),
                        ratio
                    );
                    corrected.push_str(chunk);
                } else {
                    // The model output is trimmed, so restore the chunk's original
                    // leading/trailing whitespace to keep seams between chunks intact.
                    let leading: String =
                        chunk.chars().take_while(|c| c.is_whitespace()).collect();
                    let trailing: String = chunk
                        .chars()
                        .rev()
                        .take_while(|c| c.is_whitespace())
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .collect();
                    corrected.push_str(&leading);
                    corrected.push_str(cleaned);
                    corrected.push_str(&trailing);
                    corrected_chunks += 1;
                }
            }
            Err(e) => {
                warn!(
                    "Transcript correction chunk {}/{} failed ({}); keeping original chunk",
                    i + 1,
                    chunks.len(),
                    e
                );
                corrected.push_str(chunk);
            }
        }
    }

    if corrected_chunks == 0 {
        info!("Transcript correction made no changes; using original transcript");
        return transcript.to_string();
    }

    info!(
        "📝 Transcript correction pass done: {}/{} chunk(s) corrected",
        corrected_chunks,
        chunks.len()
    );
    corrected
}

#[cfg(test)]
mod tests {
    use super::split_gapless;

    #[test]
    fn split_gapless_reassembles_to_original() {
        let text = "안녕하세요. 빅뱅엔젤스 미팅입니다. 팁스 운영사 추천서 관련 논의를 시작하겠습니다. "
            .repeat(50);
        let chunks = split_gapless(&text, 100);
        assert!(chunks.len() > 1);
        assert_eq!(chunks.concat(), text);
    }

    #[test]
    fn short_text_is_single_chunk() {
        let chunks = split_gapless("짧은 텍스트", 1000);
        assert_eq!(chunks, vec!["짧은 텍스트".to_string()]);
    }
}

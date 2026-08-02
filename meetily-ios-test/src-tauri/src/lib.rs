mod engine;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State};

/// Shared app state across commands.
#[derive(Default)]
struct AppState {
    stop_flag: Arc<AtomicBool>,
    transcript: Arc<Mutex<String>>,
}

#[derive(Clone, Serialize)]
struct AudioFile {
    name: String,
    path: String,
}

#[derive(Clone, Serialize)]
struct SummaryReady {
    text: String,
    elapsed_secs: f32,
}

#[derive(Clone, Serialize)]
struct EngineError {
    stage: String,
    message: String,
}

/// Resolve a path inside the bundled resources directory (assets/...).
fn resource_path(app: &AppHandle, rel: &str) -> Result<PathBuf, String> {
    let dir = app
        .path()
        .resource_dir()
        .map_err(|e| format!("resource_dir: {e}"))?;
    Ok(dir.join(rel))
}

/// List the bundled test audio clips.
#[tauri::command]
fn list_audio_files(app: AppHandle) -> Result<Vec<AudioFile>, String> {
    let audio_dir = resource_path(&app, "assets/audio")?;
    let mut files = Vec::new();
    let entries = std::fs::read_dir(&audio_dir)
        .map_err(|e| format!("read_dir {}: {e}", audio_dir.display()))?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("wav") {
            if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                files.push(AudioFile {
                    name: name.to_string(),
                    path: path.to_string_lossy().to_string(),
                });
            }
        }
    }
    files.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(files)
}

/// Start streaming transcription of the given bundled file. Emits
/// `transcript-update` events; on completion emits `transcription-done`.
#[tauri::command]
fn start_transcription(
    app: AppHandle,
    state: State<AppState>,
    file_name: String,
) -> Result<(), String> {
    state.stop_flag.store(false, Ordering::SeqCst);
    *state.transcript.lock().unwrap() = String::new();

    let stop = state.stop_flag.clone();
    let transcript_store = state.transcript.clone();
    let app_handle = app.clone();

    let model_path = resource_path(&app, "assets/models/ggml-large-v3-turbo.bin")?;
    let wav_path = resource_path(&app, &format!("assets/audio/{file_name}"))?;

    // Whisper inference is blocking CPU work — run off the async path.
    std::thread::spawn(move || {
        match engine::transcribe_streaming(&app_handle, &model_path, &wav_path, stop, transcript_store.clone()) {
            Ok(text) => {
                *transcript_store.lock().unwrap() = text.clone();
                let _ = app_handle.emit("transcription-done", text);
            }
            Err(e) => {
                let _ = app_handle.emit(
                    "engine-error",
                    EngineError { stage: "transcription".into(), message: e.to_string() },
                );
            }
        }
    });
    Ok(())
}

/// Stop transcription (if running) and summarize the accumulated transcript.
/// Emits `summary-ready` (or `engine-error`).
#[tauri::command]
fn stop_and_summarize(app: AppHandle, state: State<AppState>) -> Result<(), String> {
    state.stop_flag.store(true, Ordering::SeqCst);

    let transcript_store = state.transcript.clone();
    let app_handle = app.clone();
    let model_path = resource_path(&app, "assets/models/qwen2.5-7b-instruct-q4_k_m.gguf")?;

    std::thread::spawn(move || {
        // Poll up to ~6s for the transcription thread to commit at least one chunk
        // (a whisper chunk can take a few seconds), so Stop never races to empty.
        let mut transcript = String::new();
        for _ in 0..20 {
            std::thread::sleep(std::time::Duration::from_millis(300));
            transcript = transcript_store.lock().unwrap().clone();
            if !transcript.trim().is_empty() {
                break;
            }
        }
        if transcript.trim().is_empty() {
            let _ = app_handle.emit(
                "engine-error",
                EngineError { stage: "summary".into(), message: "전사 내용이 비어 있습니다.".into() },
            );
            return;
        }
        let _ = app_handle.emit("summary-started", ());
        let t0 = std::time::Instant::now();
        match engine::summarize(&model_path, &transcript) {
            Ok(text) => {
                let _ = app_handle.emit(
                    "summary-ready",
                    SummaryReady { text, elapsed_secs: t0.elapsed().as_secs_f32() },
                );
            }
            Err(e) => {
                let _ = app_handle.emit(
                    "engine-error",
                    EngineError { stage: "summary".into(), message: e.to_string() },
                );
            }
        }
    });
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            list_audio_files,
            start_transcription,
            stop_and_summarize
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

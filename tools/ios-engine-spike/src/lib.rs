// iOS engine feasibility spike.
//
// This references real whisper-rs symbols so that a successful build for
// aarch64-apple-ios-sim proves whisper.cpp actually cross-compiles AND links
// for the iOS simulator (not just that the Rust shim type-checks).

use llama_cpp_2::llama_backend::LlamaBackend;
use whisper_rs::{WhisperContext, WhisperContextParameters};

/// Initialize the llama.cpp backend in-process. Referencing this forces
/// llama-cpp-sys-2 (llama.cpp) to be compiled and linked for the target,
/// proving the summary engine can run in-process on iOS (no sidecar).
pub fn init_llama() -> Result<LlamaBackend, String> {
    LlamaBackend::init().map_err(|e| format!("llama backend init failed: {e}"))
}

/// Attempt to load a Whisper ggml model. We don't call this in the build check;
/// its mere presence forces whisper-rs-sys (whisper.cpp) to be compiled and
/// linked into the staticlib for the target.
pub fn load_whisper(model_path: &str) -> Result<WhisperContext, String> {
    WhisperContext::new_with_params(model_path, WhisperContextParameters::default())
        .map_err(|e| format!("whisper load failed: {e}"))
}

/// Exported C symbol so the staticlib has a stable entry point an iOS app
/// (or a smoke test) could call.
#[no_mangle]
pub extern "C" fn ios_engine_spike_ping() -> i32 {
    42
}

# Qwen3-ASR 엔진 추가 검토 보고서

작성일: 2026-06-10 (웹 조사 기준 시점: 2026년 6월)
대상: meetily macOS Tauri 앱 — Rust 백엔드 트랜스크립션 엔진 추가

---

## 1. 요약 (추천 경로)

**추천: sherpa-onnx 공식 Rust 크레이트(`sherpa-onnx`, v1.13.2+) 경로로 Qwen3-ASR-0.6B int8 모델을 통합한다.**

근거:

1. **한국어 정확도** — Qwen3-ASR은 한국어(ko)를 공식 지원하는 30개 언어(+22개 중국어 방언, 총 52개) 모델이며, 기술 보고서 기준 Fleurs-ko CER 3.72(0.6B) / 2.57(1.7B), Common Voice-ko CER 8.48(0.6B) / 5.88(1.7B)로 보고됨 ([arXiv 2601.21337](https://arxiv.org/html/2601.21337v1)). Whisper large-v3과의 한국어 직접 비교 수치는 공개 자료에서 확인하지 못했으나(아래 "미검증" 참조), 영어·중국어·다국어 벤치마크(MLS, Common Voice, MLC-SLM)에서는 Whisper-large-v3를 일관되게 상회한다고 보고됨.
2. **컨텍스트 바이어싱 공식 지원** — Qwen3-ASR은 시스템 프롬프트에 임의 텍스트를 넣어 용어를 바이어싱하는 기능을 공식 지원하며 ([Qwen3-ASR 소개](https://qwen-ai.com/qwen-asr/), [기술 보고서](https://arxiv.org/html/2601.21337v1)), sherpa-onnx는 v1.12.35(2026-04)에서 Qwen3-ASR hotwords 지원, v1.12.36에서 `OfflineQwen3ASRModelConfig`의 hotwords 인자와 스트림별 언어 힌트를 추가함 ([sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases)). 이는 meetily의 "회의 참석자 이름·사내 용어 바이어싱" 목표에 직결됨.
3. **GGUF/llama.cpp 경로는 아직 미성숙** — ggml-org의 GGUF 변환이 존재하고 llama.cpp 멀티모달로 구동되지만, 오디오 입력은 공식적으로 "experimental" 단계이며 약 2분 이상 오디오에서 출력이 비는 미해결 버그가 보고됨 ([llama.cpp issue #21847](https://github.com/ggml-org/llama.cpp/issues/21847)). 실시간 회의 트랜스크립션 용도로는 리스크가 큼.
4. **통합 비용** — meetily의 Parakeet은 `ort`(ONNX Runtime) 크레이트를 직접 사용하므로(`frontend/src-tauri/Cargo.toml`의 `ort = "2.0.0-rc.10"`), `ort`로 직접 Qwen3 ONNX를 돌리는 것도 이론상 가능하나 인코더/디코더 오케스트레이션·토크나이저·프롬프트 처리를 전부 자체 구현해야 함. sherpa-onnx 크레이트는 이를 C API 위에 RAII로 감싼 완성품을 제공하고 macOS arm64 프리빌트 정적 링크를 지원함 ([docs.rs/sherpa-onnx](https://docs.rs/sherpa-onnx)).

차선책: 1.7B 모델은 정확도가 더 높지만(특히 Common Voice-ko 5.88 vs 8.48) 모델 크기·메모리가 커지므로, 0.6B int8로 시작해 품질 피드백 후 1.7B 추가를 검토.

---

## 2. 배경: 현재 meetily 구조

- 트랜스크립션 엔진 추상화: `frontend/src-tauri/src/audio/transcription/provider.rs`의 `TranscriptionProvider` 트레이트 (async `transcribe(audio: Vec<f32>, language: Option<String>) -> TranscriptResult`, `is_model_loaded`, `get_current_model`, `provider_name`).
- 엔진 선택: `engine.rs`의 `TranscriptionEngine` enum (`Whisper` / `Parakeet` / `Provider(Arc<dyn TranscriptionProvider>)`) + `validate_transcription_model_ready` / `get_or_init_transcription_engine`이 `config.provider` 문자열("localWhisper" / "parakeet")로 분기.
- Parakeet 구현: `parakeet_provider.rs`가 `crate::parakeet_engine::ParakeetEngine`(내부적으로 `ort` ONNX Runtime 사용)을 감싸는 형태. **sherpa는 현재 사용하지 않음** — 따라서 sherpa-onnx 도입은 새 네이티브 의존성 추가를 의미함(아래 리스크 참조). 단, ONNX Runtime 자체는 이미 번들되어 있어 ORT 생태계 자체는 낯설지 않음.

---

## 3. Qwen3-ASR 모델 개요 (조사 결과)

- **출시**: 2026년 1월경, Qwen3-ASR-0.6B / 1.7B, Apache 2.0 ([HF: Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B), [arXiv 2601.21337](https://arxiv.org/abs/2601.21337)).
- **아키텍처**: Qwen3 LLM 백본 + AuT 오디오 인코더(0.6B: 180M 인코더/hidden 896, 1.7B: 300M 인코더/hidden 1024) + 프로젝터. 8배 다운샘플링, 12.5Hz 토큰 레이트. 참고로 0.6B의 실제 파라미터는 약 0.9B(HF 표기).
- **언어**: 30개 언어 + 22개 중국어 방언 = 52개. 한국어(ko) 포함.
- **스트리밍**: 1~8초 동적 어텐션 윈도우로 단일 가중치가 오프라인/스트리밍 겸용. 스트리밍 시 LibriSpeech-other WER 4.51 (오프라인 3.38, 1.7B 기준).
- **정확도(공개 수치)**:
  - 한국어: Fleurs-ko CER 3.72(0.6B) / 2.57(1.7B), Common Voice-ko CER 8.48(0.6B) / 5.88(1.7B) ([기술 보고서](https://arxiv.org/html/2601.21337v1)).
  - 영어 Tedlium WER 4.50 vs Whisper-large-v3 계열 6.84~7.69; 중국어에서 큰 격차로 우세. MLS·Common Voice·MLC-SLM에서 1.7B가 Whisper-large-v3 포함 오픈소스 베이스라인을 일관 상회.
  - **⚠️ 미검증**: Whisper large-v3의 "한국어" 수치와의 1:1 비교표는 기술 보고서·모델 카드에서 확인하지 못함. "한국어에서 Whisper large-v3보다 낫다"는 주장은 다국어 벤치마크 추세에서의 추정이며, 도입 전 자체 한국어 회의 오디오로 A/B 평가 필요.
- **컨텍스트 바이어싱**: 시스템 프롬프트의 컨텍스트 토큰을 배경지식으로 활용하도록 학습됨(보고서 명시). 공식 API에서는 최대 10,000 토큰의 임의 텍스트 바이어싱을 지원 ([qwen-ai.com/qwen-asr](https://qwen-ai.com/qwen-asr/)). 단, 바이어싱 적용/미적용 WER 정량 비교는 보고서에 없음.
- **기타 추론 옵션**: transformers/vLLM(공식), MLX 포트([mlx-qwen3-asr](https://github.com/moona3k/mlx-qwen3-asr/)), C 단일파일 구현([antirez/qwen-asr](https://github.com/antirez/qwen-asr)) 등 생태계가 빠르게 형성 중 — Rust 통합 관점에서는 sherpa-onnx와 GGUF가 현실적 후보.

---

## 4. 옵션 비교: sherpa-onnx vs GGUF(llama.cpp)

| 항목 | sherpa-onnx 경로 | GGUF / llama.cpp 경로 |
|---|---|---|
| **성숙도** | v1.12.34(2026-03-26)에 Qwen3-ASR 도입, v1.12.35~36에서 hotwords·언어 힌트·fp32 초기화 수정, 최신 v1.13.2(2026-05)까지 유지보수 활발 ([releases](https://github.com/k2-fsa/sherpa-onnx/releases)). 12개 언어 바인딩 제공 | 오디오 입력이 공식 문서상 "experimental" ([multimodal.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)). 약 2분 초과 오디오에서 빈 출력 버그 미해결 ([#21847](https://github.com/ggml-org/llama.cpp/issues/21847)) |
| **한국어 정확도** | 동일 모델 가중치 → 원모델 수준 기대. int8 양자화에 의한 한국어 CER 열화 정도는 **미검증** | 동일 모델 가중치. GGUF 양자화 열화 역시 **미검증** |
| **컨텍스트 바이어싱** | **공식 지원**: `OfflineQwen3ASRModelConfig`에 hotwords 인자 (v1.12.36+), 스트림별 언어 힌트 | 명시적 바이어싱 API 없음. 프롬프트 주입으로 흉내낼 수는 있으나 llama-server/mtmd 경로에서의 동작은 **미검증** |
| **통합 난이도** | 낮음~중간: 공식 Rust 크레이트(`sherpa-onnx` v1.13.2)가 `OfflineQwen3ASRModelConfig`/`OfflineRecognizer` 노출, macOS arm64 프리빌트 자동 다운로드 + 정적 링크 ([docs.rs](https://docs.rs/sherpa-onnx)). 단 빌드 시 네트워크 의존(아래 리스크) | 중간~높음: whisper-rs처럼 잘 정리된 Rust 바인딩이 멀티모달 오디오(mtmd) 경로에는 없음. llama-server 사이드카(HTTP) 또는 FFI 자체 바인딩 필요 |
| **바이너리/모델 크기** | 네이티브 라이브러리 추가(정적 링크, 자체 onnxruntime 포함 — 기존 `ort`와 ORT 중복 가능). 모델: 0.6B int8 약 1GB 수준(서드파티 변환 [pantinor 레포](https://huggingface.co/pantinor/sherpa-onnx-qwen3-asr-0.6b-int8) 기준 encoder 182MB + decoder 757MB + frontend 44MB ≈ 988MB; **공식 k2-fsa 아카이브의 정확한 크기는 미확인**) | llama.cpp 라이브러리(이미 whisper.cpp/ggml을 whisper-rs로 쓰고 있어 ggml 계열 중복). 모델: [ggml-org/Qwen3-ASR-0.6B-GGUF](https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF) + mmproj (정확한 크기 미확인, BF16 기준 수백 MB~1GB대 추정) |
| **메모리/속도** | 0.6B int8: Intel N100(8W)에서 VAD 청킹 시 RTF 0.71, 디코더 토큰당 ~100ms 보고. **청킹 없이 10분 오디오 처리 시 15GB+ 메모리 보고** → 30초 내외 청킹 필수 ([NVIDIA 포럼](https://forums.developer.nvidia.com/t/how-to-use-qwen3-asr-0-6b-on-jetson-orin-nano/361835), [Daumee ONNX-CPU](https://huggingface.co/Daumee/Qwen3-ASR-0.6B-ONNX-CPU)). Apple Silicon 수치는 **미확인**(CPU 추론, Metal 가속 아님) | Metal 가속 가능(llama.cpp 강점)이나 장문 오디오 버그로 실효성 낮음. 스트리밍/청크 디코딩 미성숙 |
| **라이선스** | sherpa-onnx: Apache 2.0. 모델: Apache 2.0 | llama.cpp: MIT. 모델: Apache 2.0 |
| **macOS arm64** | 공식 지원(프리빌트 정적/동적 아카이브) | 지원(llama.cpp 자체는 macOS 일급 지원) |

**판정**: 컨텍스트 바이어싱이 핵심 요구사항인 이상, 현재(2026-06) 시점에서 실질적 선택지는 sherpa-onnx뿐. GGUF 경로는 llama.cpp 오디오 지원이 안정화되고 장문 버그가 해결된 뒤 재평가.

참고: 커뮤니티 Rust 바인딩 [sherpa-rs](https://github.com/thewh1teagle/sherpa-rs)는 **2026-06-06 아카이브(개발 중단)**되었고 공식 `sherpa-onnx` 크레이트 사용을 권고하므로, 반드시 공식 크레이트([crates.io/crates/sherpa-onnx](https://crates.io/crates/sherpa-onnx))를 사용할 것.

---

## 5. TranscriptionProvider 구현 스케치 (의사코드)

`parakeet_provider.rs` 패턴을 따르되, sherpa-onnx `OfflineRecognizer`를 감싸는 형태. sherpa-onnx의 recognizer/stream은 `Send + Sync`가 보장되지 않을 수 있으므로 Parakeet/Whisper 엔진처럼 내부 엔진 구조체 + `tokio::sync::Mutex`(또는 전용 blocking 스레드)로 감싼다.

```rust
// audio/transcription/qwen3_provider.rs (의사코드)

use super::provider::{TranscriptionError, TranscriptionProvider, TranscriptResult};
use async_trait::async_trait;
use std::sync::Arc;

pub struct Qwen3AsrEngine {
    // sherpa_onnx::OfflineRecognizer는 !Send 가능성 → Mutex + spawn_blocking으로 격리
    recognizer: tokio::sync::Mutex<Option<sherpa_onnx::OfflineRecognizer>>,
    current_model: tokio::sync::RwLock<Option<String>>,
    hotwords: tokio::sync::RwLock<Option<String>>, // 회의별 컨텍스트 바이어싱 텍스트
}

impl Qwen3AsrEngine {
    pub async fn load_model(&self, model_dir: &Path, hotwords: Option<&str>) -> Result<()> {
        // sherpa-onnx-qwen3-asr-0.6B-int8 아카이브 구성 파일 경로 지정
        let qwen3_cfg = sherpa_onnx::OfflineQwen3ASRModelConfig {
            encoder: model_dir.join("encoder.int8.onnx"),
            decoder: model_dir.join("decoder.int8.onnx"),
            // (+ conv frontend / tokens 등 아카이브 동봉 파일 — 실제 필드명은
            //  docs.rs/sherpa-onnx v1.13.2 기준으로 확정할 것)
            hotwords: hotwords.map(String::from), // v1.12.36+ 컨텍스트 바이어싱
            ..Default::default()
        };
        let config = sherpa_onnx::OfflineRecognizerConfig {
            model_config: sherpa_onnx::OfflineModelConfig {
                qwen3_asr: Some(qwen3_cfg),
                num_threads: 4,
                ..Default::default()
            },
            ..Default::default()
        };
        let rec = tokio::task::spawn_blocking(move || {
            sherpa_onnx::OfflineRecognizer::create(&config)
        }).await??;
        *self.recognizer.lock().await = Some(rec);
        Ok(())
    }

    pub async fn transcribe_audio(
        &self, audio: Vec<f32>, language: Option<String>,
    ) -> Result<String> {
        // 메모리 폭주 방지: 30초 단위 청킹 (10분 무청킹 시 15GB+ 보고됨)
        let guard = self.recognizer.lock().await;
        let rec = guard.as_ref().ok_or(ModelNotLoaded)?;
        let mut out = String::new();
        for chunk in audio.chunks(16_000 * 30) {
            let stream = rec.create_stream();
            // v1.12.36+: 스트림별 언어 힌트 (ko 강제 → 언어 오인식 방지)
            if let Some(lang) = &language { stream.set_language_hint(lang); } // 실제 API명 확인 필요
            stream.accept_waveform(16_000, chunk);
            rec.decode(&stream); // CPU-bound → spawn_blocking 권장
            out.push_str(&stream.get_result().text);
        }
        Ok(out)
    }
}

pub struct Qwen3AsrProvider {
    engine: Arc<Qwen3AsrEngine>,
}

#[async_trait]
impl TranscriptionProvider for Qwen3AsrProvider {
    async fn transcribe(
        &self, audio: Vec<f32>, language: Option<String>,
    ) -> Result<TranscriptResult, TranscriptionError> {
        // provider.rs 계약: 16kHz mono f32 입력 → 그대로 전달 가능
        match self.engine.transcribe_audio(audio, language).await {
            Ok(text) => Ok(TranscriptResult {
                text: text.trim().to_string(),
                confidence: None,   // sherpa-onnx Qwen3 결과의 신뢰도 노출 여부 미확인
                is_partial: false,  // 오프라인 recognizer → 부분 결과 없음
            }),
            Err(e) => Err(TranscriptionError::EngineFailed(e.to_string())),
        }
    }

    async fn is_model_loaded(&self) -> bool { self.engine.is_loaded().await }
    async fn get_current_model(&self) -> Option<String> { self.engine.current_model().await }
    fn provider_name(&self) -> &'static str { "Qwen3-ASR" }
}
```

배선(wiring) 변경점:

1. `engine.rs`: `validate_transcription_model_ready`와 `get_or_init_transcription_engine`의 `match config.provider.as_str()`에 `"qwen3Asr"` 분기 추가. 새 코드는 enum 변형 추가 대신 기존 `TranscriptionEngine::Provider(Arc<dyn TranscriptionProvider>)` 경로를 사용(주석상 "preferred for new code").
2. 모델 다운로드: Whisper/Parakeet과 동일한 모델 매니저 패턴으로 sherpa-onnx 공식 릴리스 아카이브(`sherpa-onnx-qwen3-asr-0.6B-int8-…`, [GitHub releases](https://github.com/k2-fsa/sherpa-onnx/releases))를 받도록 추가.
3. 컨텍스트 바이어싱 UI: 회의 시작 전 참석자명/용어 입력 → hotwords 문자열로 전달. hotwords 변경 시 recognizer 재생성 필요 여부는 API 확인 필요(설정이 config 단위라면 재생성, 스트림 단위라면 즉시 적용).

---

## 6. 리스크

1. **한국어 우위 미검증**: Whisper large-v3 대비 "한국어" 직접 비교 수치를 찾지 못함. Fleurs/CommonVoice CER은 양호하나 회의체 한국어(겹침 발화, 코드스위칭)는 별개 → 도입 결정 전 사내 한국어 회의 샘플로 whisper large-v3 vs Qwen3-0.6B/1.7B A/B 필수.
2. **메모리 폭주**: 무청킹 장문 처리 시 15GB+ 보고. meetily는 실시간 세그먼트 단위 처리라 완화되지만, retranscription(파일 재전사) 경로에서는 반드시 청킹 적용.
3. **속도(Apple Silicon CPU 추론)**: sherpa-onnx는 macOS에서 기본 CPU(onnxruntime) 추론. 디코더 토큰당 ~100ms(int8, 저전력 x86 기준) 보고 — M시리즈에서는 더 빠를 것으로 예상되나 **실측 필요**. 실시간(RTF<1) 미달 시 0.6B int8로도 부족할 수 있음. CoreML EP 활용 가능성은 미확인.
4. **ONNX Runtime 중복**: 기존 `ort` 크레이트(Parakeet)와 sherpa-onnx 내장 onnxruntime이 이중 번들될 수 있음 → 바이너리 크기 증가 및 심볼 충돌 가능성(정적 링크라 충돌 가능성은 낮으나 검증 필요).
5. **빌드 파이프라인**: 공식 크레이트는 빌드 시 프리빌트 아카이브를 GitHub에서 자동 다운로드(오프라인/CI 캐싱 고려, `SHERPA_ONNX_LIB_DIR`로 고정 가능).
6. **신뢰도/부분 결과 부재**: confidence·partial 미지원 가능성 → 기존 UI가 이를 옵션으로 다루므로 영향 적음.
7. **hotwords 동작 품질 미검증**: sherpa-onnx의 hotwords가 Qwen3의 "시스템 프롬프트 컨텍스트" 방식과 동등한 효과를 내는지, 한국어 고유명사에 효과적인지 정량 자료 없음.
8. **모델 크기 ~1GB**: 기존 whisper small(수백 MB) 대비 다운로드·디스크 부담 증가. 공식 int8 아카이브 정확한 크기는 다운로드 후 확인 필요.

---

## 7. 권장 다음 단계

1. **스파이크(1~2일)**: 별도 Rust 바이너리로 `sherpa-onnx` 크레이트 + `sherpa-onnx-qwen3-asr-0.6B-int8` 아카이브를 받아 M시리즈 맥에서 (a) 한국어 회의 샘플 CER, (b) RTF, (c) 피크 메모리, (d) hotwords 효과(참석자명 10개 주입 전/후)를 측정.
2. **whisper large-v3 / Parakeet과 A/B**: 동일 한국어 오디오 3종(깨끗한 발화, 회의 잡음, 전문용어 다수)으로 비교. 여기서 우위가 확인될 때만 본 통합 진행.
3. **본 통합**: `qwen3_provider.rs` + `engine.rs` 분기 + 모델 다운로드 매니저 + 설정 UI(provider 선택, 컨텍스트 바이어싱 텍스트 입력). 새 코드는 `TranscriptionEngine::Provider` 트레이트 경로 사용.
4. **청킹·재전사 경로 점검**: 30초 청킹을 retranscription/import 경로에 강제.
5. **후속 검토**: (a) 1.7B int8 추가(정확도 우선 옵션), (b) llama.cpp GGUF 경로는 장문 오디오 버그(#21847) 해결 및 오디오 지원 안정화 후 Metal 가속 목적으로 재평가, (c) MLX 포트는 Tauri/Rust 통합이 어려워 보류.

---

## 부록: 주요 출처

- Qwen3-ASR 기술 보고서: https://arxiv.org/html/2601.21337v1 (한국어 CER, 아키텍처, 스트리밍, 컨텍스트 바이어싱)
- 모델 카드: https://huggingface.co/Qwen/Qwen3-ASR-0.6B (Apache 2.0, 언어 목록, 벤치마크)
- Qwen3-ASR 기능 개요: https://qwen-ai.com/qwen-asr/ (52개 언어, 컨텍스트 바이어싱 1만 토큰)
- sherpa-onnx 릴리스: https://github.com/k2-fsa/sherpa-onnx/releases (v1.12.34 Qwen3 도입, v1.12.35 hotwords, v1.12.36 언어 힌트, 최신 v1.13.2)
- sherpa-onnx 공식 Rust 크레이트: https://docs.rs/sherpa-onnx , https://crates.io/crates/sherpa-onnx
- sherpa-onnx Qwen3-ASR 문서: https://k2-fsa.github.io/sherpa/onnx/qwen3-asr/index.html
- sherpa-rs(아카이브됨, 사용 금지): https://github.com/thewh1teagle/sherpa-rs
- GGUF 변환: https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF , https://huggingface.co/ggml-org/Qwen3-ASR-1.7B-GGUF
- llama.cpp 멀티모달 문서(오디오 experimental): https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md
- llama.cpp 장문 오디오 버그: https://github.com/ggml-org/llama.cpp/issues/21847
- 성능/메모리 보고(서드파티): https://huggingface.co/Daumee/Qwen3-ASR-0.6B-ONNX-CPU , https://forums.developer.nvidia.com/t/how-to-use-qwen3-asr-0-6b-on-jetson-orin-nano/361835
- int8 ONNX 파일 크기 참고(서드파티 변환): https://huggingface.co/pantinor/sherpa-onnx-qwen3-asr-0.6b-int8 (~988MB)

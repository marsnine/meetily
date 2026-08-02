import Foundation
import LiteRTLM

/// Writes a line to stderr so `devicectl device process launch --console` can
/// capture results for unattended verification.
func edgeLog(_ s: String) {
    FileHandle.standardError.write(Data(("[EDGE] " + s + "\n").utf8))
}

/// Current thermal pressure — a proxy for heat. nominal < fair < serious < critical.
func thermalString() -> String {
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: return "nominal"
    case .fair: return "fair"
    case .serious: return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
}

/// On-device Gemma 4 E2B ASR + meeting-minutes summary via the OFFICIAL
/// google-ai-edge/LiteRT-LM Swift API (v0.13.1). `backend` selects CPU vs GPU
/// (Metal) for the LLM; the audio encoder runs on CPU (model constraint).
@MainActor
final class Transcriber: ObservableObject {
    @Published var status: String = "대기 중 — 클립을 탭하면 모델을 로드하고 전사합니다."
    @Published var transcript: String = ""
    @Published var summary: String = ""
    @Published var lastTranscript: String = ""
    @Published var isBusy: Bool = false
    // LLM backend: GPU (Metal) is the default — same as Google's Eloquent (LLM on
    // GPU, audio encoder on CPU per model constraint). Much faster + cooler than CPU.
    @Published var backend: String = "gpu"   // "gpu" or "cpu"

    let chunkSeconds: Double = 30
    private var engine: Engine?
    private var loadedBackend: String?

    // MARK: - Model file (copied to a writable dir for XNNPack caches)

    /// EDGE_MODEL=gemma3n selects Gemma 3n E2B (audio encoder can run on GPU);
    /// default is Gemma 4 E2B (audio encoder is CPU-only).
    private var modelResource: String {
        ProcessInfo.processInfo.environment["EDGE_MODEL"] == "gemma3n"
            ? "gemma-3n-E2B-it-int4" : "gemma-4-E2B-it"
    }

    private func writableModelURL() -> URL? {
        let res = modelResource
        guard let bundled = Bundle.main.url(forResource: res, withExtension: "litertlm") else {
            return nil
        }
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dest = docs.appendingPathComponent("\(res).litertlm")
        if FileManager.default.fileExists(atPath: dest.path) { return dest }
        status = "모델 준비 중… (최초 1회 복사, ~2.6GB)"
        do { try FileManager.default.copyItem(at: bundled, to: dest); return dest }
        catch { status = "❌ 모델 복사 실패: \(error.localizedDescription)"; return bundled }
    }

    private func backendValue(_ s: String) -> Backend { s == "gpu" ? .gpu : .cpu() }

    // MARK: - Engine lifecycle

    private func ensureEngine() async -> Engine? {
        if let engine, loadedBackend == backend { return engine }
        engine = nil
        guard let modelURL = writableModelURL() else {
            status = "❌ 모델 파일(gemma-4-E2B-it.litertlm)을 못 찾음"; return nil
        }
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("litertlm_cache")
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)

        status = "\(backend.uppercased()) · 모델 로딩 중…"
        edgeLog("MODEL_LOADING backend=\(backend) thermal=\(thermalString())")
        let t0 = Date()
        do {
            ExperimentalFlags.optIntoExperimentalAPIs()
            ExperimentalFlags.enableBenchmark = true
            // Audio encoder backend — default CPU (Gemma 4 E2B constraint), but
            // EDGE_AUDIO_BACKEND=gpu lets us test whether full-GPU audio works.
            let audioBE: Backend = ProcessInfo.processInfo.environment["EDGE_AUDIO_BACKEND"] == "gpu" ? .gpu : .cpu()
            let config = try EngineConfig(
                modelPath: modelURL.path,
                backend: backendValue(backend),
                visionBackend: nil,           // skip vision encoder
                audioBackend: audioBE,
                maxNumTokens: 4096,
                cacheDir: cacheDir.path
            )
            let e = Engine(engineConfig: config)
            try await e.initialize()
            engine = e
            loadedBackend = backend
            let dt = Date().timeIntervalSince(t0)
            status = String(format: "%@ · 모델 로드 %.1fs", backend.uppercased(), dt)
            edgeLog(String(format: "MODEL_LOADED backend=%@ load=%.1fs thermal=%@", backend, dt, thermalString()))
            return e
        } catch {
            status = "❌ 모델 로드 실패: \(error.localizedDescription)"
            edgeLog("MODEL_LOAD_FAILED backend=\(backend) \(error.localizedDescription)")
            return nil
        }
    }

    /// Transcribe one audio chunk in a fresh conversation (independent context).
    private func transcribeChunk(_ engine: Engine, wavData: Data) async throws -> String {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("chunk_\(UUID().uuidString).wav")
        try wavData.write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }
        // Explicit sampler: the default is too greedy and degenerates into
        // repetition loops ("네 네 네…") on conversational/filler-heavy Korean
        // audio. Nucleus sampling (topP/topK) + temperature breaks the loops.
        let sampler = try SamplerConfig(topK: 40, topP: 0.95, temperature: 0.7, seed: 0)
        let conv = try await engine.createConversation(with: ConversationConfig(samplerConfig: sampler))
        let msg = Message(contents: [.audioFile(tmp.path), .text("Transcribe this audio verbatim.")])
        let resp = try await conv.sendMessage(msg)
        return Self.stripRunawayRepetition(resp.toString.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    /// Gemma's audio decoder sometimes fails to emit end-of-speech and loops on a
    /// filler token ("어, 어, 어 …" ×1000s) until the token budget. Cut the output
    /// at the start of any token repeated 4+ times in a row, keeping the good prefix.
    static func stripRunawayRepetition(_ text: String) -> String {
        let words = text.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
        var out: [String] = []
        var prev = ""
        var run = 0
        for w in words {
            let norm = w.trimmingCharacters(in: CharacterSet(charactersIn: ",.?!…\"'·"))
            if norm == prev && !norm.isEmpty {
                run += 1
                if run >= 3 { break }   // 4th consecutive repeat → stop (drop the loop)
            } else {
                prev = norm; run = 0
            }
            out.append(w)
        }
        return out.joined(separator: " ")
    }

    /// Text generation in a fresh conversation (Gemma prompt template applied internally).
    private func generateText(_ engine: Engine, prompt: String) async throws -> String {
        let sampler = try SamplerConfig(topK: 64, topP: 0.95, temperature: 0.3, seed: 0)
        let conv = try await engine.createConversation(with: ConversationConfig(samplerConfig: sampler))
        let resp = try await conv.sendMessage(Message(prompt))
        return resp.toString.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Unattended entrypoint

    func preload() async { _ = await ensureEngine() }

    func autoRunIfRequested() async {
        if let be = ProcessInfo.processInfo.environment["EDGE_BACKEND"], !be.isEmpty { backend = be }
        guard let clip = ProcessInfo.processInfo.environment["EDGE_AUTORUN"], !clip.isEmpty else {
            await preload(); return
        }
        edgeLog("AUTORUN_START clip=\(clip) backend=\(backend)")
        await transcribe(resource: clip, title: clip)
        if ProcessInfo.processInfo.environment["EDGE_SUMMARIZE"] == "1" { await summarize() }
        edgeLog("AUTORUN_COMPLETE clip=\(clip)")
    }

    // MARK: - Chunked transcription

    func transcribe(resource: String, title: String) async {
        guard !isBusy else { return }
        isBusy = true
        defer { isBusy = false }

        guard let engine = await ensureEngine() else { return }
        guard let url = Bundle.main.url(forResource: resource, withExtension: "wav"),
              let wav = WavAudio.read(url) else {
            status = "❌ 오디오 없음/파싱 실패: \(resource).wav"; edgeLog("AUDIO_READ_FAILED \(resource)"); return
        }

        let total = wav.durationSeconds
        var nChunks = max(1, Int(ceil(total / chunkSeconds)))
        if let cap = ProcessInfo.processInfo.environment["EDGE_MAX_CHUNKS"], let n = Int(cap), n > 0 {
            nChunks = min(nChunks, n)
        }

        transcript = ""
        var full = ""
        let t0 = Date()
        edgeLog(String(format: "LONG_START clip=%@ dur=%.0fs chunks=%d backend=%@ thermal=%@",
                        resource, total, nChunks, backend, thermalString()))

        for idx in 0..<nChunks {
            let s = Double(idx) * chunkSeconds
            let e = min(total, s + chunkSeconds)
            let chunkWav = wav.wavData(from: s, to: e)
            status = "\(backend.uppercased()) · 전사 중 \(idx + 1)/\(nChunks)…"
            let c0 = Date()
            do {
                let piece = try await transcribeChunk(engine, wavData: chunkWav)
                if !piece.isEmpty { full += (full.isEmpty ? "" : " ") + piece }
                transcript = full
                let cdt = Date().timeIntervalSince(c0)
                edgeLog(String(format: "CHUNK %d/%d %.0f-%.0fs infer=%.1fs thermal=%@: %@",
                                idx + 1, nChunks, s, e, cdt, thermalString(), String(piece.prefix(70))))
            } catch {
                edgeLog("CHUNK_FAIL \(idx + 1)/\(nChunks) \(error.localizedDescription)")
            }
        }

        let dt = Date().timeIntervalSince(t0)
        transcript = full
        lastTranscript = full
        let rt = total > 0 ? dt / total : 0
        status = String(format: "전사 완료 · %d청크 · %.0fs (%.1fx 실시간)", nChunks, dt, rt)
        edgeLog(String(format: "LONG_DONE clip=%@ total=%.0fs realtime=%.1fx chars=%d thermal=%@",
                        resource, dt, rt, full.count, thermalString()))
        edgeLog("TRANSCRIPT_BEGIN[\(resource)]"); edgeLog(full); edgeLog("TRANSCRIPT_END[\(resource)]")
    }

    // MARK: - Meeting-minutes summary (map-reduce)

    private func isKorean(_ s: String) -> Bool {
        let hangul = s.unicodeScalars.filter { $0.value >= 0xAC00 && $0.value <= 0xD7A3 }.count
        return hangul > s.count / 20
    }

    private func splitText(_ s: String, maxChars: Int) -> [String] {
        var pieces: [String] = []; var cur = ""
        for w in s.split(separator: " ", omittingEmptySubsequences: true) {
            if cur.count + w.count + 1 > maxChars && !cur.isEmpty { pieces.append(cur); cur = "" }
            cur += (cur.isEmpty ? "" : " ") + w
        }
        if !cur.isEmpty { pieces.append(cur) }
        return pieces.isEmpty ? [s] : pieces
    }

    func summarize() async {
        guard !isBusy else { return }
        let text = lastTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { status = "요약할 전사가 없습니다"; return }
        isBusy = true
        defer { isBusy = false }
        guard let engine = await ensureEngine() else { return }

        let ko = isKorean(text)
        summary = ""
        let t0 = Date()
        edgeLog("SUMMARY_START chars=\(text.count) ko=\(ko)")

        var notes = text
        let pieces = splitText(text, maxChars: 2400)
        if pieces.count > 1 {
            var partials: [String] = []
            for (i, p) in pieces.enumerated() {
                status = "요약 1단계 \(i + 1)/\(pieces.count)…"
                let inst = ko
                    ? "다음 회의 전사 일부의 핵심을 간결한 불릿으로 정리하세요. 전사에 있는 내용만 쓰고 숫자·고유명사는 보존:\n\n\(p)"
                    : "Summarize the key points of this meeting transcript segment as concise bullets. Use only what's stated; keep numbers and names:\n\n\(p)"
                if let r = try? await generateText(engine, prompt: inst) {
                    partials.append(r); edgeLog("SUMMARY_MAP \(i + 1)/\(pieces.count) -> \(r.count) chars")
                }
            }
            notes = partials.joined(separator: "\n")
        }

        status = "회의록 작성 중…"
        let finalInst = ko ? """
        다음 회의 내용을 바탕으로 한국어 회의록을 작성하세요. 형식:
        ## 회의 개요
        ## 핵심 논의 사항
        ## 결정 사항
        ## 후속 조치(액션 아이템)
        내용에 없는 것을 지어내지 말고 숫자·고유명사는 보존하세요.

        회의 내용:
        \(notes)
        """ : """
        Write structured meeting minutes in English from the content below. Format:
        ## Overview
        ## Key Discussion Points
        ## Decisions
        ## Action Items
        Do not invent anything; preserve numbers and names.

        Content:
        \(notes)
        """
        do {
            let result = try await generateText(engine, prompt: finalInst)
            summary = result
            let dt = Date().timeIntervalSince(t0)
            status = String(format: "회의록 완료 · %.0fs", dt)
            edgeLog(String(format: "SUMMARY_DONE %.0fs chars=%d", dt, summary.count))
            edgeLog("SUMMARY_BEGIN"); edgeLog(summary); edgeLog("SUMMARY_END")
        } catch {
            status = "❌ 요약 실패: \(error.localizedDescription)"
            edgeLog("SUMMARY_FAILED \(error.localizedDescription)")
        }
    }
}

import SwiftUI

private struct Clip: Identifiable {
    let id: String       // resource name (without .wav)
    let title: String
    let sub: String
}

struct ContentView: View {
    @StateObject private var t = Transcriber()

    private let clips: [Clip] = [
        Clip(id: "en_sample", title: "English (short)", sub: "영어 · ~7초"),
        Clip(id: "ko_sample", title: "한국어 (short)", sub: "한국어 · ~13초"),
        Clip(id: "en_long",   title: "English meeting", sub: "영어 회의 · ~68초 · 청크"),
        Clip(id: "en_meeting", title: "English meeting (full)", sub: "영어 회의 · ~93초 · 청크"),
        Clip(id: "ko_long",   title: "한국어 회의 (실제 통화)", sub: "한국어 · ~16분 · 청크"),
        Clip(id: "ko_meeting38", title: "한국어 IR미팅 (실제)", sub: "한국어 · ~38분 · 청크")
    ]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 14) {
                    header

                    // LLM backend (GPU = Metal, like Eloquent; audio encoder is CPU either way)
                    Picker("백엔드", selection: $t.backend) {
                        Text("GPU (Metal)").tag("gpu")
                        Text("CPU").tag("cpu")
                    }
                    .pickerStyle(.segmented)
                    .disabled(t.isBusy)

                    // Clip picker
                    VStack(spacing: 8) {
                        ForEach(clips) { c in
                            Button {
                                Task { await t.transcribe(resource: c.id, title: c.title) }
                            } label: { clipRow(c) }
                            .buttonStyle(.plain)
                            .disabled(t.isBusy)
                        }
                    }

                    // Status
                    HStack(spacing: 8) {
                        if t.isBusy { ProgressView().scaleEffect(0.8) }
                        Text(t.status)
                            .font(.footnote)
                            .foregroundStyle(t.status.hasPrefix("❌") ? .red : .secondary)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .frame(minHeight: 22)

                    // Transcript
                    if !t.transcript.isEmpty {
                        section(title: "📝 전사", body: t.transcript)

                        Button {
                            Task { await t.summarize() }
                        } label: {
                            Label("회의록 생성", systemImage: "doc.text.magnifyingglass")
                                .font(.headline)
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 10)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(t.isBusy)
                    }

                    // Summary (rendered as markdown when possible)
                    if !t.summary.isEmpty {
                        summarySection(t.summary)
                    }
                }
                .padding()
            }
            .navigationTitle("Meetily Edge")
            .navigationBarTitleDisplayMode(.inline)
            .task { await t.autoRunIfRequested() }
        }
    }

    private var header: some View {
        VStack(spacing: 2) {
            Text("Gemma 4 E2B · LiteRT-LM · 온디바이스")
                .font(.caption).foregroundStyle(.secondary)
            Text("오디오 전사 + 회의록 요약 · 오프라인 · CPU")
                .font(.caption2).foregroundStyle(.tertiary)
        }
    }

    private func clipRow(_ c: Clip) -> some View {
        HStack {
            Image(systemName: "waveform.circle.fill").font(.title2).foregroundStyle(.blue)
            VStack(alignment: .leading, spacing: 2) {
                Text(c.title).font(.subheadline).fontWeight(.semibold)
                Text(c.sub).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            Image(systemName: "chevron.right").font(.caption).foregroundStyle(.tertiary)
        }
        .padding(12)
        .background(Color.gray.opacity(0.10))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private func section(title: String, body: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.subheadline).fontWeight(.bold)
            Text(body)
                .font(.callout)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(12)
                .background(Color.gray.opacity(0.06))
                .clipShape(RoundedRectangle(cornerRadius: 12))
        }
    }

    private func summarySection(_ md: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("📋 회의록").font(.subheadline).fontWeight(.bold)
            Group {
                if let attr = try? AttributedString(
                    markdown: md,
                    options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)
                ) {
                    Text(attr)
                } else {
                    Text(md)
                }
            }
            .font(.callout)
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(12)
            .background(Color.blue.opacity(0.06))
            .clipShape(RoundedRectangle(cornerRadius: 12))
        }
    }
}

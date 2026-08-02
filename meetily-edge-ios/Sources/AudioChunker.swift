import Foundation

/// Minimal WAV reader/slicer for 16 kHz mono PCM16 clips. Gemma's audio encoder
/// works on short clips (the LLM context can't hold 30–50 min at ~6 audio
/// tokens/sec), so long meetings are split into fixed-length segments, each
/// transcribed independently and concatenated.
struct WavAudio {
    let sampleRate: Int
    let channels: Int
    let bitsPerSample: Int
    let pcm: Data   // raw interleaved PCM bytes

    var bytesPerFrame: Int { (bitsPerSample / 8) * channels }
    var durationSeconds: Double { Double(pcm.count / max(1, bytesPerFrame)) / Double(sampleRate) }

    static func read(_ url: URL) -> WavAudio? {
        guard let data = try? Data(contentsOf: url), data.count > 44,
              data.prefix(4).elementsEqual(Array("RIFF".utf8)) else { return nil }
        func u32(_ o: Int) -> Int { Int(data[o]) | Int(data[o+1])<<8 | Int(data[o+2])<<16 | Int(data[o+3])<<24 }
        func u16(_ o: Int) -> Int { Int(data[o]) | Int(data[o+1])<<8 }
        var sampleRate = 16000, channels = 1, bits = 16
        var pcm = Data()
        var i = 12
        while i + 8 <= data.count {
            let id = data.subdata(in: i..<i+4)
            let sz = u32(i+4)
            let body = i + 8
            if id.elementsEqual(Array("fmt ".utf8)) {
                channels = u16(body + 2)
                sampleRate = u32(body + 4)
                bits = u16(body + 14)
            } else if id.elementsEqual(Array("data".utf8)) {
                let end = min(body + sz, data.count)
                if body < end { pcm = data.subdata(in: body..<end) }
            }
            i = body + sz + (sz & 1)
        }
        guard !pcm.isEmpty else { return nil }
        return WavAudio(sampleRate: sampleRate, channels: channels, bitsPerSample: bits, pcm: pcm)
    }

    /// A standalone WAV `Data` for the [startSec, endSec) slice.
    func wavData(from startSec: Double, to endSec: Double) -> Data {
        let startByte = max(0, Int(startSec * Double(sampleRate)) * bytesPerFrame)
        let endByte = min(pcm.count, Int(endSec * Double(sampleRate)) * bytesPerFrame)
        let seg = startByte < endByte ? pcm.subdata(in: startByte..<endByte) : Data()
        return Self.makeWav(pcm: seg, sampleRate: sampleRate, channels: channels, bits: bitsPerSample)
    }

    static func makeWav(pcm: Data, sampleRate: Int, channels: Int, bits: Int) -> Data {
        func u32(_ v: Int) -> Data { var x = UInt32(v).littleEndian; return Data(bytes: &x, count: 4) }
        func u16(_ v: Int) -> Data { var x = UInt16(v).littleEndian; return Data(bytes: &x, count: 2) }
        let byteRate = sampleRate * channels * bits / 8
        let blockAlign = channels * bits / 8
        var d = Data()
        d.append(Data("RIFF".utf8)); d.append(u32(36 + pcm.count)); d.append(Data("WAVE".utf8))
        d.append(Data("fmt ".utf8)); d.append(u32(16)); d.append(u16(1)); d.append(u16(channels))
        d.append(u32(sampleRate)); d.append(u32(byteRate)); d.append(u16(blockAlign)); d.append(u16(bits))
        d.append(Data("data".utf8)); d.append(u32(pcm.count)); d.append(pcm)
        return d
    }
}

#!/usr/bin/env python3
"""Drive Meetily's built-in llama-helper sidecar to summarize a transcript with
the bundled Qwen3.5-4B model — exactly the desktop "Built-in AI" summary path,
replicated standalone for verification. Reads transcript from a file, prints the
Korean summary and timing to stdout.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
HELPER = WORKSPACE / "frontend/src-tauri/binaries/llama-helper-aarch64-apple-darwin"
MODEL = "/Users/michael/Library/Application Support/com.meetily.ai/models/summary/Qwen3.5-4B-Q4_K_M.gguf"

# Mirrors models.rs QWEN35_NONTHINKING_TEMPLATE
QWEN_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

SYSTEM_PROMPT = (
    "당신은 전문 회의 비서입니다. 아래 회의 전사록을 바탕으로 한국어로 "
    "구조화된 회의록을 작성하세요. 다음 형식을 따르세요:\n"
    "1. 회의 개요 (Executive Summary)\n"
    "2. 핵심 논의 사항\n"
    "3. 주요 결정 사항\n"
    "4. 후속 조치 (Action Items: 담당자와 기한이 언급된 경우 포함)\n\n"
    "전사록에 없는 내용을 지어내지 말고, 숫자·금액·고유명사는 그대로 보존하세요."
)


def main():
    transcript_path = Path(sys.argv[1]) if len(sys.argv) > 1 else WORKSPACE / "artifacts/ir_ko_transcript.txt"
    transcript = transcript_path.read_text(encoding="utf-8").strip()
    if not transcript:
        print("ERROR: empty transcript", file=sys.stderr)
        sys.exit(1)

    user_prompt = f"다음은 회의 전사록입니다:\n\n{transcript}"
    full_prompt = QWEN_TEMPLATE.format(system=SYSTEM_PROMPT, user=user_prompt)

    request = {
        "type": "generate",
        "prompt": full_prompt,
        "max_tokens": 4096,
        "context_size": 32768,
        "model_path": MODEL,
        "temperature": 0.5,
    }

    print(f"[driver] transcript chars: {len(transcript)}", file=sys.stderr)
    print(f"[driver] spawning {HELPER.name} (loading Qwen3.5-4B, may take ~10-30s)...", file=sys.stderr)

    proc = subprocess.Popen(
        ["nice", "-n", "10", str(HELPER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,  # llama-helper logs go to stderr; suppress for clean output
        text=True,
        bufsize=1,
    )

    t0 = time.perf_counter()
    proc.stdin.write(json.dumps(request) + "\n")
    proc.stdin.flush()

    # Read one JSON response line (model load + generation happen here)
    response_line = proc.stdout.readline()
    elapsed = time.perf_counter() - t0

    try:
        proc.stdin.close()
        proc.terminate()
    except Exception:
        pass

    if not response_line:
        print("ERROR: no response from sidecar", file=sys.stderr)
        sys.exit(1)

    resp = json.loads(response_line)
    text = resp.get("text") or resp.get("message") or ""
    err = resp.get("error")

    print("=" * 70)
    print(f"BUILT-IN QWEN3.5-4B SUMMARY  (latency: {elapsed:.1f}s, incl. model load)")
    print("=" * 70)
    print(text.strip())
    if err:
        print(f"\n[sidecar error field]: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()

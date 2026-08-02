const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const $ = (id) => document.getElementById(id);

function showScreen(id) {
  document.querySelectorAll(".screen").forEach((s) => s.classList.remove("active"));
  $(id).classList.add("active");
}

function toast(msg) {
  const t = $("error-toast");
  t.textContent = "⚠ " + msg;
  t.hidden = false;
  setTimeout(() => (t.hidden = true), 6000);
}

async function loadFiles() {
  try {
    const files = await invoke("list_audio_files");
    const list = $("file-list");
    list.innerHTML = "";
    $("files-empty").hidden = true;
    files.forEach((f) => {
      const label = f.name.replace(/\.wav$/i, "").replace(/^\d+_/, "");
      const li = document.createElement("li");
      li.className = "file-item";
      li.innerHTML = `
        <div class="file-ic">🎙️</div>
        <div class="file-body">
          <div class="file-name">${label}</div>
          <div class="file-sub">16kHz 모노 · 탭하여 실시간 전사</div>
        </div>
        <div class="file-go">›</div>`;
      li.addEventListener("click", () => startTranscription(f.name, label));
      list.appendChild(li);
    });
    if (files.length === 0) {
      $("files-empty").hidden = false;
      $("files-empty").textContent = "번들된 음성 파일이 없습니다.";
    }
  } catch (e) {
    toast("파일 목록 로드 실패: " + e);
  }
}

async function startTranscription(fileName, label) {
  $("t-filename").textContent = label;
  $("t-lang").textContent = "auto";
  $("transcript").innerHTML = "";
  $("t-progress").style.width = "0%";
  $("btn-stop").disabled = false;
  showScreen("screen-transcribe");
  try {
    await invoke("start_transcription", { fileName });
  } catch (e) {
    toast("전사 시작 실패: " + e);
  }
}

async function stopAndSummarize() {
  $("btn-stop").disabled = true;
  $("summary").textContent = "";
  $("summary-status").hidden = false;
  $("btn-restart").hidden = true;
  showScreen("screen-summary");
  try {
    await invoke("stop_and_summarize");
  } catch (e) {
    toast("요약 시작 실패: " + e);
  }
}

window.addEventListener("DOMContentLoaded", () => {
  $("btn-stop").addEventListener("click", stopAndSummarize);
  $("btn-restart").addEventListener("click", () => {
    showScreen("screen-files");
  });

  // Engine timing/status (model load secs, per-chunk ms, GPU/CPU) — shown live.
  listen("engine-status", (e) => {
    $("t-lang").textContent = e.payload;
  });

  // Live transcription chunks
  listen("transcript-update", (e) => {
    const p = e.payload;
    $("t-progress").style.width = Math.round(p.progress * 100) + "%";
    $("transcript").textContent = p.full_text;
    $("transcript").scrollTop = $("transcript").scrollHeight;
  });

  listen("transcription-done", () => {
    $("t-progress").style.width = "100%";
  });

  listen("summary-started", () => {
    $("summary-status").hidden = false;
  });

  listen("summary-ready", (e) => {
    $("summary-status").hidden = true;
    $("summary").textContent = e.payload.text;
    const note = document.createElement("div");
    note.className = "summary-foot";
    note.textContent = `온디바이스 생성 · ${e.payload.elapsed_secs.toFixed(1)}초 (CPU)`;
    $("summary").appendChild(note);
    $("btn-restart").hidden = false;
  });

  listen("engine-error", (e) => {
    $("summary-status").hidden = true;
    toast(`[${e.payload.stage}] ${e.payload.message}`);
    $("btn-restart").hidden = false;
  });

  loadFiles();
});

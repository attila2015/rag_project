"""
Guide technique — Document Intelligence avec Qwen2.5-VL
Page Streamlit dédiée aux data scientists.
Couvre : architecture, setup, pipeline, prompt engineering,
         évaluation, fine-tuning, MLOps.
"""
import streamlit as st
import streamlit.components.v1 as _stc
from pathlib import Path
import subprocess
import sys
import os

ROOT = Path(__file__).parent.parent

st.set_page_config(
    page_title="Guide technique — Document Intelligence",
    page_icon="📖",
    layout="wide",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.section-header {
    background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
    color: white; padding: 10px 18px; border-radius: 8px;
    font-size: 18px; font-weight: 700; margin: 24px 0 12px 0;
}
.callout-info {
    background: #eff6ff; border-left: 4px solid #3b82f6;
    padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 8px 0;
}
.callout-warn {
    background: #fffbeb; border-left: 4px solid #f59e0b;
    padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 8px 0;
}
.callout-tip {
    background: #f0fdf4; border-left: 4px solid #22c55e;
    padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 8px 0;
}
.metric-card {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 14px 18px; text-align: center;
}
.metric-val { font-size: 28px; font-weight: 800; color: #4f46e5; }
.metric-lab { font-size: 12px; color: #64748b; margin-top: 2px; }
.arch-box {
    background: #1e1b4b; color: #e0e7ff;
    border-radius: 10px; padding: 18px 22px;
    font-family: monospace; font-size: 13px; line-height: 1.8;
}
.step-card {
    border: 1px solid #e5e7eb; border-radius: 10px;
    padding: 14px 18px; margin: 8px 0;
    display: flex; gap: 12px; align-items: flex-start;
}
.step-num {
    background: #4f46e5; color: white;
    border-radius: 50%; width: 30px; height: 30px; min-width: 30px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("# 📖 Guide technique — Document Intelligence")
st.markdown(
    "Ce guide couvre l'ensemble du pipeline, les choix techniques, les paramètres "
    "d'optimisation et les pistes de fine-tuning."
)

# ─── Quick status ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">⚡ État de l\'environnement</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

def _check(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        return r.returncode == 0, r.stdout.strip()
    except Exception:
        return False, ""

with col1:
    ok, ver = _check(["python", "--version"])
    st.markdown(
        f'<div class="metric-card"><div class="metric-val">{"✓" if ok else "✗"}</div>'
        f'<div class="metric-lab">Python<br>{ver or "introuvable"}</div></div>',
        unsafe_allow_html=True,
    )

with col2:
    model_files = list((ROOT / "models").glob("*.gguf"))
    st.markdown(
        f'<div class="metric-card"><div class="metric-val">{len(model_files)}</div>'
        f'<div class="metric-lab">Modèle(s) GGUF<br>dans models/</div></div>',
        unsafe_allow_html=True,
    )

with col3:
    ok_st, _ = _check(["python", "-c", "import streamlit"])
    ok_oai, _ = _check(["python", "-c", "import openai"])
    ok_pil, _ = _check(["python", "-c", "import PIL"])
    n_deps = sum([ok_st, ok_oai, ok_pil])
    st.markdown(
        f'<div class="metric-card"><div class="metric-val">{n_deps}/3</div>'
        f'<div class="metric-lab">Dépendances clés<br>streamlit · openai · Pillow</div></div>',
        unsafe_allow_html=True,
    )

with col4:
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:8080/v1/models", timeout=1)
        server_status, server_color = "En ligne", "#22c55e"
    except Exception:
        server_status, server_color = "Hors ligne", "#ef4444"
    st.markdown(
        f'<div class="metric-card"><div class="metric-val" style="color:{server_color}">●</div>'
        f'<div class="metric-lab">Serveur llama.cpp<br>{server_status} :8080</div></div>',
        unsafe_allow_html=True,
    )


# ─── Architecture ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🏗️ Architecture du pipeline</div>', unsafe_allow_html=True)

_stc.html("""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
body { margin:0; padding:0; background:transparent; font-family:'Segoe UI',sans-serif; }
/* ── Architecture diagram ── */
.arch-wrap {
    display: flex; flex-direction: column; align-items: center;
    gap: 0; padding: 24px 12px; background: #0f172a; border-radius: 14px;
    font-family: 'Segoe UI', sans-serif;
}
.arch-row { display: flex; align-items: center; justify-content: center; gap: 10px; }
.arch-node {
    border-radius: 10px; padding: 10px 18px; text-align: center;
    font-size: 13px; font-weight: 600; line-height: 1.4; min-width: 160px;
}
.arch-node-input  { background:#1d4ed8; color:#fff; border: 2px solid #3b82f6; }
.arch-node-util   { background:#0f766e; color:#fff; border: 2px solid #14b8a6; }
.arch-node-server { background:#6d28d9; color:#fff; border: 2px solid #a78bfa;
                    padding: 14px 24px; min-width: 300px; }
.arch-node-model  { background:#4c1d95; color:#c4b5fd; border: 1px dashed #7c3aed;
                    font-size: 11px; margin-top: 6px; min-width: 260px; }
.arch-node-classify { background:#b45309; color:#fff; border: 2px solid #f59e0b; }
.arch-node-result   { background:#1e3a5f; color:#bae6fd; border: 1px dashed #38bdf8;
                      font-size: 11px; min-width: 180px; }
.arch-node-extract  { background:#065f46; color:#fff; border: 2px solid #10b981; }
.arch-node-output   { background:#1e293b; color:#94a3b8; border: 1px solid #475569;
                      font-size: 11px; }
.arch-node-store    { background:#1c1917; color:#d6d3d1; border: 1px solid #78716c;
                      font-size: 11px; }
.arch-arrow { color: #64748b; font-size: 22px; line-height: 1; }
.arch-arrow-h { color: #475569; font-size: 18px; }
.arch-label { color: #64748b; font-size: 10px; text-align: center; margin: 1px 0; }
.arch-group { display: flex; flex-direction: column; align-items: center; gap: 4px; }
.arch-badge {
    display: inline-block; background: #3730a3; color: #c7d2fe;
    border-radius: 4px; padding: 1px 7px; font-size: 10px; margin-top: 3px;
}
</style>

<div class="arch-wrap">

  <!-- ROW 1: Input -->
  <div class="arch-row">
    <div class="arch-node arch-node-input">
      📄 Document<br><span style="font-size:10px;opacity:.8">PDF · JPG · PNG</span>
    </div>
  </div>

  <div class="arch-arrow">↓</div>
  <div class="arch-label">pdf2image / PIL.Image.open</div>

  <!-- ROW 2: image_utils -->
  <div class="arch-row">
    <div class="arch-node arch-node-util">
      🖼 image_utils.py<br>
      <span class="arch-badge">resize ≤ 1024px</span>
      <span class="arch-badge">base64 JPEG</span>
    </div>
  </div>

  <div class="arch-arrow">↓</div>
  <div class="arch-label">HTTP POST /v1/chat/completions (OpenAI API)</div>

  <!-- ROW 3: llama.cpp server -->
  <div class="arch-row">
    <div class="arch-group">
      <div class="arch-node arch-node-server">
        ⚙️ llama.cpp server <span style="font-size:11px;opacity:.7">:8080</span><br>
        <span style="font-size:10px;font-weight:400;">Vulkan · Intel Arc · CPU fallback</span>
      </div>
      <div class="arch-node arch-node-model">
        🧠 Qwen2.5-VL-7B-Instruct · Q4_K_M · GGUF<br>
        ViT encoder (vision tokens) + Qwen2.5-7B LLM decoder<br>
        <span style="color:#a78bfa;">mmproj-F16.gguf (multimodal projector)</span>
      </div>
    </div>
  </div>

  <div class="arch-arrow">↓</div>
  <div class="arch-label">raw JSON text</div>

  <!-- ROW 4: classify + result -->
  <div class="arch-row">
    <div class="arch-node arch-node-classify">
      🏷 classify.py<br><span style="font-size:10px;">Pass 1</span>
    </div>
    <div class="arch-arrow-h">→</div>
    <div class="arch-node arch-node-result">
      ClassificationResult<br>doc_type · confidence<br>language · category
    </div>
  </div>

  <div class="arch-arrow">↓</div>
  <div class="arch-label">doc_type → prompt selection</div>

  <!-- ROW 5: extract + result -->
  <div class="arch-row">
    <div class="arch-node arch-node-extract">
      🔬 extract.py<br><span style="font-size:10px;">Pass 2</span>
    </div>
    <div class="arch-arrow-h">→</div>
    <div class="arch-node arch-node-result">
      ExtractionResult<br>fields · line_items<br>tables · raw_text
    </div>
  </div>

  <div class="arch-arrow">↓</div>

  <!-- ROW 6: outputs -->
  <div class="arch-row" style="gap:16px;">
    <div class="arch-node arch-node-output">⬇ JSON / CSV<br>export</div>
    <div class="arch-node arch-node-store">🗄 ChromaDB<br>vectorstore</div>
    <div class="arch-node arch-node-output">📈 MLflow<br>logs/mlruns</div>
  </div>

</div>
</body></html>""", height=680, scrolling=False)

st.markdown("**Choix de design :**")
_stc.html("""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body{margin:0;padding:2px;background:transparent;font-family:'Segoe UI',sans-serif;}
table{width:100%;border-collapse:collapse;border-radius:12px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,.4);}
thead tr{background:#1e1b4b;}
th{color:#c7d2fe;padding:10px 16px;font-size:11px;letter-spacing:.8px;text-transform:uppercase;font-weight:700;}
td{padding:10px 16px;font-size:13px;border-bottom:1px solid rgba(255,255,255,.04);vertical-align:top;}
.r1{background:linear-gradient(90deg,#052e16,#0a3d20);color:#bbf7d0;}
.r2{background:linear-gradient(90deg,#082031,#0c2d45);color:#bae6fd;}
.r3{background:linear-gradient(90deg,#1c1204,#241a06);color:#fde68a;}
.r4{background:linear-gradient(90deg,#1e1040,#261650);color:#ddd6fe;}
.r5{background:linear-gradient(90deg,#1a0a28,#21103a);color:#f0abfc;}
.badge{display:inline-block;padding:2px 8px;border-radius:5px;font-size:11px;font-weight:700;margin-right:4px;}
.b-green{background:#14532d;color:#86efac;}
.b-blue{background:#1e3a5f;color:#7dd3fc;}
.b-amber{background:#78350f;color:#fde68a;}
.b-purple{background:#4c1d95;color:#ddd6fe;}
.b-pink{background:#701a75;color:#f0abfc;}
</style></head><body>
<table>
<thead><tr><th>Décision</th><th>Justification</th></tr></thead>
<tbody>
<tr class="r1"><td><span class="badge b-green">① Two-pass</span><br>Classify → Extract</td><td>La classification informe le choix du prompt d'extraction → meilleure précision que single-pass</td></tr>
<tr class="r2"><td><span class="badge b-blue">② llama.cpp</span><br>Backend inference</td><td>Zéro dépendance PyTorch ; GGUF quantifié tourne sur GPU consommateur (8 GB) ; API OpenAI-compatible</td></tr>
<tr class="r3"><td><span class="badge b-amber">③ Température 0.1</span><br>Sampling</td><td>Sorties JSON déterministes ; évite les hallucinations de structure</td></tr>
<tr class="r4"><td><span class="badge b-purple">④ Resize 1024px</span><br>Préprocessing</td><td>Compromis qualité/vitesse — Qwen2.5-VL encode les images en patches 14×14 ; au-delà de 1280px, pas de gain perceptible sur documents</td></tr>
<tr class="r5"><td><span class="badge b-pink">⑤ Schéma JSON</span><br>Prompt engineering</td><td>Anchor fort → le modèle suit la structure même sur des documents partiellement illisibles</td></tr>
</tbody>
</table>
</body></html>""", height=260)

# ─── Architecture technique — niveau serveur ──────────────────────────────────
st.markdown('<div class="section-header">🖥️ Architecture technique — Stack serveur & hardware</div>', unsafe_allow_html=True)

_stc.html("""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
body { margin:0; padding:0; background:transparent; font-family:'Segoe UI',sans-serif; }
/* ── Server stack diagram ── */
.srv-wrap {
    background: #0a0e1a; border-radius: 14px; padding: 28px 20px;
    font-family: 'Segoe UI', sans-serif; display: flex; gap: 24px;
    align-items: flex-start; justify-content: center; flex-wrap: wrap;
}

/* ─ Left column: client ─ */
.srv-col { display: flex; flex-direction: column; align-items: center; gap: 0; min-width: 170px; }
.srv-col-label {
    color: #64748b; font-size: 10px; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; margin-bottom: 10px;
}

/* ─ Layer boxes ─ */
.srv-layer {
    width: 100%; border-radius: 8px; padding: 9px 14px;
    text-align: center; font-size: 12px; font-weight: 600; line-height: 1.5;
    margin-bottom: 4px;
}
.srv-arrow { color: #334155; font-size: 18px; text-align: center; margin: 2px 0; }
.srv-connector {
    display: flex; align-items: center; gap: 0;
    color: #475569; font-size: 11px; padding: 28px 4px;
}
.srv-connector-line {
    flex: 1; height: 2px; background: repeating-linear-gradient(
        90deg, #334155 0px, #334155 6px, transparent 6px, transparent 12px);
}

/* Layer colors */
.l-app    { background:#1e3a5f; color:#bae6fd; border:1px solid #0ea5e9; }
.l-py     { background:#0f3460; color:#93c5fd; border:1px dashed #3b82f6; font-size:11px; }
.l-api    { background:#1a2744; color:#7dd3fc; border:1px solid #1d4ed8; font-size:11px; }
.l-net    { background:#0f172a; color:#6366f1; border:1px dashed #4338ca; font-size:10px; }
.l-srv    { background:#2d1a5e; color:#c4b5fd; border:2px solid #7c3aed; }
.l-bin    { background:#1e1040; color:#a78bfa; border:1px dashed #6d28d9; font-size:11px; }
.l-model  { background:#13082e; color:#ddd6fe; border:1px solid #5b21b6; font-size:10px; }
.l-mmproj { background:#0d0520; color:#8b5cf6; border:1px dashed #7c3aed; font-size:10px; }
.l-vk     { background:#1a1020; color:#f0abfc; border:1px solid #a21caf; font-size:11px; }
.l-gpu    { background:#2a0a2e; color:#f5d0fe; border:2px solid #c026d3; }
.l-npu    { background:#1a0520; color:#e879f9; border:1px dashed #a21caf; font-size:10px; }
.l-hw     { background:#150020; color:#f3e8ff; border:2px solid #7c3aed; }
.l-os     { background:#0f172a; color:#94a3b8; border:1px solid #334155; font-size:10px; }
.l-store  { background:#0a2010; color:#86efac; border:1px solid #16a34a; font-size:11px; }
.l-mlflow { background:#071520; color:#7dd3fc; border:1px solid #0369a1; font-size:11px; }

.srv-badge {
    display:inline-block; background:#312e81; color:#c7d2fe;
    border-radius:3px; padding:1px 5px; font-size:9px; margin:1px;
}
.srv-badge-green { background:#14532d; color:#86efac; }
.srv-badge-purple { background:#4c1d95; color:#ddd6fe; }
</style>

<div class="srv-wrap">

  <!-- ═══ COL 1 : Client / Streamlit UI ═══ -->
  <div class="srv-col">
    <div class="srv-col-label">⬛ Client UI</div>

    <div class="srv-layer l-app">
      🌐 Navigateur<br>
      <span style="font-size:10px;opacity:.8">localhost:8501</span>
    </div>
    <div class="srv-arrow">↕</div>
    <div class="srv-layer l-py">
      🐍 Streamlit 1.x<br>
      <span class="srv-badge">app.py</span>
      <span class="srv-badge">02_Pipeline.py</span>
    </div>
    <div class="srv-arrow">↓</div>
    <div class="srv-layer l-py" style="font-size:11px;">
      📦 Python stack<br>
      <span class="srv-badge">openai</span>
      <span class="srv-badge">Pillow</span>
      <span class="srv-badge">pdf2image</span>
    </div>
    <div class="srv-arrow">↓</div>
    <div class="srv-layer l-api">
      📡 OpenAI-compatible<br>
      POST /v1/chat/completions<br>
      <span style="font-size:10px;opacity:.8;">base64 image + JSON prompt</span>
    </div>
    <div class="srv-arrow">↓</div>
    <div class="srv-layer l-store">
      🗄 ChromaDB<br>
      <span style="font-size:10px;">data/vectorstore/ · persist</span>
    </div>
    <div class="srv-arrow">↓</div>
    <div class="srv-layer l-mlflow">
      📈 MLflow<br>
      <span style="font-size:10px;">logs/mlruns/ · :5000</span>
    </div>
  </div>

  <!-- ═══ CONNECTOR ═══ -->
  <div class="srv-connector">
    <div class="srv-connector-line"></div>
    <div style="padding:0 8px;color:#4f46e5;font-size:16px;">⟷</div>
    <div style="color:#6366f1;font-size:10px;writing-mode:vertical-lr;transform:rotate(180deg);padding:4px 2px;">HTTP :8080</div>
    <div style="padding:0 8px;color:#4f46e5;font-size:16px;">⟷</div>
    <div class="srv-connector-line"></div>
  </div>

  <!-- ═══ COL 2 : Inference server ═══ -->
  <div class="srv-col" style="min-width:220px;">
    <div class="srv-col-label">⚙️ Inference Server</div>

    <div class="srv-layer l-srv">
      ⚙️ llama-server.exe<br>
      <span style="font-size:10px;font-weight:400;">llama.cpp b8262 · host 0.0.0.0:8080</span>
    </div>
    <div class="srv-arrow">↓</div>
    <div class="srv-layer l-bin">
      📁 bin/vulkan/ (Intel Arc)<br>
      <span class="srv-badge srv-badge-purple">ggml_vulkan</span>
      <span class="srv-badge">cpu fallback</span>
    </div>
    <div class="srv-arrow">↓</div>
    <div class="srv-layer l-model">
      🧠 Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf<br>
      ~5.2 GB · 4-bit quantized<br>
      <span class="srv-badge srv-badge-purple">ViT-G/14 patches</span>
      <span class="srv-badge srv-badge-purple">Qwen2.5-7B LLM</span>
    </div>
    <div class="srv-arrow">↓</div>
    <div class="srv-layer l-mmproj">
      🔗 mmproj-F16.gguf<br>
      Multimodal projector (vision→tokens)<br>
      ~300 MB · float16
    </div>
    <div class="srv-arrow">↓</div>
    <div class="srv-layer l-vk">
      🔷 Vulkan API<br>
      <span style="font-size:10px;">GPU compute · shader dispatch</span><br>
      <span class="srv-badge">n_gpu_layers: 20</span>
      <span class="srv-badge">ctx: 4096</span>
    </div>
    <div class="srv-arrow">↓</div>
    <div class="srv-layer l-gpu">
      🎮 Intel Arc Graphics (UMA)<br>
      <span style="font-size:10px;font-weight:400;">~17.5 GB shared RAM · Xe-HPG</span><br>
      <span class="srv-badge srv-badge-purple">Meteor Lake iGPU</span>
    </div>
    <div class="srv-arrow">↓</div>
    <div class="srv-layer l-npu">
      🔮 Intel NPU AI Boost<br>
      <span style="font-size:10px;">~11.5 TOPS · OpenVINO (optionnel)</span><br>
      <span class="srv-badge">INT8/INT4 matmul</span>
    </div>
    <div class="srv-arrow">↓</div>
    <div class="srv-layer l-hw">
      💻 Intel Core Ultra 5 125H<br>
      <span style="font-size:10px;font-weight:400;">Meteor Lake · 14 cores · 32 GB DDR5</span><br>
      <span class="srv-badge srv-badge-green">CPU + GPU + NPU on-die</span>
    </div>
    <div class="srv-arrow">↓</div>
    <div class="srv-layer l-os">
      🪟 Windows 11 · Python 3.12<br>
      <span style="font-size:10px;">Intel Graphics Driver 31.x · Vulkan SDK</span>
    </div>
  </div>

</div>
</body></html>""", height=920, scrolling=False)

st.markdown("**Workflow détaillé — du clic au JSON :**")
_stc.html("""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body{margin:0;padding:2px;background:transparent;font-family:'Segoe UI',sans-serif;}
table{width:100%;border-collapse:collapse;border-radius:12px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,.4);}
thead tr{background:#1e1b4b;}
th{color:#c7d2fe;padding:9px 14px;font-size:11px;letter-spacing:.8px;text-transform:uppercase;font-weight:700;}
td{padding:8px 14px;font-size:12px;border-bottom:1px solid rgba(255,255,255,.04);}
.num{font-weight:800;font-size:14px;text-align:center;width:36px;}
.tag{display:inline-block;padding:2px 7px;border-radius:4px;font-size:10px;font-weight:600;}
.t-ui{background:#1e3a5f;color:#7dd3fc;}
.t-py{background:#065f46;color:#6ee7b7;}
.t-http{background:#78350f;color:#fde68a;}
.t-gpu{background:#4c1d95;color:#ddd6fe;}
.t-llm{background:#831843;color:#fbcfe8;}
.t-local{background:#1e293b;color:#94a3b8;}
tr:nth-child(1){background:#0a1628;}tr:nth-child(2){background:#0a1e28;}
tr:nth-child(3){background:#1a1228;}tr:nth-child(4){background:#100820;}
tr:nth-child(5){background:#150820;}tr:nth-child(6){background:#1a0820;}
tr:nth-child(7){background:#100820;}tr:nth-child(8){background:#0a1e28;}
tr:nth-child(9){background:#0a1628;}tr:nth-child(10){background:#071520;}
tr{color:#e2e8f0;}
</style></head><body>
<table>
<thead><tr><th>#</th><th>Acteur</th><th>Action</th><th>Protocole</th></tr></thead>
<tbody>
<tr><td class="num">1</td><td><span class="tag t-ui">UI</span> Streamlit</td><td>Upload fichier → PIL.Image ou pdf2image</td><td><span class="tag t-local">local</span></td></tr>
<tr><td class="num">2</td><td><span class="tag t-py">🐍</span> image_utils.py</td><td>Resize ≤ 1024px + base64 JPEG encode</td><td><span class="tag t-local">in-process</span></td></tr>
<tr><td class="num">3</td><td><span class="tag t-py">🐍</span> openai client</td><td>POST /v1/chat/completions avec image_url</td><td><span class="tag t-http">HTTP/1.1</span></td></tr>
<tr><td class="num">4</td><td><span class="tag t-gpu">⚙️</span> llama-server.exe</td><td>Reçoit la requête, tokenise via mmproj</td><td><span class="tag t-gpu">Vulkan dispatch</span></td></tr>
<tr><td class="num">5</td><td><span class="tag t-gpu">🔷</span> ggml_vulkan</td><td>Dispatch GPU shaders → inference Intel Arc</td><td><span class="tag t-gpu">GPU compute</span></td></tr>
<tr><td class="num">6</td><td><span class="tag t-llm">🧠</span> Qwen2.5-VL</td><td>Génère la réponse JSON token par token</td><td><span class="tag t-llm">LLM decode</span></td></tr>
<tr><td class="num">7</td><td><span class="tag t-gpu">⚙️</span> llama-server.exe</td><td>Stream la réponse au client</td><td><span class="tag t-http">HTTP/1.1</span></td></tr>
<tr><td class="num">8</td><td><span class="tag t-py">🐍</span> classify / extract</td><td>Parse le JSON, structure le résultat</td><td><span class="tag t-local">in-process</span></td></tr>
<tr><td class="num">9</td><td><span class="tag t-local">🗄</span> vector_store.py</td><td>Upsert dans ChromaDB (embeddings)</td><td><span class="tag t-local">local SQLite</span></td></tr>
<tr><td class="num">10</td><td><span class="tag t-ui">📈</span> MLflow</td><td>Log métriques + params dans logs/mlruns/</td><td><span class="tag t-local">local FileStore</span></td></tr>
</tbody>
</table>
</body></html>""", height=330)


# ─── Modèle : Qwen2.5-VL ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">🧠 Qwen2.5-VL — Ce qu\'il faut savoir</div>', unsafe_allow_html=True)

col_l, col_r = st.columns(2)

with col_l:
    st.markdown("""
**Architecture (7B)**
- Vision encoder : **ViT-G/14** (SigLIP-400M) adapté pour documents haute résolution
- Résolution dynamique : découpage adaptatif en tiles (up to 36 tiles × 1024px)
- LLM backbone : **Qwen2.5-7B** (MHA + RoPE + SwiGLU)
- Context window : 32k tokens
- Entraîné sur : documents, tableaux, OCR, graphiques, scènes naturelles

**Formats GGUF disponibles**
""")
    _stc.html("""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body{margin:0;padding:2px;background:transparent;font-family:'Segoe UI',sans-serif;}
table{width:100%;border-collapse:collapse;border-radius:12px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,.4);}
thead tr{background:#1e1b4b;}
th{color:#c7d2fe;padding:9px 12px;font-size:10px;letter-spacing:.8px;text-transform:uppercase;font-weight:700;}
td{padding:9px 12px;font-size:12px;border-bottom:1px solid rgba(255,255,255,.04);}
.r1{background:linear-gradient(90deg,#052e16,#0a3d20);color:#bbf7d0;}
.r2{background:linear-gradient(90deg,#082031,#0c2d45);color:#bae6fd;}
.r3{background:linear-gradient(90deg,#1c1410,#261a14);color:#fde68a;}
.r4{background:linear-gradient(90deg,#1e293b,#263245);color:#94a3b8;}
.medal{font-size:14px;}
.badge{display:inline-block;padding:1px 6px;border-radius:4px;font-size:10px;font-weight:700;}
.rec{background:#14532d;color:#86efac;}
.speed-fast{color:#4ade80;font-weight:700;}
.speed-med{color:#facc15;font-weight:700;}
.speed-slow{color:#f97316;font-weight:600;}
.speed-vslow{color:#ef4444;font-weight:600;}
</style></head><body>
<table>
<thead><tr><th>#</th><th>Quant</th><th>VRAM</th><th>Qualité</th><th>Vitesse</th></tr></thead>
<tbody>
<tr class="r1"><td class="medal">🥇</td><td><strong>Q4_K_M</strong> <span class="badge rec">★ Reco</span></td><td>~5.2 GB</td><td>⭐⭐⭐ Bon</td><td class="speed-fast">⚡⚡⚡ Rapide</td></tr>
<tr class="r2"><td class="medal">🥈</td><td><strong>Q5_K_M</strong></td><td>~6.1 GB</td><td>⭐⭐⭐⭐ Très bon</td><td class="speed-med">⚡⚡ Moyen</td></tr>
<tr class="r3"><td class="medal">🥉</td><td><strong>Q8_0</strong></td><td>~8.1 GB</td><td>⭐⭐⭐⭐⭐ Quasi-FP16</td><td class="speed-slow">⚡ Lent</td></tr>
<tr class="r4"><td>4</td><td><strong>F16</strong></td><td>~14 GB</td><td>🎯 Référence</td><td class="speed-vslow">🐌 Très lent</td></tr>
</tbody>
</table>
</body></html>""", height=185)
    st.info("**Recommandation :** Q4_K_M pour l'expérimentation, Q5_K_M pour la production.")

with col_r:
    st.markdown("""
**Capacités document-spécifiques**
- OCR natif (pas de preprocessing Tesseract nécessaire)
- Lecture de tableaux, colonnes, formulaires
- Compréhension de mise en page (header, footer, numéro de page)
- Multi-langue (FR, EN, DE, ES, ZH…)
- Signatures, tampons, annotations manuscrites (partiel)

**Limites à connaître**
- Documents très denses > 3 colonnes : découpage recommandé
- Tableaux complexes imbriqués : taux d'erreur élevé
- Handwriting dégradé : performance chute fortement
- Confidentialité : **les données restent 100% locales** (avantage clé)

**Comparaison avec concurrents**
""")
    _stc.html("""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body{margin:0;padding:2px;background:transparent;font-family:'Segoe UI',sans-serif;}
table{width:100%;border-collapse:collapse;border-radius:12px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,.4);}
thead tr{background:#1e1b4b;}
th{color:#c7d2fe;padding:9px 12px;font-size:10px;letter-spacing:.8px;text-transform:uppercase;font-weight:700;}
td{padding:9px 12px;font-size:12px;border-bottom:1px solid rgba(255,255,255,.04);}
.r-hero{background:linear-gradient(90deg,#052e16,#0a3d20);color:#bbf7d0;}
.r2{background:linear-gradient(90deg,#082031,#0c2d45);color:#bae6fd;}
.r3{background:linear-gradient(90deg,#1c1410,#261a14);color:#fde68a;}
.r4{background:#0f172a;color:#94a3b8;}
.yes{color:#4ade80;font-weight:800;font-size:15px;}
.yes2{color:#86efac;font-weight:800;}
.yes3{color:#fbbf24;font-weight:800;}
.no{color:#f87171;font-weight:700;}
.medal{font-size:14px;}
.badge{display:inline-block;padding:1px 6px;border-radius:4px;font-size:10px;font-weight:700;background:#14532d;color:#86efac;}
.cloud{display:inline-block;padding:1px 6px;border-radius:4px;font-size:10px;font-weight:700;background:#450a0a;color:#fca5a5;}
</style></head><body>
<table>
<thead><tr><th>#</th><th>Modèle</th><th>OCR</th><th>Tables</th><th>Local</th><th>Taille</th></tr></thead>
<tbody>
<tr class="r-hero"><td class="medal">🥇</td><td><strong>Qwen2.5-VL 7B</strong> <span class="badge">★ Notre choix</span></td><td><span class="yes2">✓✓</span></td><td><span class="yes2">✓✓</span></td><td><span class="yes">✓</span></td><td>7B</td></tr>
<tr class="r2"><td class="medal">🥈</td><td><strong>Mistral Pixtral</strong></td><td><span class="yes2">✓✓</span></td><td><span class="yes2">✓✓</span></td><td><span class="yes">✓</span></td><td>12B</td></tr>
<tr class="r3"><td class="medal">🥉</td><td><strong>LLaVA-Next</strong></td><td><span class="yes2">✓</span></td><td><span class="yes2">✓</span></td><td><span class="yes">✓</span></td><td>7B–34B</td></tr>
<tr class="r4"><td>☁</td><td><strong>GPT-4o</strong> <span class="cloud">cloud</span></td><td><span class="yes3">✓✓✓</span></td><td><span class="yes3">✓✓✓</span></td><td><span class="no">✗</span></td><td>—</td></tr>
</tbody>
</table>
</body></html>""", height=195)


# ─── Pipeline pas à pas ───────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔄 Pipeline — Détail technique étape par étape</div>', unsafe_allow_html=True)

steps = [
    ("⚙️", "Configuration du serveur",
     """**Ce qui se passe :** L'UI stocke les paramètres en `st.session_state`.
La connexion est testée via `GET /v1/models` (endpoint OpenAI-compatible).

**Démarrage direct depuis la sidebar :**
Le pipeline expose un bouton **▶ Démarrer llama.cpp** dans la sidebar avec un choix de backend :
- `auto` : détecte automatiquement Intel Arc (Vulkan) ou CPU
- `vulkan` : force Intel Arc via `bin/vulkan/llama-server.exe` (20 GPU layers)
- `cpu` : force CPU via `bin/cpu/llama-server.exe` (0 GPU layers, plus lent)

Le script `scripts/start_server.ps1 -Backend <choix>` est lancé via `subprocess.Popen`.
⚠️ **Délai de chargement** : le modèle Q4_K_M (~5.2 GB) prend **30–90 secondes** à charger.
Le statut reste "hors ligne" jusqu'au message `llama server listening at http://0.0.0.0:8080` dans le terminal.

**Paramètres critiques :**
- `n_gpu_layers` : nombre de couches déchargées sur GPU. Chaque couche ≈ 150 MB VRAM.
  Formule : `n_gpu_layers = min(32, VRAM_disponible_GB * 1000 / 150)`
- `n_ctx` : fenêtre de contexte en tokens. Une image = 256–1024 tokens selon résolution.
  Augmenter si extraction incomplète (>2048 recommandé).
- `temperature=0.1` : suffisant pour JSON structuré. Ne pas dépasser 0.3.
"""),
    ("📄", "Upload et préprocessing",
     """**Pipeline image :**
```
PIL.Image.open() → EXIF rotation → convert("RGB")
→ resize (max 1024px, LANCZOS) → JPEG compress (q=92)
→ base64 encode → data URI
```

**PDF :** `pdf2image` convertit chaque page en PIL.Image via Poppler (DPI=200).
DPI 200 = bon compromis lisibilité/tokens. DPI 150 pour docs simples, 300 pour docs très denses.

**Pourquoi JPEG et pas PNG ?**
JPEG divise par 3-5× la taille du payload HTTP → latence réduite.
Qualité 92 préserve le texte sans artefacts visibles.
"""),
    ("🏷️", "Classification",
     """**Prompt strategy — Chain-of-schema :**
Le système prompt impose le mode JSON-only dès le début.
Le user prompt contient le schéma attendu en exemple → "few-shot structurel".

```python
# Appel effectif
response = client.chat.completions.create(
    model="qwen2.5-vl-7b-q4_k_m",
    temperature=0.1,
    messages=[
        {"role": "system", "content": "...JSON only..."},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text",      "text": schema_prompt},
        ]},
    ]
)
```

**Types détectés :** invoice · contract · bank_statement · report · form · receipt · id_document · letter · other

**Interprétation de la confidence :**
- > 0.85 : passer directement en extraction
- 0.60–0.85 : vérifier le type avant extraction
- < 0.60 : document ambigu ou faible qualité scan
"""),
    ("🔬", "Extraction",
     """**Prompt adaptatif par type :**
`extract.py` sélectionne un template parmi 4 (invoice, bank_statement, contract, default).
Chaque template ancre les noms de champs attendus → réduit les hallucinations.

**Parsing robuste :**
```python
# infer.py — parse_json_from_response()
# 1. Supprime les fences markdown (```json ... ```)
# 2. json.loads() direct
# 3. Fallback : regex {[\s\S]+} pour extraire le premier bloc JSON
```

**Quand le JSON échoue :**
Symptôme → `ValueError: Cannot parse JSON`
- Augmenter `max_tokens` (le modèle a été coupé en plein milieu)
- Réduire la complexité du schéma
- Ajouter dans le system prompt : *"Never truncate the JSON. If unsure of a value, use null."*
"""),
    ("📊", "Résultats et export",
     """**Structures de sortie (Pydantic) :**
```python
ExtractionResult(
    document_type = "invoice",
    confidence    = 0.91,
    fields        = {"invoice_number": "INV-001", "total": "4250 EUR", ...},
    line_items    = [LineItem(description=..., qty=..., unit_price=..., total=...)],
    tables        = [[...]],  # brut
    metadata      = ExtractionMetadata(model=..., latency_ms=3420, ...)
)
```

**MLflow logging (optionnel) :**
Chaque run loggue : `classify_confidence`, `n_fields`, `latency_ms`, artifact JSON.
Accès : `mlflow ui --backend-store-uri logs/mlruns` → http://localhost:5000
"""),
]

for icon, title, content in steps:
    with st.expander(f"{icon}  {title}", expanded=False):
        st.markdown(content)


# ─── Prompt Engineering ───────────────────────────────────────────────────────
st.markdown('<div class="section-header">✍️ Prompt Engineering — Bonnes pratiques</div>', unsafe_allow_html=True)

col_do, col_dont = st.columns(2)

with col_do:
    st.markdown("#### ✅ À faire")
    st.markdown("""
**1. Schéma JSON dans le prompt**
```
Retourne exactement ce JSON :
{
  "invoice_number": "",
  "date": "",
  "total": ""
}
```

**2. Instructions négatives explicites**
```
Si un champ n'est pas visible, utilise null.
Ne génère jamais de valeurs inventées.
```

**3. Langue de sortie**
```
Tous les noms de champs en anglais.
Les valeurs dans la langue du document.
```

**4. Temperature = 0.1** pour JSON, 0.3 pour résumés

**5. Répéter le schéma** dans system ET user prompt
pour les documents complexes
""")

with col_dont:
    st.markdown("#### ❌ À éviter")
    st.markdown("""
**1. Instructions vagues**
```
❌ "Extrais les informations importantes"
✓  "Extrais invoice_number, date, total"
```

**2. Schéma trop long en une passe**
→ Couper en 2 appels : métadonnées d'abord,
  lignes ensuite

**3. Demander du markdown**
```
❌ "Formate en tableau markdown"
✓  "Retourne un JSON avec une liste line_items"
```

**4. Temperature élevée**
→ Temperature > 0.5 produit des JSON malformés

**5. Ignorer max_tokens**
→ Sous-estimer tronque le JSON à mi-chemin
→ Budget : ~2 tokens / caractère JSON
""")

st.markdown("#### Patterns avancés")

tab_p1, tab_p2, tab_p3 = st.tabs(["Two-pass (recommandé)", "Self-validation", "Multi-page"])

with tab_p1:
    st.markdown("""
**Deux appels séquentiels :**
1. Appel 1 → classification rapide (schema léger, ~50 tokens out)
2. Appel 2 → extraction complète avec prompt adapté au type détecté

**Avantage :** Le prompt d'extraction peut être 3× plus précis car il connaît le contexte.
**Coût :** +1 aller-retour (~500ms latence supplémentaire, négligeable vs qualité gagnée).
""")

with tab_p2:
    st.code("""# Ajouter en fin de user prompt :
SELF_VALIDATE = '''
Avant de répondre, vérifie que :
1. Chaque champ a une valeur ou null (jamais "")
2. Les montants incluent la devise
3. Les dates sont au format ISO 8601 (YYYY-MM-DD)
4. Le JSON est valide et complet
'''""", language="python")

with tab_p3:
    st.markdown("""
**Pour les documents multi-pages (contrats, rapports) :**
```python
# Stratégie recommandée :
# 1. Page 1 → classification + extraction header
# 2. Pages 2..N → extraction incrémentale avec contexte
# 3. Merge des résultats

results = []
for i, page_img in enumerate(pages):
    context = f"Page {i+1}/{len(pages)}. Type détecté: {doc_type}."
    r = extract_with_context(page_img, context, previous_fields=results)
    results.append(r)
```
""")


# ─── Évaluation ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📏 Évaluation de la qualité d\'extraction</div>', unsafe_allow_html=True)

st.markdown("""
Il n'existe pas de métriques standard pour l'extraction document → définir les vôtres.

**Métriques recommandées par niveau :**
""")

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.markdown("""
**Champ-niveau**
- **Field Recall** = champs extraits / champs attendus
- **Field Precision** = champs corrects / champs extraits
- **Exact Match** = valeur identique à la GT
- **Fuzzy Match** (pour montants, dates) = tolérance de format

```python
from difflib import SequenceMatcher
def fuzzy(a, b):
    return SequenceMatcher(None, str(a), str(b)).ratio()
```
""")
with col_m2:
    st.markdown("""
**Document-niveau**
- **Confidence threshold** : seuil à calibrer (0.7–0.85)
- **Parse success rate** : % de réponses JSON valides
- **Latency P50/P95** : objectif < 5s/doc sur GPU 8GB
- **Type accuracy** : classification correcte

**Outil rapide :**
```python
# Pour 50 docs labelisés :
# precision/recall par type de champ
# confusion matrix doc_type
```
""")
with col_m3:
    st.markdown("**Repères terrain (Qwen2.5-VL 7B Q4)**")
    _stc.html("""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body{margin:0;padding:2px;background:transparent;font-family:'Segoe UI',sans-serif;}
table{width:100%;border-collapse:collapse;border-radius:12px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,.4);}
thead tr{background:#1e1b4b;}
th{color:#c7d2fe;padding:9px 12px;font-size:10px;letter-spacing:.8px;text-transform:uppercase;font-weight:700;}
td{padding:8px 12px;font-size:12px;border-bottom:1px solid rgba(255,255,255,.04);}
.r1{background:linear-gradient(90deg,#052e16,#0a3d20);color:#bbf7d0;}
.r2{background:linear-gradient(90deg,#0a2810,#0e3218);color:#a7f3d0;}
.r3{background:linear-gradient(90deg,#082031,#0c2d45);color:#bae6fd;}
.r4{background:linear-gradient(90deg,#1c1410,#261a14);color:#fde68a;}
.r5{background:linear-gradient(90deg,#2a0a0a,#380e0e);color:#fca5a5;}
.medal{font-size:13px;text-align:center;}
.bar-wrap{background:#1e293b;border-radius:4px;height:8px;width:100%;margin-top:3px;}
.bar{height:8px;border-radius:4px;}
</style></head><body>
<table>
<thead><tr><th>#</th><th>Doc type</th><th>Field F1</th><th>Parse OK</th></tr></thead>
<tbody>
<tr class="r1"><td class="medal">🥇</td><td>Invoice (simple)</td><td>~0.87<div class="bar-wrap"><div class="bar" style="width:87%;background:#4ade80;"></div></div></td><td>~96%</td></tr>
<tr class="r2"><td class="medal">🥈</td><td>Form</td><td>~0.82<div class="bar-wrap"><div class="bar" style="width:82%;background:#86efac;"></div></div></td><td>~94%</td></tr>
<tr class="r3"><td class="medal">🥉</td><td>Bank stmt</td><td>~0.79<div class="bar-wrap"><div class="bar" style="width:79%;background:#7dd3fc;"></div></div></td><td>~93%</td></tr>
<tr class="r4"><td>4</td><td>Invoice (dense)</td><td>~0.74<div class="bar-wrap"><div class="bar" style="width:74%;background:#fde68a;"></div></div></td><td>~91%</td></tr>
<tr class="r5"><td>5</td><td>Contract</td><td>~0.61<div class="bar-wrap"><div class="bar" style="width:61%;background:#f87171;"></div></div></td><td>~89%</td></tr>
</tbody>
</table>
</body></html>""", height=245)
    st.caption("*Mesures indicatives sur docs FR/EN*")


# ─── Fine-tuning ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🎯 Fine-tuning Qwen2.5-VL — Quand et comment</div>', unsafe_allow_html=True)

col_ft1, col_ft2 = st.columns([1, 1])

with col_ft1:
    st.markdown("""
**Quand fine-tuner ?**

Fine-tuner n'est utile que si :
- F1 de base < 0.70 sur votre type de document spécifique
- Format propriétaire non couvert (bulletins de paie FR, liasses fiscales…)
- Volume > 10k docs/jour → latence critique → modèle plus petit fine-tuné

**Ne pas fine-tuner si :**
- Le prompt engineering permet d'atteindre F1 > 0.80
- < 200 exemples labelisés disponibles
- GPU < 24 GB (utiliser LoRA + 4-bit minimum)

**Dataset minimum :**
- 200–500 paires `{image, JSON attendu}` par type de document
- Ratio train/val : 80/20
- Augmentation : rotation ±5°, bruit JPEG, variation lumière
""")

with col_ft2:
    st.markdown("""
**Stack recommandée :**
```python
# notebooks/finetune_qwen_vl.ipynb
transformers >= 4.44
peft          >= 0.11   # LoRA
trl           >= 0.9    # SFTTrainer
bitsandbytes  >= 0.43   # 4-bit QLoRA
accelerate    >= 0.30

# Config LoRA minimale :
LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj","v_proj","k_proj","o_proj"],
    lora_dropout=0.05,
)
```

**Étapes :**
1. Préparer dataset en format messages ChatML
2. Lancer `notebooks/finetune_qwen_vl.ipynb`
3. Sauvegarder l'adapter LoRA
4. Merger avec le modèle de base : `merge_and_unload()`
5. Convertir en GGUF : `llama.cpp/convert_hf_to_gguf.py`
6. Remplacer le `.gguf` dans `models/`

**VRAM requise :**
""")
    _stc.html("""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body{margin:0;padding:2px;background:transparent;font-family:'Segoe UI',sans-serif;}
table{width:100%;border-collapse:collapse;border-radius:12px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,.4);}
thead tr{background:#1e1b4b;}
th{color:#c7d2fe;padding:9px 12px;font-size:10px;letter-spacing:.8px;text-transform:uppercase;font-weight:700;}
td{padding:10px 12px;font-size:12px;border-bottom:1px solid rgba(255,255,255,.04);}
.r1{background:linear-gradient(90deg,#052e16,#0a3d20);color:#bbf7d0;}
.r2{background:linear-gradient(90deg,#082031,#0c2d45);color:#bae6fd;}
.medal{font-size:14px;text-align:center;}
.badge{display:inline-block;padding:2px 7px;border-radius:4px;font-size:11px;font-weight:700;background:#14532d;color:#86efac;}
</style></head><body>
<table>
<thead><tr><th>#</th><th>Config</th><th>VRAM</th><th>Durée (500 ex)</th></tr></thead>
<tbody>
<tr class="r1"><td class="medal">🥇</td><td><strong>QLoRA 4-bit</strong> <span class="badge">★ Accessible</span></td><td>18–20 GB</td><td>~2h RTX 3090</td></tr>
<tr class="r2"><td class="medal">🥈</td><td><strong>LoRA FP16</strong></td><td>28–32 GB</td><td>~1h A100</td></tr>
</tbody>
</table>
</body></html>""", height=120)


# ─── MLOps ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔧 MLOps minimal — Logging et versioning</div>', unsafe_allow_html=True)

col_ml1, col_ml2 = st.columns(2)

with col_ml1:
    # Check if MLflow UI is running
    _mlflow_up = False
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:5000", timeout=1)
        _mlflow_up = True
    except Exception:
        pass

    if _mlflow_up:
        st.success("✓ MLflow UI en ligne — [http://localhost:5000](http://localhost:5000)")
    else:
        st.warning("⚠ MLflow UI hors ligne")
        st.code("pwsh poc_qwen/scripts/start_mlflow.ps1", language="powershell")
        st.caption("Ouvrez un nouveau terminal et lancez cette commande, puis accédez à http://localhost:5000")

    st.markdown("""
**MLflow — loggué à chaque extraction :**
- Paramètres : `doc_type`, `source_file`
- Métriques : `classify_confidence`, `extract_confidence`, `n_fields`, `n_line_items`, `latency_ms`
- Experiment : `qwen_vl_pipeline` dans `logs/mlruns/`

**Versionner les modèles :**
```python
import mlflow
mlflow.set_experiment("qwen_vl_production")
with mlflow.start_run(run_name="qwen-7b-q4-v2"):
    mlflow.log_param("model_version", "v2-finetuned")
    mlflow.log_artifact("models/icb_bert_onto_mlp.joblib")
```
""")

with col_ml2:
    st.markdown("""
**Logging structuré (recommandé en prod)**

```python
import logging, json
from datetime import datetime

def log_inference(doc_path, result, latency_ms):
    entry = {
        "ts":         datetime.utcnow().isoformat(),
        "doc":        str(doc_path),
        "doc_type":   result.get("document_type"),
        "confidence": result.get("confidence"),
        "n_fields":   len(result.get("fields", {})),
        "latency_ms": latency_ms,
        "success":    result.get("success", True),
    }
    logging.info(json.dumps(entry))
```

**Stack monitoring production :**
```
Streamlit → llama.cpp → Prometheus scrape /metrics
                      → Grafana dashboard
                      → Alerting latence P95 > 10s
```

**Reproductibilité :**
- Fixer le seed (non disponible sur llama.cpp server)
- Logguer la version du `.gguf` utilisé (hash MD5)
- Sauvegarder le prompt exact dans chaque run MLflow
""")


# ─── Troubleshooting ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🐛 Troubleshooting — Problèmes fréquents</div>', unsafe_allow_html=True)

issues = [
    ("Connection refused :8081 / Serveur hors ligne",
     "Le serveur n'est pas démarré, ou le modèle est encore en cours de chargement.",
     "**Option 1 — depuis la sidebar du pipeline :**  \n"
     "Cliquer **▶ Démarrer llama.cpp** → choisir `auto`, `vulkan` (Intel Arc) ou `cpu`.  \n"
     "⚠️ Attendre **30–90 secondes** (chargement ~5.2 GB). Le statut reste rouge jusqu'au message "
     "`llama server listening at http://0.0.0.0:8080` dans le terminal.  \n\n"
     "**Option 2 — terminal :**  \n"
     "```powershell\npwsh scripts/start_server.ps1 -Backend vulkan  # Intel Arc\npwsh scripts/start_server.ps1 -Backend cpu     # fallback\n```"),
    ("CUDA out of memory",
     "`n_gpu_layers` trop élevé pour votre GPU.",
     "Réduire `n_gpu_layers`. Formule : `floor(VRAM_GB * 1000 / 150)`. "
     "Exemple : 8 GB → max 53 layers mais Qwen2.5-VL 7B n'a que 32 layers."),
    ("ValueError: Cannot parse JSON",
     "Le modèle a retourné du texte hors JSON ou a été tronqué.",
     "1. Augmenter `max_tokens` à 3000+  \n"
     "2. Ajouter au system prompt : *Never truncate JSON. Use null for unknown fields.*  \n"
     "3. Vérifier la réponse brute dans l'onglet JSON."),
    ("Extraction vide ou champs à null",
     "Image trop petite, résolution insuffisante ou document trop dense.",
     "1. Augmenter DPI lors de la conversion PDF (200→300)  \n"
     "2. Découper l'image en zones (header, body, footer)  \n"
     "3. Prétraiter avec un filtre de netteté (Pillow `ImageFilter.SHARPEN`)."),
    ("Réponse très lente (>30s)",
     "CPU only ou peu de GPU layers.",
     "Vérifier avec `nvidia-smi` que le GPU est utilisé.  \n"
     "Sinon : réinstaller avec `-GPU` ou utiliser Q4_K_S (variante plus légère)."),
    ("Mauvaise classification (doc_type = 'other')",
     "Document ambigu ou hors domaine.",
     "1. Éditer le prompt de classification et ajouter des exemples : "
     "*'Un document avec un numéro d'ordre et un montant est une invoice.'*  \n"
     "2. Ajouter votre type dans la liste du prompt (`<invoice|contrat_bail|...>`)."),
    ("DLL load failed (Windows)",
     "Visual C++ Redistributable manquant.",
     "Installer depuis microsoft.com : **Visual C++ Redistributable 2019+** (x64)."),
]

for title, cause, fix in issues:
    with st.expander(f"❗ {title}"):
        st.markdown(f"**Cause :** {cause}")
        st.markdown(f"**Fix :**  \n{fix}")


# ─── Liens et ressources ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">📚 Ressources</div>', unsafe_allow_html=True)

col_r1, col_r2, col_r3 = st.columns(3)
with col_r1:
    st.markdown("""
**Modèle**
- [Qwen2.5-VL Paper (arXiv)](https://arxiv.org/abs/2502.13923)
- [HF Model Card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [GGUF files](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-GGUF)
""")
with col_r2:
    st.markdown("""
**Infra**
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [GGUF spec](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
""")
with col_r3:
    st.markdown("""
**Fine-tuning**
- [PEFT LoRA docs](https://huggingface.co/docs/peft/conceptual_guides/lora)
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [QLoRA paper](https://arxiv.org/abs/2305.14314)
""")


# ─── LoRA — Schéma illustratif ────────────────────────────────────────────────
st.markdown('<div class="section-header">🔗 LoRA — Comprendre le mécanisme et savoir quand l\'utiliser</div>', unsafe_allow_html=True)

with st.expander("📐 Schéma 1 — Décomposition bas-rang : comment LoRA modifie les poids", expanded=True):
    _stc.html("""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
body{margin:0;padding:0;background:transparent;font-family:'Segoe UI',sans-serif;}
.lora-wrap{background:#0a0e1a;border-radius:14px;padding:28px 28px;display:flex;flex-direction:column;gap:20px;width:100%;box-sizing:border-box;}
.lora-title{color:#c7d2fe;font-size:14px;font-weight:700;letter-spacing:.5px;text-align:center;margin-bottom:4px;}
.lora-row{display:flex;align-items:stretch;justify-content:center;gap:20px;width:100%;}
.lora-col{display:flex;flex-direction:column;align-items:center;gap:8px;flex:1;}
.lora-box{border-radius:10px;padding:12px 18px;text-align:center;font-size:13px;font-weight:600;line-height:1.6;width:100%;box-sizing:border-box;}
.b-frozen{background:#1e1040;color:#a78bfa;border:2px solid #6d28d9;}
.b-trainA{background:#052e16;color:#6ee7b7;border:2px solid #16a34a;}
.b-trainB{background:#0c2d45;color:#7dd3fc;border:2px solid #0369a1;}
.b-sum{background:#1e1b4b;color:#e0e7ff;border:2px solid #4f46e5;font-size:14px;}
.b-input{background:#1c1917;color:#d6d3d1;border:1px solid #78716c;}
.b-output{background:#1c1917;color:#fde68a;border:1px solid #b45309;}
.lora-arrow{color:#475569;font-size:22px;line-height:1;}
.lora-arrow-h{color:#475569;font-size:18px;}
.lora-badge{display:inline-block;padding:2px 7px;border-radius:4px;font-size:10px;font-weight:700;margin:2px;}
.bg{background:#312e81;color:#c7d2fe;}
.bg2{background:#14532d;color:#86efac;}
.bg3{background:#1e3a5f;color:#7dd3fc;}
.bg4{background:#7f1d1d;color:#fca5a5;}
.lora-eq{background:#0f172a;border-radius:10px;padding:14px 20px;text-align:center;
  font-family:'Courier New',monospace;font-size:15px;color:#e2e8f0;border:1px solid #334155;
  letter-spacing:.5px;}
.lora-eq .freeze{color:#a78bfa;}
.lora-eq .train{color:#6ee7b7;}
.lora-eq .op{color:#94a3b8;}
.lora-label{color:#64748b;font-size:10px;text-align:center;margin:2px 0;}
.lora-sep{border:none;border-top:1px solid #1e293b;margin:8px 0;}
.lora-note{background:#172030;border-left:3px solid #3b82f6;padding:8px 14px;
  border-radius:0 8px 8px 0;color:#93c5fd;font-size:11px;margin-top:4px;}
</style>
<div class="lora-wrap">

  <div class="lora-title">Principe fondamental — ΔW = A · B (poids figés + adaptateurs entraînables)</div>

  <!-- Full fine-tuning vs LoRA -->
  <div class="lora-row">

    <!-- Full FT -->
    <div class="lora-col">
      <div style="color:#f87171;font-size:12px;font-weight:700;letter-spacing:.5px;">FULL FINE-TUNING</div>
      <div class="lora-box" style="background:#2a0808;color:#fca5a5;border:2px solid #b91c1c;">
        W <span class="lora-badge bg4">7B params</span><br>
        <span style="font-size:11px;opacity:.8;">Toute la matrice mise à jour<br>gradient → chaque paramètre</span>
      </div>
      <div style="background:#1a0606;border-radius:8px;padding:10px 16px;text-align:center;
        color:#fca5a5;font-size:12px;border:1px dashed #991b1b;width:100%;box-sizing:border-box;">
        VRAM entraînement : <strong>~80 GB</strong> (BF16)<br>
        Checkpoint : <strong>~14 GB</strong> · Params : <strong>100%</strong>
      </div>
    </div>

    <!-- VS -->
    <div style="color:#64748b;font-size:30px;font-weight:200;padding:0 12px;align-self:center;">vs</div>

    <!-- LoRA -->
    <div class="lora-col">
      <div style="color:#4ade80;font-size:12px;font-weight:700;letter-spacing:.5px;">LORA FINE-TUNING</div>
      <div class="lora-box b-frozen">
        W <span class="lora-badge bg">figé · 7B params</span><br>
        <span style="font-size:11px;opacity:.8;">Aucune mise à jour — gradient off</span>
      </div>
      <div style="color:#4ade80;font-size:18px;align-self:center;">+</div>
      <div style="display:flex;gap:8px;align-items:stretch;width:100%;">
        <div class="lora-box b-trainA" style="flex:1;">
          <strong>A</strong><br><span class="lora-badge bg2">d × r</span><br>
          <span style="font-size:10px;">Init Kaiming</span>
        </div>
        <div style="color:#6ee7b7;font-size:20px;align-self:center;padding:0 4px;">×</div>
        <div class="lora-box b-trainB" style="flex:1;">
          <strong>B</strong><br><span class="lora-badge bg3">r × k</span><br>
          <span style="font-size:10px;">Init Zéros</span>
        </div>
      </div>
      <div class="lora-label" style="font-size:11px;">rang r = 8–64 ≪ d, k — contrôle capacité d'adaptation</div>
      <div style="background:#071a10;border-radius:8px;padding:10px 16px;text-align:center;
        color:#6ee7b7;font-size:12px;border:1px dashed #166534;width:100%;box-sizing:border-box;">
        VRAM entraînement : <strong>~18–20 GB</strong> (QLoRA 4-bit)<br>
        Adapter : <strong>~10–50 MB</strong> · Params : <strong>~0.3%</strong>
      </div>
    </div>
  </div>

  <hr class="lora-sep">

  <!-- Équation -->
  <div class="lora-eq">
    output = <span class="freeze">W</span> · x &nbsp;<span class="op">+</span>&nbsp; <span class="train">(A · B)</span> · x &nbsp;<span class="op">·</span>&nbsp; <span style="color:#fde68a;">(α / r)</span>
  </div>
  <div style="display:flex;justify-content:center;gap:24px;flex-wrap:wrap;margin-top:4px;">
    <span style="font-size:11px;color:#a78bfa;">W : poids figés (base model)</span>
    <span style="font-size:11px;color:#6ee7b7;">A·B : mise à jour entraînable</span>
    <span style="font-size:11px;color:#fde68a;">α/r : facteur d'échelle (lora_alpha)</span>
  </div>

  <hr class="lora-sep">

  <!-- Merge -->
  <div style="display:flex;align-items:center;justify-content:center;gap:10px;flex-wrap:wrap;">
    <div class="lora-box b-frozen" style="min-width:120px;">W (figé)</div>
    <div class="lora-arrow-h">+</div>
    <div class="lora-box b-trainA" style="min-width:80px;">A · B</div>
    <div class="lora-arrow-h" style="color:#fde68a;">──merge_and_unload()──►</div>
    <div class="lora-box" style="background:#1e3a5f;color:#bae6fd;border:2px solid #0ea5e9;min-width:140px;">
      W_merged<br>
      <span style="font-size:10px;">Modèle standalone</span><br>
      <span class="lora-badge bg3">0 latence inférence</span>
    </div>
  </div>
  <div class="lora-note">
    ✓ Après merge : modèle identique à un fully fine-tuned — aucun overhead d'inférence.
    L'adapter (A, B) peut être stocké séparément et re-appliqué à volonté sur n'importe quelle copie du modèle de base.
  </div>

</div>
</body></html>""", height=780, scrolling=False)

with st.expander("📊 Schéma 2 — Quand utiliser LoRA vs Full Fine-tuning ?", expanded=True):
    _stc.html("""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
body{margin:0;padding:4px;background:transparent;font-family:'Segoe UI',sans-serif;}
.wrap{display:flex;flex-direction:column;gap:18px;width:100%;box-sizing:border-box;}

/* Comparison table */
table{width:100%;border-collapse:collapse;border-radius:12px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,.4);}
thead tr{background:#1e1b4b;}
th{color:#c7d2fe;padding:11px 16px;font-size:12px;letter-spacing:.8px;text-transform:uppercase;font-weight:700;}
td{padding:11px 16px;font-size:13px;border-bottom:1px solid rgba(255,255,255,.04);vertical-align:middle;}
.c-lora{background:linear-gradient(90deg,#052e16,#0a3d20);color:#bbf7d0;}
.c-full{background:linear-gradient(90deg,#300a0a,#3d1010);color:#fca5a5;}
.c-both{background:linear-gradient(90deg,#1c1204,#261a06);color:#fde68a;}
.yes{color:#4ade80;font-weight:800;font-size:15px;}
.no{color:#f87171;font-weight:700;font-size:15px;}
.dep{color:#fbbf24;font-weight:700;font-size:13px;}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700;margin:1px;}
.bl{background:#1e3a5f;color:#7dd3fc;}
.bg{background:#14532d;color:#86efac;}
.br{background:#7f1d1d;color:#fca5a5;}
.ba{background:#78350f;color:#fde68a;}

/* Flow diagram */
.flow-wrap{background:#0a0e1a;border-radius:12px;padding:20px 24px;display:flex;flex-direction:column;align-items:center;gap:6px;width:100%;box-sizing:border-box;}
.flow-q{background:#1e3a5f;color:#bae6fd;border:2px solid #0ea5e9;border-radius:10px;padding:12px 24px;
  text-align:center;font-size:13px;font-weight:600;width:100%;max-width:420px;box-sizing:border-box;}
.flow-branches{display:flex;gap:0;align-items:flex-start;width:100%;justify-content:space-around;}
.flow-branch{display:flex;flex-direction:column;align-items:center;gap:5px;flex:1;padding:0 12px;}
.flow-yes{color:#4ade80;font-size:12px;font-weight:700;}
.flow-no{color:#f87171;font-size:12px;font-weight:700;}
.flow-arrow{color:#475569;font-size:20px;}
.flow-subbranches{display:flex;gap:16px;width:100%;justify-content:center;}
.flow-subbranch{display:flex;flex-direction:column;align-items:center;gap:4px;flex:1;}
.flow-q-sub{background:#1e3a5f;color:#bae6fd;border:2px solid #0ea5e9;border-radius:10px;padding:10px 14px;
  text-align:center;font-size:12px;font-weight:600;width:100%;box-sizing:border-box;}
.flow-end-lora{background:#052e16;color:#86efac;border:2px solid #16a34a;border-radius:10px;
  padding:10px 14px;text-align:center;font-size:12px;font-weight:700;width:100%;box-sizing:border-box;}
.flow-end-full{background:#300a0a;color:#fca5a5;border:2px solid #b91c1c;border-radius:10px;
  padding:10px 14px;text-align:center;font-size:12px;font-weight:700;width:100%;box-sizing:border-box;}
.flow-end-prompt{background:#1c1204;color:#fde68a;border:2px solid #b45309;border-radius:10px;
  padding:10px 14px;text-align:center;font-size:12px;font-weight:700;width:100%;box-sizing:border-box;}
.flow-divider{width:1px;background:#1e293b;align-self:stretch;margin:0 8px;}
</style>
<div class="wrap">

<!-- Decision flow -->
<div class="flow-wrap">
  <div style="color:#64748b;font-size:11px;font-weight:700;letter-spacing:.8px;margin-bottom:6px;">ARBRE DE DÉCISION — LoRA OU FULL FINE-TUNING ?</div>

  <div class="flow-q">VRAM disponible pour l'entraînement ?</div>
  <div class="flow-arrow">↓</div>

  <div class="flow-branches">

    <!-- Branch gauche : ≥ 80 GB -->
    <div class="flow-branch">
      <span class="flow-yes">≥ 80 GB</span>
      <div class="flow-arrow">↓</div>
      <div class="flow-q-sub">Tâche critique / prod ?</div>
      <div class="flow-subbranches">
        <div class="flow-subbranch">
          <span class="flow-yes">Oui</span>
          <div class="flow-arrow">↓</div>
          <div class="flow-end-full">Full Fine-tuning</div>
        </div>
        <div class="flow-subbranch">
          <span class="flow-no">Non</span>
          <div class="flow-arrow">↓</div>
          <div class="flow-end-lora">LoRA rang élevé (r=64)</div>
        </div>
      </div>
    </div>

    <div class="flow-divider"></div>

    <!-- Branch milieu : 24–80 GB -->
    <div class="flow-branch">
      <span style="color:#fbbf24;font-size:12px;font-weight:700;">24–80 GB</span>
      <div class="flow-arrow">↓</div>
      <div class="flow-q-sub">Données disponibles ?</div>
      <div class="flow-subbranches">
        <div class="flow-subbranch">
          <span class="flow-yes">&gt; 500 ex.</span>
          <div class="flow-arrow">↓</div>
          <div class="flow-end-lora">LoRA BF16 (r=16–32)</div>
        </div>
        <div class="flow-subbranch">
          <span class="flow-no">200–500 ex.</span>
          <div class="flow-arrow">↓</div>
          <div class="flow-end-lora">LoRA r=8 + early stop</div>
        </div>
      </div>
    </div>

    <div class="flow-divider"></div>

    <!-- Branch droite : < 24 GB -->
    <div class="flow-branch">
      <span class="flow-no">&lt; 24 GB</span>
      <div class="flow-arrow">↓</div>
      <div class="flow-q-sub">Données disponibles ?</div>
      <div class="flow-subbranches">
        <div class="flow-subbranch">
          <span class="flow-yes">&gt; 200 ex.</span>
          <div class="flow-arrow">↓</div>
          <div class="flow-end-lora">QLoRA 4-bit (r=8–16)</div>
        </div>
        <div class="flow-subbranch">
          <span class="flow-no">&lt; 200 ex.</span>
          <div class="flow-arrow">↓</div>
          <div class="flow-end-prompt">Prompt engineering d'abord</div>
        </div>
      </div>
    </div>

  </div>
</div>

<!-- Comparison table -->
<table>
<thead><tr><th>Critère</th><th>LoRA / QLoRA</th><th>Full Fine-tuning</th></tr></thead>
<tbody>
<tr class="c-lora"><td>VRAM entraînement (7B)</td><td><span class="badge bg">18–20 GB</span> QLoRA 4-bit · <span class="badge bl">28 GB</span> LoRA BF16</td><td><span class="badge br">~80 GB</span> BF16 complet</td></tr>
<tr class="c-lora"><td>Taille checkpoint</td><td><span class="badge bg">10–50 MB</span> adapter seul (sans base)</td><td><span class="badge br">~14 GB</span> modèle complet</td></tr>
<tr class="c-lora"><td>Params entraînables</td><td><span class="badge bg">~0.1–1%</span> du modèle total</td><td><span class="badge br">100%</span></td></tr>
<tr class="c-both"><td>Qualité vs base</td><td>Comparable — F1 ±2% sur tâches de domaine</td><td>Baseline de référence</td></tr>
<tr class="c-lora"><td>Multi-tâche simultané</td><td><span class="yes">✓</span> N adapters légers sur 1 base figée</td><td><span class="no">✗</span> N modèles complets (×14 GB chacun)</td></tr>
<tr class="c-lora"><td>Latence inférence</td><td><span class="yes">0</span> après <code>merge_and_unload()</code></td><td><span class="yes">0</span></td></tr>
<tr class="c-full"><td>Tâches très différentes du pré-entraînement</td><td><span class="dep">⚠</span> rang élevé requis (r≥64)</td><td><span class="yes">✓</span> optimal</td></tr>
<tr class="c-lora"><td>Cas d'usage typique</td><td>Adaptation domaine, format propriétaire, ressources limitées, multi-tâche</td><td>Changement radical de comportement, sécurité critique, budget illimité</td></tr>
</tbody>
</table>
</div>
</body></html>""", height=720, scrolling=False)

with st.expander("⚙️ Schéma 3 — Workflow complet LoRA → GGUF (pour ce pipeline)", expanded=False):
    _stc.html("""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
body{margin:0;padding:2px;background:transparent;font-family:'Segoe UI',sans-serif;}
.wrap{background:#0a0e1a;border-radius:14px;padding:22px 28px;display:flex;flex-direction:column;align-items:center;gap:0;width:100%;box-sizing:border-box;}
.step{display:flex;align-items:flex-start;gap:16px;width:100%;padding:10px 0;}
.step-num{background:#4f46e5;color:#fff;border-radius:50%;width:32px;height:32px;min-width:32px;
  display:flex;align-items:center;justify-content:center;font-weight:800;font-size:14px;}
.step-body{flex:1;}
.step-title{color:#e2e8f0;font-size:13px;font-weight:700;margin-bottom:3px;}
.step-desc{color:#94a3b8;font-size:11px;line-height:1.6;}
.step-code{background:#0f172a;border-radius:6px;padding:6px 10px;font-family:'Courier New',monospace;
  font-size:10px;color:#7dd3fc;margin-top:4px;border:1px solid #1e293b;}
.step-arrow{color:#334155;font-size:18px;text-align:center;padding:2px 0 2px 22px;}
.badge{display:inline-block;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:700;margin:1px;}
.bg{background:#14532d;color:#86efac;}
.bl{background:#1e3a5f;color:#7dd3fc;}
.bp{background:#4c1d95;color:#ddd6fe;}
.ba{background:#78350f;color:#fde68a;}
</style>
<div class="wrap">

<div style="color:#6366f1;font-size:11px;font-weight:700;letter-spacing:.8px;margin-bottom:12px;">
  WORKFLOW : BASE MODEL → LORA → MERGE → GGUF → REMPLACEMENT dans models/</div>

<div class="step">
  <div class="step-num">1</div>
  <div class="step-body">
    <div class="step-title">Charger le modèle de base (HuggingFace)</div>
    <div class="step-desc">Modèle non quantifié pour l'entraînement LoRA — Qwen2.5-VL-7B-Instruct en BF16</div>
    <div class="step-code">model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16)</div>
  </div>
</div>
<div class="step-arrow">↓</div>

<div class="step">
  <div class="step-num">2</div>
  <div class="step-body">
    <div class="step-title">Configurer et appliquer LoRA <span class="badge bp">PEFT</span></div>
    <div class="step-desc">Cibler les projections attention. Rang r=16 pour documents, alpha=32, dropout=0.05.</div>
    <div class="step-code">config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj","k_proj","o_proj"])
model = get_peft_model(model, config)  # ~0.3% params entraînables</div>
  </div>
</div>
<div class="step-arrow">↓</div>

<div class="step">
  <div class="step-num">3</div>
  <div class="step-body">
    <div class="step-title">Entraîner sur dataset document <span class="badge ba">SFTTrainer</span></div>
    <div class="step-desc">Format ChatML avec paires image→JSON. Min 200–500 exemples par type de document.</div>
    <div class="step-code">trainer = SFTTrainer(model, train_dataset=ds, args=TrainingArguments(output_dir="lora_out", ...))</div>
  </div>
</div>
<div class="step-arrow">↓</div>

<div class="step">
  <div class="step-num">4</div>
  <div class="step-body">
    <div class="step-title">Merger et sauvegarder <span class="badge bg">merge_and_unload()</span></div>
    <div class="step-desc">Fusionner l'adapter dans les poids — modèle standalone sans overhead.</div>
    <div class="step-code">merged = model.merge_and_unload()
merged.save_pretrained("./qwen_finetuned_hf")</div>
  </div>
</div>
<div class="step-arrow">↓</div>

<div class="step">
  <div class="step-num">5</div>
  <div class="step-body">
    <div class="step-title">Convertir en GGUF + quantifier <span class="badge bl">llama.cpp</span></div>
    <div class="step-desc">Convertir le HF checkpoint en GGUF Q4_K_M pour llama-server.</div>
    <div class="step-code">python llama.cpp/convert_hf_to_gguf.py ./qwen_finetuned_hf --outtype q4_k_m --outfile models/qwen_ft.gguf</div>
  </div>
</div>
<div class="step-arrow">↓</div>

<div class="step">
  <div class="step-num">6</div>
  <div class="step-body">
    <div class="step-title">Remplacer le modèle dans models/ <span class="badge bg">Production</span></div>
    <div class="step-desc">Copier le nouveau .gguf dans models/ — llama-server le détecte automatiquement au prochain démarrage.</div>
    <div class="step-code">cp models/qwen_ft.gguf models/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf
# Puis : pwsh scripts/start_server.ps1</div>
  </div>
</div>

</div>
</body></html>""", height=530, scrolling=False)

    st.info(
        "**Résumé LoRA en une phrase :** au lieu de mettre à jour les 7 milliards de paramètres, "
        "on entraîne seulement deux petites matrices A (d×r) et B (r×k) par couche — "
        "le rang `r` (typiquement 8–64) contrôle la capacité d'adaptation. "
        "Résultat : ~0.3% de paramètres entraînables, VRAM divisée par 4, qualité comparable."
    )

st.divider()
st.caption("Guide généré pour le POC Document Intelligence — Qwen2.5-VL · Mise à jour : Mars 2026")

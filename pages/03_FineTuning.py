"""
Fine-tuning Qwen2.5-VL-3B avec LoRA r=4 sur Intel Arc (IPEX XPU)
Workflow guidé 6 étapes — objectif 30–45 min sur Core Ultra 5 125H
"""
import streamlit as st
import streamlit.components.v1 as _stc
from pathlib import Path
import json, subprocess, sys, time, os, shutil, zipfile, tempfile
from datetime import datetime

ROOT       = Path(__file__).parent.parent
LOGS_DIR   = ROOT / "logs"
FT_DIR     = ROOT / "finetuning"
MODELS_DIR = ROOT / "models"
SERVED_DIR = ROOT / "served_models"

for d in [LOGS_DIR, FT_DIR, MODELS_DIR, SERVED_DIR]:
    d.mkdir(exist_ok=True)

st.set_page_config(
    page_title="Fine-tuning — Qwen2.5-VL-3B · XPU",
    page_icon="🎯",
    layout="wide",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.section-header {
    background: linear-gradient(90deg, #0f766e 0%, #0369a1 100%);
    color: white; padding: 10px 18px; border-radius: 8px;
    font-size: 18px; font-weight: 700; margin: 24px 0 12px 0;
}
.step-badge {
    display:inline-flex; align-items:center; justify-content:center;
    width:28px; height:28px; border-radius:50%;
    font-weight:800; font-size:13px; margin-right:8px;
}
.step-done  { background:#16a34a; color:#fff; }
.step-active{ background:#0ea5e9; color:#fff; }
.step-todo  { background:#334155; color:#94a3b8; }
.callout-info { background:#eff6ff; border-left:4px solid #3b82f6;
    padding:12px 16px; border-radius:0 8px 8px 0; margin:8px 0; }
.callout-warn { background:#fffbeb; border-left:4px solid #f59e0b;
    padding:12px 16px; border-radius:0 8px 8px 0; margin:8px 0; }
.callout-ok   { background:#f0fdf4; border-left:4px solid #22c55e;
    padding:12px 16px; border-radius:0 8px 8px 0; margin:8px 0; }
.metric-card {
    background:#f8fafc; border:1px solid #e2e8f0;
    border-radius:10px; padding:14px 18px; text-align:center;
}
.metric-val { font-size:26px; font-weight:800; color:#0f766e; }
.metric-lab { font-size:12px; color:#64748b; margin-top:2px; }
</style>
""", unsafe_allow_html=True)

# ─── Session state ─────────────────────────────────────────────────────────────
def _init():
    defaults = {
        "ft_step": 0,
        "ft_dataset_path": None,
        "ft_n_train": 0,
        "ft_n_val": 0,
        "ft_run_id": None,
        "ft_process_pid": None,
        "ft_config": {},
        "ft_metrics": [],
        "ft_adapter_path": None,
        "ft_gguf_path": None,
        "ft_doc_type": "chart",
        "ft_labels": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()
S = st.session_state

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("# 🎯 Fine-tuning — Qwen2.5-VL-3B · Intel Arc XPU")

# ─── Introduction justificative ───────────────────────────────────────────────
with st.expander("📌 Pourquoi ce choix dans notre contexte ?", expanded=(S.ft_step == 0)):
    col_txt, col_diag = st.columns([1, 1])

    with col_txt:
        st.markdown("""
**Contexte machine :** Intel Core Ultra 5 125H · 32 GB DDR5 · Intel Arc iGPU (Xe-HPG · ~17.5 GB partagé)

**Problème :** `bitsandbytes` (QLoRA classique) est CUDA-only → Intel Arc exclu.

**Solution retenue : LoRA FP16 + IPEX XPU**
- `intel_extension_for_pytorch` expose le backend `xpu` pour Intel Arc
- Pas de quantification 4-bit → LoRA FP16 standard sur XPU
- Qwen2.5-VL-**3B** au lieu du 7B → moitié moins de paramètres, RAM ×2 disponible
- `r=4` → ~0.05% des paramètres entraînables seulement

**Pourquoi 80 exemples suffisent ici ?**
Le modèle 3B est pré-entraîné sur des millions de documents. On n'apprend pas à lire —
on affine uniquement le **format de sortie JSON** et le **vocabulaire métier**.
80 exemples propres calibrent ce biais en 3 epochs.

**Estimation temps :**
- 80 ex. × 3 epochs = 240 steps
- ~7–8s/step sur Arc XPU
- **Total : ~30–40 min**
""")

    with col_diag:
        _stc.html("""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
body{margin:0;padding:0;background:transparent;font-family:'Segoe UI',sans-serif;}
.w{background:#0a0e1a;border-radius:12px;padding:20px;display:flex;flex-direction:column;gap:6px;}
.row{display:flex;align-items:center;gap:10px;}
.box{border-radius:8px;padding:8px 14px;font-size:12px;font-weight:600;flex:1;text-align:center;line-height:1.5;}
.b1{background:#1e1040;color:#a78bfa;border:2px solid #6d28d9;}
.b2{background:#052e16;color:#6ee7b7;border:2px solid #16a34a;}
.b3{background:#0c2d45;color:#7dd3fc;border:2px solid #0369a1;}
.b4{background:#1a0520;color:#e879f9;border:1px solid #a21caf;}
.arr{color:#475569;font-size:20px;text-align:center;}
.badge{display:inline-block;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:700;margin:1px;}
.bg{background:#14532d;color:#86efac;}
.bp{background:#4c1d95;color:#ddd6fe;}
.br{background:#7f1d1d;color:#fca5a5;}
</style>
<div class="w">
  <div class="row">
    <div class="box b1">🧠 Qwen2.5-VL-<strong>3B</strong><br><span class="badge bp">HF FP16 ~6 GB</span></div>
  </div>
  <div class="arr">↓</div>
  <div class="row">
    <div class="box b2">⚡ IPEX XPU<br><span class="badge bg">Intel Arc iGPU</span> · FP16</div>
  </div>
  <div class="arr">↓</div>
  <div class="row">
    <div class="box b3">🔧 LoRA r=4<br>q_proj · v_proj · <span class="badge bp">~0.05% params</span></div>
  </div>
  <div class="arr">↓</div>
  <div class="row" style="gap:6px;">
    <div class="box b4" style="flex:.5;">80 ex.<br><span style="font-size:10px;">3 epochs</span></div>
    <div style="color:#475569;font-size:16px;">×</div>
    <div class="box b4" style="flex:.5;">~8s/step<br><span style="font-size:10px;">Arc XPU</span></div>
    <div style="color:#4ade80;font-size:16px;">=</div>
    <div class="box" style="background:#052e16;color:#6ee7b7;border:2px solid #16a34a;flex:.8;">
      <strong>~35 min</strong><br><span style="font-size:10px;">240 steps</span>
    </div>
  </div>
  <div class="arr">↓</div>
  <div class="row">
    <div class="box b1">merge_and_unload() → GGUF Q4_K_M<br><span class="badge bg">~3.1 GB · prêt pour llama-server</span></div>
  </div>
</div>
</body></html>""", height=370)

# ─── Documentation Infrastructure ────────────────────────────────────────────
with st.expander("🏗️ Infrastructure & Stack — mise en place complète", expanded=False):
    _stc.html("""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
body{margin:0;padding:0;font-family:'Segoe UI',sans-serif;font-size:13px;background:transparent;}
.infra{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:4px;}
.block{background:#0f172a;border-radius:10px;padding:14px 16px;border:1px solid #1e293b;}
.block h3{color:#7dd3fc;font-size:13px;margin:0 0 10px 0;letter-spacing:.5px;text-transform:uppercase;}
.row{display:flex;align-items:flex-start;gap:8px;margin-bottom:8px;}
.icon{font-size:16px;flex-shrink:0;margin-top:1px;}
.col{display:flex;flex-direction:column;}
.label{color:#94a3b8;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.4px;}
.val{color:#e2e8f0;font-size:12px;margin-top:1px;line-height:1.5;}
.tag{display:inline-block;padding:1px 7px;border-radius:4px;font-size:10px;font-weight:700;margin:1px 2px 0 0;}
.tok  {background:#14532d;color:#86efac;}
.tblu {background:#0c2d45;color:#7dd3fc;}
.tora {background:#431407;color:#fdba74;}
.tpur {background:#2e1065;color:#c4b5fd;}
.tgry {background:#1e293b;color:#94a3b8;}
.sep{height:1px;background:#1e293b;margin:6px 0;}
.full{grid-column:1/-1;}
.arrow{color:#0ea5e9;font-size:11px;font-weight:700;margin-top:3px;}
.flow{display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin-top:6px;}
.fbox{background:#1e293b;border-radius:6px;padding:4px 10px;color:#e2e8f0;font-size:11px;font-weight:600;}
.farr{color:#475569;font-size:14px;}
</style>
<div class="infra">

  <!-- Machine -->
  <div class="block">
    <h3>💻 Machine</h3>
    <div class="row"><span class="icon">🖥️</span><div class="col">
      <span class="label">CPU</span>
      <span class="val">Intel Core Ultra 5 125H · 12C/16T · 4.5 GHz</span>
    </div></div>
    <div class="row"><span class="icon">⚡</span><div class="col">
      <span class="label">GPU</span>
      <span class="val">Intel Arc iGPU (Xe-HPG) · ~8 TFLOPS FP16<br>
        <span class="tag tblu">Partagé DDR5</span>
        <span class="tag tok">17.5 GB alloués</span>
      </span>
    </div></div>
    <div class="row"><span class="icon">🧱</span><div class="col">
      <span class="label">RAM</span>
      <span class="val">32 GB DDR5 — partagé CPU + iGPU</span>
    </div></div>
    <div class="row"><span class="icon">💾</span><div class="col">
      <span class="label">OS</span>
      <span class="val">Windows 11 Home · Python 3.12.3</span>
    </div></div>
  </div>

  <!-- Stack logiciel -->
  <div class="block">
    <h3>📦 Stack logiciel</h3>
    <div class="row"><span class="icon">🔥</span><div class="col">
      <span class="label">PyTorch</span>
      <span class="val">2.8.0+cpu <span class="tag tgry">compatible IPEX XPU</span></span>
    </div></div>
    <div class="row"><span class="icon">⚙️</span><div class="col">
      <span class="label">IPEX</span>
      <span class="val">intel-extension-for-pytorch 2.8.10+xpu<br>
        <span class="tag tora">Requiert Intel oneAPI Base Toolkit</span>
      </span>
    </div></div>
    <div class="row"><span class="icon">🤗</span><div class="col">
      <span class="label">Transformers / PEFT / TRL</span>
      <span class="val">5.2.0 / 0.18.1 / 0.29.0</span>
    </div></div>
    <div class="row"><span class="icon">📊</span><div class="col">
      <span class="label">MLflow</span>
      <span class="val">Expérience : <code style="color:#7dd3fc;background:#0c1a2e;padding:1px 4px;border-radius:3px;">qwen_vl_finetuning</code></span>
    </div></div>
  </div>

  <!-- Modèles -->
  <div class="block">
    <h3>🧠 Modèles</h3>
    <div class="row"><span class="icon">🚀</span><div class="col">
      <span class="label">Production (inférence llama.cpp)</span>
      <span class="val">Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf <span class="tag tgry">~4.7 GB</span></span>
    </div></div>
    <div class="row"><span class="icon">🎯</span><div class="col">
      <span class="label">Fine-tuning (HF FP16)</span>
      <span class="val">Qwen2.5-VL-3B-Instruct <span class="tag tok">7.1 GB · 2 safetensors</span><br>
        <code style="color:#94a3b8;font-size:11px;">finetuning/base/Qwen2.5-VL-3B-Instruct/</code>
      </span>
    </div></div>
    <div class="sep"></div>
    <div class="row"><span class="icon">📁</span><div class="col">
      <span class="label">Sorties entraînement</span>
      <span class="val">
        <span class="tag tpur">adapter/</span> LoRA weights PEFT<br>
        <span class="tag tblu">merged/</span> HF FP16 standalone<br>
        <span class="tag tok">*.gguf</span> Q4_K_M → llama-server
      </span>
    </div></div>
  </div>

  <!-- Prompt Registry -->
  <div class="block">
    <h3>📋 Prompt & Schema Registry</h3>
    <div class="row"><span class="icon">🗂️</span><div class="col">
      <span class="label">Fichier</span>
      <span class="val"><code style="color:#7dd3fc;background:#0c1a2e;padding:1px 4px;border-radius:3px;">prompts/registry.json</code></span>
    </div></div>
    <div class="row"><span class="icon">🔒</span><div class="col">
      <span class="label">Versioning</span>
      <span class="val">MAJOR.MINOR.PATCH · immutable · SHA-256</span>
    </div></div>
    <div class="row"><span class="icon">📌</span><div class="col">
      <span class="label">Prompts</span>
      <span class="val">classify · extract.invoice · extract.bank_statement · extract.contract · extract.default</span>
    </div></div>
    <div class="row"><span class="icon">✅</span><div class="col">
      <span class="label">Schemas + eval_fields</span>
      <span class="val">Champs F1 par type doc · réutilisés train &amp; eval</span>
    </div></div>
  </div>

  <!-- oneAPI Setup + REX -->
  <div class="block full">
    <h3>⚙️ Intel Arc XPU — Configuration complète &amp; Retour d'expérience</h3>

    <!-- Flux final validé -->
    <div style="color:#86efac;font-size:11px;font-weight:700;margin-bottom:8px;">
      ✅ Configuration validée — torch.xpu.is_available() → True · Device: Intel(R) Arc(TM) Graphics
    </div>
    <div class="flow" style="margin-bottom:10px;">
      <div class="fbox" style="border:1px solid #16a34a;">1. Arc GPU driver<br><span style="color:#86efac;font-size:10px;">Préinstallé Windows 11</span></div><div class="farr">→</div>
      <div class="fbox">2. oneAPI Base Toolkit 2025.1<br><span style="color:#f59e0b;font-size:10px;">winget install Intel.OneAPI.BaseToolkit</span></div><div class="farr">→</div>
      <div class="fbox">3. IPEX 2.8.10+xpu<br><span style="color:#7dd3fc;font-size:10px;">pip install intel-extension-for-pytorch<br>--extra-index-url pytorch-extension.intel.com/.../xpu/</span></div><div class="farr">→</div>
      <div class="fbox" style="border:1px solid #16a34a;">4. torch 2.8.0+xpu<br><span style="color:#86efac;font-size:10px;">pip install torch==2.8.0+xpu<br>--index-url pytorch.org/whl/xpu</span></div><div class="farr">→</div>
      <div class="fbox">5. PATH oneAPI bin/<br><span style="color:#94a3b8;font-size:10px;">start_with_xpu.bat</span></div><div class="farr">→</div>
      <div class="fbox" style="border:2px solid #16a34a;background:#052e16;">XPU: True<br><span style="color:#86efac;font-size:10px;">~35 min fine-tuning</span></div>
    </div>

    <div class="sep"></div>

    <!-- REX : erreurs rencontrées -->
    <div style="color:#7dd3fc;font-size:11px;font-weight:700;margin:8px 0 6px 0;">📋 Retour d'expérience — erreurs rencontrées dans l'ordre</div>
    <table style="width:100%;border-collapse:collapse;font-size:11px;">
      <tr style="color:#64748b;border-bottom:1px solid #1e293b;">
        <th style="text-align:left;padding:4px 8px;">#</th>
        <th style="text-align:left;padding:4px 8px;">Erreur</th>
        <th style="text-align:left;padding:4px 8px;">Cause</th>
        <th style="text-align:left;padding:4px 8px;">Fix appliqué</th>
      </tr>
      <tr style="border-bottom:1px solid #0f172a;">
        <td style="padding:5px 8px;color:#94a3b8;">1</td>
        <td style="padding:5px 8px;color:#fca5a5;font-family:monospace;">OSError WinError 126<br>esimd_kernels.dll</td>
        <td style="padding:5px 8px;color:#e2e8f0;">IPEX installé mais oneAPI Base Toolkit absent → DLL SYCL manquantes</td>
        <td style="padding:5px 8px;color:#86efac;">winget install Intel.OneAPI.BaseToolkit 2025.1.3.8 (UAC requis)</td>
      </tr>
      <tr style="border-bottom:1px solid #0f172a;">
        <td style="padding:5px 8px;color:#94a3b8;">2</td>
        <td style="padding:5px 8px;color:#fca5a5;font-family:monospace;">OSError WinError 126<br>intel-ext-pt-gpu-bitsandbytes.dll</td>
        <td style="padding:5px 8px;color:#e2e8f0;">torch 2.8.0+cpu installé → <code style="color:#fca5a5;">c10_xpu.dll</code> absent (CPU build = pas de XPU runtime)</td>
        <td style="padding:5px 8px;color:#86efac;">pip install torch==2.8.0+xpu --index-url pytorch.org/whl/xpu</td>
      </tr>
      <tr style="border-bottom:1px solid #0f172a;">
        <td style="padding:5px 8px;color:#94a3b8;">3</td>
        <td style="padding:5px 8px;color:#fca5a5;font-family:monospace;">OSError encore après oneAPI installé</td>
        <td style="padding:5px 8px;color:#e2e8f0;">Les DLL oneAPI (sycl8.dll, svml…) ne sont pas dans le PATH du terminal — oneAPI n'ajoute pas le PATH automatiquement</td>
        <td style="padding:5px 8px;color:#86efac;">Ajout permanent via <code>SetEnvironmentVariable('PATH', User)</code> + start_with_xpu.bat</td>
      </tr>
      <tr>
        <td style="padding:5px 8px;color:#94a3b8;">4</td>
        <td style="padding:5px 8px;color:#fca5a5;font-family:monospace;">winget: installeur introuvable (WinGet cache)</td>
        <td style="padding:5px 8px;color:#e2e8f0;">WinGet avait nettoyé le cache .exe avant que l'installeur le retrouve</td>
        <td style="padding:5px 8px;color:#86efac;">winget relancé via PowerShell Start-Process -Verb RunAs (UAC) → installation confirmée</td>
      </tr>
    </table>

    <div class="sep"></div>
    <div style="display:flex;flex-wrap:wrap;gap:16px;font-size:11px;color:#64748b;margin-top:4px;">
      <span>🚀 <strong style="color:#7dd3fc;">Lancer Streamlit avec XPU</strong> : double-clic sur <code style="color:#e2e8f0;">start_with_xpu.bat</code></span>
      <span>⚡ <strong style="color:#86efac;">Résultat</strong> : torch.xpu.is_available() → True · Intel Arc Graphics détecté · ~35 min fine-tuning</span>
      <span>🔄 <strong style="color:#c4b5fd;">Fallback</strong> : sans le .bat → CPU automatique (~2–3h)</span>
    </div>
  </div>

  <!-- Pipeline finetune_xpu.py -->
  <div class="block full">
    <h3>🔄 Flux d'entraînement — finetune_xpu.py</h3>
    <div class="flow">
      <div class="fbox">registry.json<br><span style="color:#94a3b8;font-size:10px;">prompts + schemas</span></div><div class="farr">→</div>
      <div class="fbox">DocumentDataset<br><span style="color:#94a3b8;font-size:10px;">×2 tâches/image</span></div><div class="farr">→</div>
      <div class="fbox">LoRA r=4<br><span style="color:#94a3b8;font-size:10px;">q_proj · v_proj</span></div><div class="farr">→</div>
      <div class="fbox">Trainer HF<br><span style="color:#94a3b8;font-size:10px;">IPEX XPU / CPU</span></div><div class="farr">→</div>
      <div class="fbox">MetricsCallback<br><span style="color:#94a3b8;font-size:10px;">MLflow + .jsonl</span></div><div class="farr">→</div>
      <div class="fbox">evaluate_model()<br><span style="color:#94a3b8;font-size:10px;">F1 eval_fields</span></div><div class="farr">→</div>
      <div class="fbox">merge_and_unload()<br><span style="color:#94a3b8;font-size:10px;">HF FP16</span></div><div class="farr">→</div>
      <div class="fbox" style="border:1px solid #16a34a;">GGUF Q4_K_M<br><span style="color:#86efac;font-size:10px;">served_models/</span></div>
    </div>
    <div class="sep"></div>
    <div style="display:flex;flex-wrap:wrap;gap:16px;margin-top:6px;font-size:11px;color:#64748b;">
      <span>📥 <strong style="color:#7dd3fc;">Inputs</strong> : images + labels.json (doc_type, expected fields)</span>
      <span>📤 <strong style="color:#86efac;">Outputs</strong> : adapter/ · merged/ · ft_*.jsonl · ft_*_eval.json · MLflow run</span>
      <span>🔁 <strong style="color:#c4b5fd;">Cohérence</strong> : mêmes prompts qu'en inférence (classify.py / extract.py)</span>
    </div>
  </div>

</div>
</body></html>""", height=1200)

# ─── Sidebar : Navigation + Monitoring ────────────────────────────────────────
STEPS = [
    ("🔍", "Environnement"),
    ("📂", "Dataset"),
    ("⚙️", "Config"),
    ("🚀", "Entraînement"),
    ("📊", "Évaluation"),
    ("🗂️", "Model Registry"),
    ("📦", "GGUF"),
    ("🗄️", "Servir"),
]

with st.sidebar:
    st.markdown("### 🎯 Fine-tuning — Étapes")
    for i, (icon, label) in enumerate(STEPS):
        if i < S.ft_step:
            badge = "✅"
            style = "color:#16a34a;font-weight:700;"
        elif i == S.ft_step:
            badge = "▶"
            style = "color:#0ea5e9;font-weight:700;"
        else:
            badge = str(i)
            style = "color:#64748b;"
        st.markdown(
            f'<div style="padding:4px 0;{style}">'
            f'{badge} {icon} {label}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Sous-phases entraînement (toujours visible si un run existe) ───────
    _sb_state = {}
    _sb_run_id = None
    # Priorité : run sélectionné en session, sinon le plus récent sur disque
    _session_run = getattr(S, "ft_run_id", None)
    if _session_run and (FT_DIR / _session_run / "training_state.json").exists():
        _sb_run_id = _session_run
    elif FT_DIR.exists():
        for _d in sorted(FT_DIR.iterdir(), reverse=True):
            if _d.is_dir() and (_d / "training_state.json").exists():
                _sb_run_id = _d.name
                break
    if _sb_run_id:
        try:
            _sb_state = json.loads((FT_DIR / _sb_run_id / "training_state.json").read_text(encoding="utf-8"))
        except Exception:
            pass
    if _sb_run_id:
        _cur_phase = _sb_state.get("phase", "training")
        _PHASES = [
            ("training",   "🔵", "Entraînement"),
            ("merging",    "🔀", "Merge"),
            ("testing",    "🟡", "Test"),
            ("validation", "🟠", "Validation"),
            ("done",       "✅", "Terminé"),
        ]
        _phase_order = [p[0] for p in _PHASES]
        _cur_idx = _phase_order.index(_cur_phase) if _cur_phase in _phase_order else 0
        st.markdown(
            f'<div style="margin-top:6px;font-size:11px;color:#94a3b8;font-weight:600;letter-spacing:.05em;">'
            f'PHASES · <span style="color:#64748b;">{_sb_run_id}</span></div>',
            unsafe_allow_html=True,
        )
        for _pi, (_pk, _pico, _plbl) in enumerate(_PHASES):
            if _pi < _cur_idx:
                _ps = "color:#16a34a;font-weight:700;"
                _pb = "✅"
            elif _pi == _cur_idx:
                _ps = "color:#0ea5e9;font-weight:700;"
                _pb = "▶"
            else:
                _ps = "color:#475569;"
                _pb = "○"
            st.markdown(
                f'<div style="padding:2px 0 2px 8px;{_ps}font-size:13px;">'
                f'{_pb} {_pico} {_plbl}'
                f'</div>',
                unsafe_allow_html=True,
            )
        # Steps progress
        _done_s = _sb_state.get("completed_steps", 0)
        _tot_s  = _sb_state.get("total_steps", 0)
        if _tot_s > 0:
            _sp = min(_done_s / _tot_s, 1.0)
            _sc = "#16a34a" if _sp >= 1.0 else "#0ea5e9"
            st.markdown(
                f'<div style="margin:4px 8px 0;">'
                f'<div style="display:flex;justify-content:space-between;font-size:11px;color:#94a3b8;">'
                f'<span>Steps</span><span>{_done_s}/{_tot_s}</span></div>'
                f'<div style="background:#1e293b;border-radius:4px;height:5px;margin-top:2px;">'
                f'<div style="background:{_sc};width:{_sp*100:.0f}%;height:5px;border-radius:4px;"></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Monitoring système ──
    st.markdown("### 📡 Monitoring")
    try:
        import psutil, time as _t
        _cpu   = psutil.cpu_percent(interval=0.1)
        _ram   = psutil.virtual_memory()
        _net   = psutil.net_io_counters()
        _disk  = psutil.disk_io_counters()

        # CPU
        _cpu_color = "#ef4444" if _cpu > 85 else ("#f59e0b" if _cpu > 60 else "#22c55e")
        st.markdown(
            f'<div style="margin:4px 0;">'
            f'<div style="display:flex;justify-content:space-between;font-size:12px;">'
            f'<span>🖥️ CPU</span><span style="color:{_cpu_color};font-weight:700;">{_cpu:.0f}%</span>'
            f'</div>'
            f'<div style="background:#1e293b;border-radius:4px;height:6px;margin-top:3px;">'
            f'<div style="background:{_cpu_color};width:{min(_cpu,100):.0f}%;height:6px;border-radius:4px;"></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        # RAM
        _ram_pct   = _ram.percent
        _ram_color = "#ef4444" if _ram_pct > 85 else ("#f59e0b" if _ram_pct > 70 else "#22c55e")
        _ram_used  = _ram.used / 1024**3
        _ram_total = _ram.total / 1024**3
        st.markdown(
            f'<div style="margin:4px 0;">'
            f'<div style="display:flex;justify-content:space-between;font-size:12px;">'
            f'<span>🧠 RAM</span><span style="color:{_ram_color};font-weight:700;">{_ram_used:.1f}/{_ram_total:.0f} GB</span>'
            f'</div>'
            f'<div style="background:#1e293b;border-radius:4px;height:6px;margin-top:3px;">'
            f'<div style="background:{_ram_color};width:{min(_ram_pct,100):.0f}%;height:6px;border-radius:4px;"></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        # GPU — WMI (instantané, system-wide, Intel Arc inclus)
        @st.cache_data(ttl=4)
        def _gpu_stats_win():
            """Lit utilisation + mémoire GPU via WMI (pas de délai de sampling)."""
            ps = (
                "try {"
                "$e=Get-WmiObject Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine -EA Stop;"
                "$u=[math]::Min(($e|Measure-Object UtilizationPercentage -Sum).Sum,100);"
                "$m=Get-WmiObject Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory -EA Stop;"
                "$d=($m|Measure-Object DedicatedUsage -Sum).Sum/1GB;"
                "$s=($m|Measure-Object SharedUsage -Sum).Sum/1GB;"
                "Write-Output \"$([math]::Round($u,1))|$([math]::Round($d,2))|$([math]::Round($s,2))\""
                "} catch { Write-Output 'err' }"
            )
            try:
                r = subprocess.run(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                    capture_output=True, text=True, timeout=5,
                )
                out = r.stdout.strip()
                if out and out != "err":
                    parts = out.split("|")
                    if len(parts) == 3:
                        try:
                            return float(parts[0] or 0), float(parts[1] or 0), float(parts[2] or 0)
                        except ValueError:
                            pass
            except subprocess.TimeoutExpired:
                pass
            return None, None, None

        _gpu_util, _gpu_ded, _gpu_shr = _gpu_stats_win()
        if _gpu_util is not None:
            _gcolor = "#ef4444" if _gpu_util > 85 else ("#f59e0b" if _gpu_util > 60 else "#6366f1")
            st.markdown(
                f'<div style="margin:4px 0;">'
                f'<div style="display:flex;justify-content:space-between;font-size:12px;">'
                f'<span>⚡ GPU</span>'
                f'<span style="color:{_gcolor};font-weight:700;">{_gpu_util:.0f}%'
                f' &nbsp;<span style="color:#94a3b8;font-weight:400;font-size:11px;">'
                f'D:{_gpu_ded:.1f}GB S:{_gpu_shr:.1f}GB</span></span>'
                f'</div>'
                f'<div style="background:#1e293b;border-radius:4px;height:6px;margin-top:3px;">'
                f'<div style="background:{_gcolor};width:{min(_gpu_util,100):.0f}%;height:6px;border-radius:4px;"></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div style="font-size:12px;color:#64748b;">⚡ GPU — N/A</div>', unsafe_allow_html=True)

        # Réseau
        _net_rx = _net.bytes_recv / 1024**2
        _net_tx = _net.bytes_sent / 1024**2
        st.markdown(
            f'<div style="margin:4px 0;font-size:12px;">'
            f'🌐 Réseau &nbsp;'
            f'<span style="color:#38bdf8;">↓ {_net_rx:.0f} MB</span> &nbsp;'
            f'<span style="color:#818cf8;">↑ {_net_tx:.0f} MB</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # I/O disque
        if _disk:
            _io_r = _disk.read_bytes  / 1024**3
            _io_w = _disk.write_bytes / 1024**3
            st.markdown(
                f'<div style="margin:4px 0;font-size:12px;">'
                f'💾 I/O &nbsp;'
                f'<span style="color:#34d399;">R {_io_r:.1f} GB</span> &nbsp;'
                f'<span style="color:#fb923c;">W {_io_w:.1f} GB</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    except ImportError:
        st.caption("psutil non installé — `pip install psutil`")


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 0 — Vérification environnement
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🔍 Étape 0 — Vérification de l\'environnement</div>', unsafe_allow_html=True)

def _check_env():
    checks = {}
    # Python
    checks["python"] = (True, sys.version.split()[0])
    # torch
    try:
        import torch
        checks["torch"] = (True, torch.__version__)
    except ImportError:
        checks["torch"] = (False, "non installé")
    # IPEX
    try:
        import intel_extension_for_pytorch as ipex
        checks["ipex"] = (True, ipex.__version__)
    except Exception:
        checks["ipex"] = (False, "oneAPI Base Toolkit requis")
    # XPU disponible
    try:
        import torch
        import intel_extension_for_pytorch  # noqa
        xpu_ok = torch.xpu.is_available()
        checks["xpu"] = (xpu_ok, "Intel Arc détecté" if xpu_ok else "Arc non accessible via XPU")
    except Exception:
        checks["xpu"] = (False, "IPEX indisponible")
    # transformers
    try:
        import transformers
        checks["transformers"] = (True, transformers.__version__)
    except ImportError:
        checks["transformers"] = (False, "pip install transformers>=4.44")
    # peft
    try:
        import peft
        checks["peft"] = (True, peft.__version__)
    except ImportError:
        checks["peft"] = (False, "pip install peft>=0.11")
    # trl
    try:
        import trl
        checks["trl"] = (True, trl.__version__)
    except ImportError:
        checks["trl"] = (False, "pip install trl>=0.9")
    # Modèle 3B local (HF cache ou finetuning/base/)
    base_path = FT_DIR / "base" / "Qwen2.5-VL-3B-Instruct"
    checks["model_3b"] = (base_path.exists(), str(base_path) if base_path.exists() else "non téléchargé")
    return checks

c_env = _check_env()
cols = st.columns(4)
items = list(c_env.items())
for i, (name, (ok, detail)) in enumerate(items):
    with cols[i % 4]:
        icon = "✓" if ok else "✗"
        color = "#22c55e" if ok else "#ef4444"
        st.markdown(
            f'<div class="metric-card"><div class="metric-val" style="color:{color};">{icon}</div>'
            f'<div class="metric-lab"><strong>{name}</strong><br>{detail}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown("")

# ── Légende des blocs de vérification ──────────────────────────────────────
_stc.html("""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body{margin:0;padding:0;font-family:'Segoe UI',sans-serif;font-size:12px;background:transparent;}
.bl{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;padding:2px 0 8px 0;}
.bi{background:#0f172a;border-radius:8px;padding:10px 12px;border:1px solid #1e293b;}
.bi-name{font-size:11px;font-weight:700;color:#7dd3fc;margin-bottom:4px;letter-spacing:.3px;}
.bi-ver{display:inline-block;padding:1px 6px;border-radius:3px;background:#1e293b;color:#94a3b8;font-size:10px;font-family:monospace;margin-bottom:4px;}
.bi-desc{color:#cbd5e1;font-size:11px;line-height:1.5;}
.bi-dep{font-size:10px;color:#f59e0b;margin-top:3px;}
</style>
<div class="bl">
  <div class="bi">
    <div class="bi-name">🐍 python</div>
    <div class="bi-ver">3.12.3</div>
    <div class="bi-desc">Runtime requis pour compatibilité IPEX 2.8.x et ensemble de la stack HuggingFace.</div>
  </div>
  <div class="bi">
    <div class="bi-name">🔥 torch</div>
    <div class="bi-ver">2.8.0+xpu</div>
    <div class="bi-desc">Build XPU obligatoire — le build <code>+cpu</code> ne contient pas <code>c10_xpu.dll</code>, IPEX ne peut pas charger le backend Arc.</div>
  </div>
  <div class="bi">
    <div class="bi-name">⚙️ ipex</div>
    <div class="bi-ver">2.8.10+xpu</div>
    <div class="bi-desc">Intel Extension for PyTorch — expose le backend <code>xpu</code> pour Intel Arc. Requiert les DLL oneAPI dans le PATH.</div>
    <div class="bi-dep">⚠ oneAPI Base Toolkit 2025.1 requis</div>
  </div>
  <div class="bi">
    <div class="bi-name">⚡ xpu</div>
    <div class="bi-ver">torch.xpu</div>
    <div class="bi-desc">Disponibilité effective du device Arc — <code>is_available() → True</code>. Sinon, fallback CPU (~3-4× plus lent).</div>
  </div>
  <div class="bi">
    <div class="bi-name">🤗 transformers</div>
    <div class="bi-ver">5.2.0</div>
    <div class="bi-desc"><code>AutoModelForVision2Seq</code>, <code>AutoProcessor</code>, tokenizer Qwen2.5-VL — chargement et inférence du modèle.</div>
  </div>
  <div class="bi">
    <div class="bi-name">🔧 peft</div>
    <div class="bi-ver">0.18.1</div>
    <div class="bi-desc">Parameter-Efficient Fine-Tuning — <code>LoraConfig(r=4)</code> sur <code>q_proj</code> + <code>v_proj</code> : seuls ~0.05% des paramètres sont entraînés.</div>
  </div>
  <div class="bi">
    <div class="bi-name">📐 trl</div>
    <div class="bi-ver">0.29.0</div>
    <div class="bi-desc"><code>SFTTrainer</code> et <code>DataCollatorForSeq2Seq</code> — orchestre la boucle d'entraînement supervisé sur les séquences image→JSON.</div>
  </div>
  <div class="bi">
    <div class="bi-name">🧠 model_3b</div>
    <div class="bi-ver">~7.1 GB</div>
    <div class="bi-desc">Qwen2.5-VL-3B-Instruct en FP16 local — modèle de base avant LoRA. Stocker dans <code>finetuning/base/</code>.</div>
  </div>
</div>
</body></html>""", height=270)

# ── Architecture fonctionnelle · technique · hardware ──────────────────────
st.markdown('<div class="section-header">🏗️ Architecture fine-tuning — Fonctionnel · Technique · Hardware</div>', unsafe_allow_html=True)

_stc.html("""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
body { margin:0; padding:0; background:transparent; font-family:'Segoe UI',sans-serif; }
.srv-wrap {
    background: #0a0e1a; border-radius: 14px; padding: 28px 20px;
    display: flex; gap: 20px; align-items: flex-start;
    justify-content: center; flex-wrap: wrap;
}
.srv-col { display: flex; flex-direction: column; align-items: center; gap: 0; min-width: 180px; }
.srv-col-label {
    color: #64748b; font-size: 10px; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; margin-bottom: 10px;
}
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
.l-app    { background:#1e3a5f; color:#bae6fd; border:1px solid #0ea5e9; }
.l-py     { background:#0f3460; color:#93c5fd; border:1px dashed #3b82f6; font-size:11px; }
.l-api    { background:#1a2744; color:#7dd3fc; border:1px solid #1d4ed8; font-size:11px; }
.l-srv    { background:#2d1a5e; color:#c4b5fd; border:2px solid #7c3aed; }
.l-bin    { background:#1e1040; color:#a78bfa; border:1px dashed #6d28d9; font-size:11px; }
.l-model  { background:#13082e; color:#ddd6fe; border:1px solid #5b21b6; font-size:10px; }
.l-gpu    { background:#2a0a2e; color:#f5d0fe; border:2px solid #c026d3; }
.l-hw     { background:#150020; color:#f3e8ff; border:2px solid #7c3aed; }
.l-os     { background:#0f172a; color:#94a3b8; border:1px solid #334155; font-size:10px; }
.l-store  { background:#0a2010; color:#86efac; border:1px solid #16a34a; font-size:11px; }
.l-mlflow { background:#071520; color:#7dd3fc; border:1px solid #0369a1; font-size:11px; }
.l-peft   { background:#1a0f40; color:#c4b5fd; border:1px solid #6d28d9; font-size:11px; }
.l-ipex   { background:#0a1a30; color:#60a5fa; border:1px dashed #2563eb; font-size:11px; }
.l-oneapi { background:#0c1f10; color:#4ade80; border:1px dashed #16a34a; font-size:10px; }
.srv-badge {
    display:inline-block; background:#312e81; color:#c7d2fe;
    border-radius:3px; padding:1px 5px; font-size:9px; margin:1px;
}
.srv-badge-green { background:#14532d; color:#86efac; }
.srv-badge-orange { background:#431407; color:#fdba74; }
.srv-badge-purple { background:#4c1d95; color:#ddd6fe; }
</style>

<div class="srv-wrap">

  <!-- ═══ COL 1 : Flux fonctionnel ═══ -->
  <div class="srv-col" style="min-width:190px;">
    <div class="srv-col-label">⬛ Flux fonctionnel</div>

    <div class="srv-layer l-app">
      🎯 03_FineTuning.py<br>
      <span style="font-size:10px;opacity:.8;">Workflow guidé 6 étapes · Streamlit</span>
    </div>
    <div class="srv-arrow">↓</div>
    <div class="srv-layer l-py">
      🐍 finetune_xpu.py<br>
      <span class="srv-badge">DocumentDataset</span>
      <span class="srv-badge">MetricsCallback</span>
    </div>
    <div class="srv-arrow">↓</div>
    <div class="srv-layer l-api">
      📋 prompt_registry.py<br>
      <span style="font-size:10px;">registry.json · prompts + schemas</span><br>
      <span class="srv-badge">get_eval_fields()</span>
      <span class="srv-badge">get_schema()</span>
    </div>
    <div class="srv-arrow">↓</div>
    <div class="srv-layer l-py" style="font-size:11px;">
      🏋️ Boucle entraînement<br>
      <span class="srv-badge">×2 tâches/image</span>
      <span class="srv-badge">classify + extract</span>
    </div>
    <div class="srv-arrow">↓</div>
    <div class="srv-layer l-api">
      📊 evaluate_model()<br>
      <span style="font-size:10px;">F1 exact-match · eval_fields registry</span>
    </div>
    <div class="srv-arrow">↓</div>
    <div class="srv-layer l-store">
      📁 Sorties<br>
      <span class="srv-badge srv-badge-green">adapter/ LoRA</span>
      <span class="srv-badge srv-badge-green">merged/ FP16</span><br>
      <span class="srv-badge srv-badge-green">*.gguf Q4_K_M</span>
    </div>
    <div class="srv-arrow">↓</div>
    <div class="srv-layer l-mlflow">
      📈 MLflow<br>
      <span style="font-size:10px;">qwen_vl_finetuning · logs/ · :5000</span>
    </div>
  </div>

  <!-- ═══ CONNECTOR ═══ -->
  <div class="srv-connector">
    <div class="srv-connector-line"></div>
    <div style="padding:0 8px;color:#4f46e5;font-size:16px;">⟷</div>
    <div style="color:#6366f1;font-size:10px;writing-mode:vertical-lr;transform:rotate(180deg);padding:4px 2px;">stack</div>
    <div style="padding:0 8px;color:#4f46e5;font-size:16px;">⟷</div>
    <div class="srv-connector-line"></div>
  </div>

  <!-- ═══ COL 2 : Stack technique ═══ -->
  <div class="srv-col" style="min-width:210px;">
    <div class="srv-col-label">⚙️ Stack technique</div>

    <div class="srv-layer l-py">
      🐍 Python 3.12.3<br>
      <span style="font-size:10px;">Windows 11 · venv isolé</span>
    </div>
    <div class="srv-arrow">↓</div>
    <div class="srv-layer l-py">
      🤗 Transformers 5.2.0<br>
      <span class="srv-badge">AutoModelForVision2Seq</span>
      <span class="srv-badge">AutoProcessor</span>
    </div>
    <div class="srv-arrow">↓</div>
    <div class="srv-layer l-peft">
      🔧 PEFT 0.18.1 — LoRA r=4<br>
      <span class="srv-badge srv-badge-purple">q_proj</span>
      <span class="srv-badge srv-badge-purple">v_proj</span>
      <span style="font-size:10px;color:#a78bfa;">~0.05% params entraînés</span>
    </div>
    <div class="srv-arrow">↓</div>
    <div class="srv-layer l-bin">
      📐 TRL 0.29.0<br>
      <span class="srv-badge">SFTTrainer</span>
      <span class="srv-badge">DataCollatorForSeq2Seq</span>
    </div>
    <div class="srv-arrow">↓</div>
    <div class="srv-layer l-ipex">
      ⚙️ IPEX 2.8.10+xpu<br>
      <span style="font-size:10px;">Backend XPU Intel Arc · FP16</span><br>
      <span class="srv-badge srv-badge-orange">oneAPI DLL requis</span>
    </div>
    <div class="srv-arrow">↓</div>
    <div class="srv-layer l-api">
      🔥 PyTorch 2.8.0+xpu<br>
      <span style="font-size:10px;">build XPU — c10_xpu.dll inclus</span>
    </div>
    <div class="srv-arrow">↓</div>
    <div class="srv-layer l-oneapi">
      🧩 Intel oneAPI Base Toolkit 2025.1<br>
      <span style="font-size:10px;">sycl8.dll · esimd_kernels.dll · mkl</span><br>
      <span class="srv-badge srv-badge-green">PATH : start_with_xpu.bat</span>
    </div>
  </div>

  <!-- ═══ CONNECTOR ═══ -->
  <div class="srv-connector">
    <div class="srv-connector-line"></div>
    <div style="padding:0 8px;color:#4f46e5;font-size:16px;">⟷</div>
    <div style="color:#6366f1;font-size:10px;writing-mode:vertical-lr;transform:rotate(180deg);padding:4px 2px;">hw</div>
    <div style="padding:0 8px;color:#4f46e5;font-size:16px;">⟷</div>
    <div class="srv-connector-line"></div>
  </div>

  <!-- ═══ COL 3 : Hardware ═══ -->
  <div class="srv-col" style="min-width:185px;">
    <div class="srv-col-label">💻 Hardware</div>

    <div class="srv-layer l-gpu">
      ⚡ Intel Arc Graphics (UMA)<br>
      <span style="font-size:10px;font-weight:400;">Xe-HPG · ~8 TFLOPS FP16</span><br>
      <span class="srv-badge srv-badge-green">torch.xpu → True</span>
      <span class="srv-badge">17.5 GB partagé</span>
    </div>
    <div class="srv-arrow">↓</div>
    <div class="srv-layer l-model">
      🧠 Qwen2.5-VL-3B-Instruct<br>
      <span style="font-size:10px;">FP16 · 7.1 GB · 2 safetensors</span><br>
      <span class="srv-badge srv-badge-purple">finetuning/base/</span>
    </div>
    <div class="srv-arrow">↓</div>
    <div class="srv-layer l-hw">
      💻 Intel Core Ultra 5 125H<br>
      <span style="font-size:10px;font-weight:400;">Meteor Lake · 14 cores · 4.5 GHz</span><br>
      <span class="srv-badge srv-badge-green">CPU + iGPU + NPU on-die</span>
    </div>
    <div class="srv-arrow">↓</div>
    <div class="srv-layer l-bin">
      🧱 32 GB DDR5<br>
      <span style="font-size:10px;">Partagé CPU + Arc UMA</span><br>
      <span class="srv-badge">~9 GB modèle + LoRA</span>
      <span class="srv-badge">~6 GB OS</span>
    </div>
    <div class="srv-arrow">↓</div>
    <div class="srv-layer l-os">
      🪟 Windows 11 Home<br>
      <span style="font-size:10px;">Intel Graphics Driver 31.x</span><br>
      <span class="srv-badge">Arc GPU préinstallé</span>
    </div>
    <div class="srv-arrow">↓</div>
    <div class="srv-layer l-store" style="border:2px solid #16a34a;">
      🎯 ~35 min · 240 steps<br>
      <span style="font-size:10px;">80 ex. × 3 epochs · ~8s/step XPU</span><br>
      <span class="srv-badge srv-badge-green">LoRA adapter → GGUF Q4_K_M</span>
    </div>
  </div>

</div>
</body></html>""", height=820)

st.divider()

env_ok = all(ok for name, (ok, _) in c_env.items() if name not in ("xpu", "ipex", "model_3b"))
xpu_ok  = c_env["xpu"][0]
ipex_ok = c_env["ipex"][0]
model_3b_ok = c_env["model_3b"][0]

if not env_ok:
    st.error("**Dépendances manquantes.** Installez-les avant de continuer.")
    st.code("pip install transformers>=4.44 peft>=0.11 trl>=0.9 accelerate", language="bash")

if not ipex_ok or not xpu_ok:
    st.warning(
        "**Intel Arc XPU non disponible** — l'entraînement utilisera le CPU (×4–5 plus lent, ~2–3h). "
        "Pour activer XPU : installez Intel oneAPI Base Toolkit puis relancez."
    )

if not model_3b_ok:
    st.warning("**Modèle 3B non trouvé localement.**")
    if st.button("⬇ Télécharger Qwen2.5-VL-3B-Instruct (~6 GB)"):
        dest = FT_DIR / "base"
        dest.mkdir(parents=True, exist_ok=True)
        st.info("Téléchargement en cours — peut prendre 10–20 min selon connexion…")
        result = subprocess.run(
            [sys.executable, "-c",
             f"from huggingface_hub import snapshot_download; "
             f"snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct', local_dir='{dest}/Qwen2.5-VL-3B-Instruct')"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            st.success("Modèle téléchargé.")
            st.rerun()
        else:
            st.error(result.stderr[-500:])

if env_ok and S.ft_step == 0:
    if st.button("✅ Environnement validé — Passer au dataset", type="primary"):
        S.ft_step = 1
        st.rerun()

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — Dataset
# ══════════════════════════════════════════════════════════════════════════════
if S.ft_step >= 1:
    st.markdown('<div class="section-header">📂 Étape 1 — Préparation du dataset</div>', unsafe_allow_html=True)

    # ── Sélection du type de document ────────────────────────────────────────
    _DOC_TYPES = {
        "chart":         "📈 Graphique financier (OHLCV)",
        "invoice":       "🧾 Facture",
        "bank_statement":"🏦 Relevé bancaire",
        "contract":      "📜 Contrat",
        "other":         "📄 Générique",
    }
    _CHART_FIELDS = ["symbol", "chart_date", "timeframe", "open", "high", "low", "close", "volume", "chart_type"]
    _SCHEMAS = {
        "chart":          {"symbol":"TVC:CAC40","chart_date":"2026-03-13","timeframe":"1D","open":"8245.50","high":"8312.75","low":"8198.25","close":"8287.00","volume":"125M","chart_type":"candlestick"},
        "invoice":        {"invoice_number":"INV-001","date":"2026-03-13","vendor":"ACME Corp","total_amount":"1250.00","currency":"EUR"},
        "bank_statement": {"iban":"FR76...","bank_name":"BNP","period_start":"2026-03-01","period_end":"2026-03-31","opening_balance":"5000.00","closing_balance":"4250.00","currency":"EUR"},
        "contract":       {"contract_date":"2026-03-13","parties":["Partie A","Partie B"],"contract_value":"50000 EUR"},
        "other":          {},
    }

    col_dt1, col_dt2 = st.columns([1, 2])
    with col_dt1:
        S.ft_doc_type = st.selectbox(
            "Type de document à fine-tuner",
            list(_DOC_TYPES.keys()),
            format_func=lambda k: _DOC_TYPES[k],
            index=list(_DOC_TYPES.keys()).index(S.ft_doc_type) if S.ft_doc_type in _DOC_TYPES else 0,
        )
    with col_dt2:
        _tmpl = _SCHEMAS.get(S.ft_doc_type, {})
        st.markdown("**Exemple d'entrée `labels.json` pour ce type :**")
        st.json({
            "image": "images/sample.jpg",
            "doc_type": S.ft_doc_type,
            "expected": _tmpl
        }, expanded=False)

    with st.expander("📋 Format attendu", expanded=False):
        st.markdown("""
**ZIP contenant :**
```
dataset.zip
├── images/
│   ├── doc_001.jpg
│   ├── doc_002.jpg
│   └── ...
└── labels.json      ← liste de paires {image, expected_json}
```
**labels.json :**
```json
[
  {
    "image": "images/doc_001.jpg",
    "doc_type": "invoice",
    "expected": {
      "invoice_number": "INV-2024-001",
      "date": "2024-01-15",
      "total": "4250.00 EUR",
      "vendor": "ACME Corp"
    }
  }
]
```
**Ou** utiliser les résultats du pipeline (extractions validées) comme source de vérité.
""")

    tab_upload, tab_label, tab_pipeline = st.tabs(["📤 Upload ZIP", "✏️ Labeling manuel", "🔁 Depuis résultats pipeline"])

    with tab_upload:
        uploaded = st.file_uploader("Dataset ZIP (images + labels.json)", type=["zip"])
        if uploaded:
            ds_path = FT_DIR / "dataset"
            if ds_path.exists():
                shutil.rmtree(ds_path)
            ds_path.mkdir(parents=True)
            zip_tmp = FT_DIR / "upload.zip"
            zip_tmp.write_bytes(uploaded.read())
            with zipfile.ZipFile(zip_tmp) as z:
                z.extractall(ds_path)
            labels_candidates = list(ds_path.rglob("labels.json"))
            if labels_candidates:
                labels = json.loads(labels_candidates[0].read_text(encoding="utf-8"))
                n_total = len(labels)
                n_val   = max(1, int(n_total * 0.2))
                n_train = n_total - n_val
                S.ft_dataset_path = str(ds_path)
                S.ft_n_train = n_train
                S.ft_n_val   = n_val
                st.success(f"✓ {n_total} exemples chargés — train: {n_train} · val: {n_val}")
                # Preview
                st.markdown("**Aperçu (3 premiers exemples) :**")
                for ex in labels[:3]:
                    with st.container():
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            img_path = ds_path / ex.get("image", "")
                            if img_path.exists():
                                st.image(str(img_path), use_container_width=True)
                        with c2:
                            st.json(ex.get("expected", {}), expanded=False)
            else:
                st.error("labels.json introuvable dans le ZIP.")

    with tab_label:
        st.markdown(f"**Labeling — `{S.ft_doc_type}` · {_DOC_TYPES.get(S.ft_doc_type,'')}**")

        # ── Capture automatique (chart uniquement) ───────────────────────────
        if S.ft_doc_type == "chart":
            with st.expander("🤖 Capture automatique TradingView (40 screenshots)", expanded=True):
                st.markdown("""
**Script Playwright** — ouvre TradingView en plein écran, survole 40 bougies hebdomadaires (1Y / 1D)
et enregistre un screenshot JPEG par point de donnée + les valeurs OHLCV depuis Yahoo Finance.

**Prérequis (à lancer une seule fois) :**
```bash
pip install playwright yfinance pillow
playwright install chromium
```
""")
                col_run, col_status = st.columns([1, 2])
                with col_run:
                    n_target = st.number_input("Nombre de screenshots", min_value=5, max_value=100, value=40, step=5)
                    if st.button("▶ Lancer la capture", type="primary"):
                        script = ROOT / "scripts" / "capture_chart_dataset.py"
                        with st.spinner("Installation dépendances + capture en cours (peut prendre 5–10 min)…"):
                            result = subprocess.run(
                                [sys.executable, str(script), str(n_target)],
                                capture_output=True, text=True, cwd=str(ROOT),
                                timeout=900,
                                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                            )
                        if result.returncode == 0:
                            # Recharger labels générés
                            lf = FT_DIR / "dataset" / "labels.json"
                            if lf.exists():
                                S.ft_labels = json.loads(lf.read_text(encoding="utf-8"))
                                n_total = len(S.ft_labels)
                                n_val   = max(1, int(n_total * 0.2))
                                S.ft_dataset_path = str(FT_DIR / "dataset")
                                S.ft_n_train = n_total - n_val
                                S.ft_n_val   = n_val
                            st.success(f"✅ {len(S.ft_labels)} screenshots capturés")
                        else:
                            st.error("Erreur capture :")
                            st.code(result.stderr[-800:])
                with col_status:
                    imgs = list((FT_DIR / "dataset" / "images").glob("chart_cac40_*.jpg")) if (FT_DIR / "dataset" / "images").exists() else []
                    lf_existing = FT_DIR / "dataset" / "labels.json"
                    if imgs:
                        st.metric("Images capturées", len(imgs))
                        if lf_existing.exists():
                            n_total = len(json.loads(lf_existing.read_text(encoding="utf-8")))
                            n_val   = max(1, int(n_total * 0.2))
                            S.ft_dataset_path = str(FT_DIR / "dataset")
                            S.ft_n_train = n_total - n_val
                            S.ft_n_val   = n_val
                            if st.button("⚙️ Passer à la configuration", type="primary"):
                                S.ft_step = 2
                                st.rerun()
                        # Aperçu des 3 dernières
                        for img_f in sorted(imgs)[-3:]:
                            st.image(str(img_f), caption=img_f.stem, use_container_width=True)
                    else:
                        st.info("Aucune image capturée pour l'instant.")

        st.divider()
        st.caption("Saisie manuelle — ajoutez ou corrigez des exemples individuellement.")

        # ── Formulaire par type de document ─────────────────────────────────
        with st.form("label_form", clear_on_submit=True):
            img_path = st.text_input("📁 Chemin image (relatif au dataset)", placeholder="images/chart_cac40_20260313.jpg")

            if S.ft_doc_type == "chart":
                c1, c2 = st.columns(2)
                with c1:
                    symbol    = st.text_input("Symbole", placeholder="TVC:CAC40")
                    chart_date= st.date_input("Date du chandelier lu")
                    timeframe = st.selectbox("Timeframe", ["1m","5m","15m","30m","1H","4H","1D","1W","1M"], index=6)
                    chart_type= st.selectbox("Type de graphique", ["candlestick","bar","line"])
                with c2:
                    open_v  = st.text_input("Open",   placeholder="8245.50")
                    high_v  = st.text_input("High",   placeholder="8312.75")
                    low_v   = st.text_input("Low",    placeholder="8198.25")
                    close_v = st.text_input("Close",  placeholder="8287.00")
                    volume  = st.text_input("Volume (optionnel)", placeholder="125M")
                submitted = st.form_submit_button("➕ Ajouter au dataset", type="primary")
                if submitted and img_path and symbol and open_v and close_v:
                    S.ft_labels.append({
                        "image": img_path,
                        "doc_type": "chart",
                        "expected": {
                            "symbol": symbol,
                            "chart_date": str(chart_date),
                            "timeframe": timeframe,
                            "open": open_v,
                            "high": high_v,
                            "low": low_v,
                            "close": close_v,
                            "volume": volume or None,
                            "chart_type": chart_type,
                        }
                    })

            elif S.ft_doc_type == "invoice":
                c1, c2 = st.columns(2)
                with c1:
                    inv_num  = st.text_input("N° facture")
                    inv_date = st.date_input("Date facture")
                    vendor   = st.text_input("Émetteur (vendor)")
                with c2:
                    total    = st.text_input("Montant total")
                    currency = st.selectbox("Devise", ["EUR","USD","GBP","CHF"])
                submitted = st.form_submit_button("➕ Ajouter", type="primary")
                if submitted and img_path and inv_num:
                    S.ft_labels.append({
                        "image": img_path, "doc_type": "invoice",
                        "expected": {"invoice_number": inv_num, "date": str(inv_date),
                                     "vendor": vendor, "total_amount": total, "currency": currency}
                    })

            else:
                raw_json = st.text_area("Champs JSON (expected)", height=120,
                                        placeholder='{"key": "value"}')
                submitted = st.form_submit_button("➕ Ajouter", type="primary")
                if submitted and img_path:
                    try:
                        expected = json.loads(raw_json) if raw_json.strip() else {}
                        S.ft_labels.append({"image": img_path, "doc_type": S.ft_doc_type, "expected": expected})
                    except json.JSONDecodeError:
                        st.error("JSON invalide.")

        # ── Tableau des labels saisis ────────────────────────────────────────
        if S.ft_labels:
            st.markdown(f"**{len(S.ft_labels)} exemple(s) saisi(s)**")
            for i, lbl in enumerate(S.ft_labels):
                with st.expander(f"#{i+1} · {lbl['image']}", expanded=False):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.json(lbl["expected"], expanded=True)
                    with c2:
                        if st.button("🗑 Supprimer", key=f"del_{i}"):
                            S.ft_labels.pop(i)
                            st.rerun()

            col_save, col_dl = st.columns(2)
            with col_save:
                if st.button("💾 Sauvegarder labels.json", type="primary"):
                    ds_path = FT_DIR / "dataset"
                    ds_path.mkdir(parents=True, exist_ok=True)
                    (ds_path / "labels.json").write_text(
                        json.dumps(S.ft_labels, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    n_total = len(S.ft_labels)
                    n_val   = max(1, int(n_total * 0.2))
                    S.ft_dataset_path = str(ds_path)
                    S.ft_n_train = n_total - n_val
                    S.ft_n_val   = n_val
                    st.success(f"✓ {n_total} exemples sauvegardés — train: {S.ft_n_train} · val: {S.ft_n_val}")
            with col_dl:
                st.download_button(
                    "⬇ Télécharger labels.json",
                    data=json.dumps(S.ft_labels, ensure_ascii=False, indent=2),
                    file_name="labels.json", mime="application/json"
                )
        else:
            st.info("Aucun exemple saisi. Remplissez le formulaire ci-dessus.")

    with tab_pipeline:
        results_dir = ROOT / "logs"
        result_files = list(results_dir.glob("extraction_*.json"))
        if result_files:
            st.markdown(f"**{len(result_files)} extraction(s) trouvée(s) dans logs/**")
            selected = st.multiselect(
                "Sélectionner les extractions validées comme données d'entraînement",
                [f.name for f in result_files],
                default=[f.name for f in result_files[:min(80, len(result_files))]]
            )
            if selected and st.button("Convertir en dataset LoRA"):
                ds_path = FT_DIR / "dataset"
                ds_path.mkdir(parents=True, exist_ok=True)
                labels = []
                for fname in selected:
                    try:
                        data = json.loads((results_dir / fname).read_text(encoding="utf-8"))
                        labels.append({
                            "image": data.get("source_file", ""),
                            "doc_type": data.get("document_type", ""),
                            "expected": data.get("fields", {})
                        })
                    except Exception:
                        pass
                (ds_path / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2))
                n_total = len(labels)
                n_val = max(1, int(n_total * 0.2))
                S.ft_dataset_path = str(ds_path)
                S.ft_n_train = n_total - n_val
                S.ft_n_val = n_val
                st.success(f"✓ Dataset créé : {n_total} exemples")
        else:
            st.info("Aucune extraction trouvée dans logs/. Lancez le pipeline sur quelques documents d'abord.")

    if S.ft_dataset_path and S.ft_step == 1:
        if S.ft_n_train < 20:
            st.warning(f"⚠ Seulement {S.ft_n_train} exemples d'entraînement — minimum recommandé : 40.")
        if st.button("✅ Dataset prêt — Passer à la configuration", type="primary"):
            S.ft_step = 2
            st.rerun()

    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — Configuration
# ══════════════════════════════════════════════════════════════════════════════
if S.ft_step >= 2:
    st.markdown('<div class="section-header">⚙️ Étape 2 — Configuration de l\'entraînement</div>', unsafe_allow_html=True)

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)

    with col_cfg1:
        st.markdown("**Modèle & LoRA**")
        model_id = st.text_input("Modèle de base", "Qwen/Qwen2.5-VL-3B-Instruct")
        lora_r   = st.select_slider("Rang LoRA (r)", [2, 4, 8, 16, 32], value=4)
        lora_alpha = st.number_input("lora_alpha", value=8, min_value=1)
        target_mods = st.multiselect(
            "Modules cibles",
            ["q_proj", "v_proj", "k_proj", "o_proj"],
            default=["q_proj", "v_proj"]
        )

    with col_cfg2:
        st.markdown("**Entraînement**")
        n_epochs  = st.slider("Epochs", 1, 5, 3)
        lr        = st.select_slider("Learning rate", [5e-5, 1e-4, 2e-4, 5e-4], value=2e-4)
        batch_sz  = st.slider("Batch size (gradient accum ×4)", 1, 4, 1)
        max_seq   = st.select_slider("Max sequence length", [512, 1024, 2048], value=1024)
        early_stop = st.checkbox("Early stopping (patience=2)", value=True)

        st.markdown("**Split du dataset**")
        n_available = S.ft_n_train + S.ft_n_val
        n_use = st.slider("Échantillon total à utiliser", min_value=10, max_value=n_available, value=n_available, step=1)
        _pct_test  = st.slider("% Test  (holdout)",     5, 30, 15, step=5)
        _pct_val   = st.slider("% Val   (monitoring)",  5, 30, 15, step=5)
        _pct_train = max(5, 100 - _pct_test - _pct_val)
        n_test_split  = max(1, int(n_use * _pct_test  / 100))
        n_val_split   = max(1, int(n_use * _pct_val   / 100))
        n_train_split = max(1, n_use - n_test_split - n_val_split)
        st.caption(f"→ {n_train_split} train ({_pct_train}%) · {n_val_split} val ({_pct_val}%) · {n_test_split} test ({_pct_test}%)  sur {n_use}/{n_available}")

    with col_cfg3:
        st.markdown("**Hardware & export**")
        device    = st.selectbox("Device", ["xpu (Intel Arc)", "cpu"], index=0 if xpu_ok else 1)
        quant_out = st.selectbox("Quantification GGUF sortie", ["q4_k_m", "q5_k_m", "q8_0"], index=0)
        run_name  = st.text_input("Nom du run", f"ft_{datetime.now().strftime('%Y%m%d_%H%M')}")

    # Estimation temps — basée sur n_train_split (échantillon réel)
    n_steps  = max(1, (n_train_split * 2 * n_epochs) // (batch_sz * 4))  # ×2 tâches/image
    sec_step = 80 if "xpu" in device else 300   # gradient_checkpointing ≈ 80s/step XPU
    est_min  = (n_steps * sec_step) // 60
    st.markdown(
        f'<div class="callout-info">⏱ Estimation : <strong>{n_steps} steps</strong> × '
        f'~{sec_step}s = <strong>~{est_min} min</strong> sur {device} '
        f'({n_train_split} train · {n_val_split} val · {n_test_split} test)</div>',
        unsafe_allow_html=True
    )

    if S.ft_step == 2:
        if st.button("✅ Config validée — Préparer l'entraînement", type="primary"):
            S.ft_config = {
                "model_id": model_id,
                "local_model": str(FT_DIR / "base" / "Qwen2.5-VL-3B-Instruct"),
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "target_modules": target_mods,
                "n_epochs": n_epochs,
                "learning_rate": lr,
                "batch_size": batch_sz,
                "max_seq_length": max_seq,
                "early_stopping": early_stop,
                "device": "xpu" if "xpu" in device else "cpu",
                "quant_out": quant_out,
                "run_name": run_name,
                "dataset_path": S.ft_dataset_path,
                "n_samples": n_use,
                "n_train":   n_train_split,
                "n_val":     n_val_split,
                "n_test":    n_test_split,
                "pct_train": _pct_train,
                "pct_val":   _pct_val,
                "pct_test":  _pct_test,
                "output_dir": str(FT_DIR / run_name),
            }
            S.ft_run_id = run_name
            Path(S.ft_config["output_dir"]).mkdir(parents=True, exist_ok=True)
            (FT_DIR / f"{run_name}_config.json").write_text(
                json.dumps(S.ft_config, indent=2), encoding="utf-8"
            )
            S.ft_step = 3
            st.rerun()

    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — Entraînement
# ══════════════════════════════════════════════════════════════════════════════
if S.ft_step >= 3:
    st.markdown('<div class="section-header">🚀 Étape 3 — Entraînement LoRA</div>', unsafe_allow_html=True)

    # ── Reprendre un run existant ──────────────────────────────────────────────
    existing_runs = []
    for _d in sorted(FT_DIR.iterdir(), reverse=True) if FT_DIR.exists() else []:
        _sf = _d / "training_state.json"
        if _d.is_dir() and _sf.exists():
            try:
                _st = json.loads(_sf.read_text(encoding="utf-8"))
                _phase = _st.get("phase", "?")
                _steps = f"{_st.get('completed_steps',0)}/{_st.get('total_steps','?')}"
                _phase_icon = {"training":"🔵","merging":"🔀","testing":"🟡","validation":"🟠","done":"✅"}.get(_phase,"❓")
                existing_runs.append((_d.name, _phase_icon, _phase, _steps, _st))
            except Exception:
                pass

    if existing_runs:
        with st.expander(f"📂 Runs existants ({len(existing_runs)}) — cliquer pour reprendre", expanded=(S.ft_run_id is None)):
            for _rid, _icon, _phase, _steps, _st in existing_runs:
                _c1, _c2, _c3, _c4 = st.columns([2, 1, 1, 1])
                with _c1:
                    st.markdown(f"`{_rid}`")
                with _c2:
                    st.markdown(f"{_icon} {_phase}")
                with _c3:
                    st.markdown(f"steps {_steps}")
                with _c4:
                    _active = (S.ft_run_id == _rid)
                    _label  = "✅ Actif" if _active else "↩ Reprendre"
                    if not _active and st.button(_label, key=f"resume_{_rid}"):
                        # Charger la config du run
                        _cfg_file = FT_DIR / f"{_rid}_config.json"
                        if _cfg_file.exists():
                            S.ft_config = json.loads(_cfg_file.read_text(encoding="utf-8"))
                        S.ft_run_id = _rid
                        S.ft_step   = 3
                        st.rerun()

    # ── Run courant ────────────────────────────────────────────────────────────
    cfg      = S.ft_config
    run_id   = S.ft_run_id or "run"
    log_file = LOGS_DIR / f"ft_{run_id}.jsonl"
    pid_file = FT_DIR / f"{run_id}.pid"

    # ── Vérifier si un entraînement est en cours ──
    is_running = False
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text())
            import psutil
            is_running = psutil.pid_exists(pid)
        except Exception:
            pass

    # ── Lire les métriques depuis le log ──
    metrics_history = []
    if log_file.exists():
        for line in log_file.read_text(encoding="utf-8").splitlines():
            try:
                metrics_history.append(json.loads(line))
            except Exception:
                pass

    # Lire training_state.json si disponible (split réel 70/15/15 fait par le script)
    training_state = {}
    state_file = FT_DIR / run_id / "training_state.json"
    if state_file.exists():
        try:
            training_state = json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    col_launch, col_status = st.columns([1, 2])

    with col_launch:
        st.markdown(f"**Run :** `{run_id}`")
        if training_state:
            n_tr = training_state.get("n_train", cfg.get("n_train", 0))
            n_va = training_state.get("n_val",   cfg.get("n_val",   0))
            n_te = training_state.get("n_test",  0)
            phase = training_state.get("phase", "training")
            st.markdown(f"**Dataset :** {n_tr} train · {n_va} val · {n_te} test")
            phase_icons = {"training": "🔵 Entraînement", "merging": "🔀 Merge", "testing": "🟡 Test", "validation": "🟠 Validation", "done": "✅ Terminé"}
            st.markdown(f"**Phase :** {phase_icons.get(phase, phase)}")
        else:
            n_tr = cfg.get('n_train', 0)
            n_va = cfg.get('n_val',   0)
            n_te = cfg.get('n_test',  0)
            st.markdown(f"**Dataset :** {n_tr} train · {n_va} val · {n_te} test")
        st.markdown(f"**Config :** r={cfg.get('lora_r',4)} · {cfg.get('n_epochs',3)} epochs · device={cfg.get('device','cpu')}")

        def _launch_phase(label: str):
            """Lance le subprocess finetune_xpu.py --one-phase pour la phase courante."""
            cfg_file   = FT_DIR / f"{run_id}_config.json"
            script     = ROOT / "scripts" / "finetune_xpu.py"
            stderr_log = LOGS_DIR / f"ft_{run_id}_stderr.log"
            proc = subprocess.Popen(
                [sys.executable, str(script), "--config", str(cfg_file), "--one-phase"],
                stdout=open(str(LOGS_DIR / f"ft_{run_id}_stdout.log"), "w"),
                stderr=open(str(stderr_log), "w"),
                cwd=str(ROOT),
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            pid_file.write_text(str(proc.pid))
            st.success(f"{label} (PID {proc.pid})")
            time.sleep(2)
            st.rerun()

        current_phase = training_state.get("phase", "training") if training_state else "training"

        # Entraînement terminé si completed_steps >= total_steps dans training_state.json
        _ts_done = (
            training_state.get("completed_steps", 0) > 0
            and training_state.get("total_steps", 1) > 0
            and training_state.get("completed_steps", 0) >= training_state.get("total_steps", 1)
        )

        if not is_running and current_phase == "training" and not _ts_done:
            if st.button("▶ Lancer l'entraînement", type="primary"):
                _launch_phase("Entraînement démarré")

        elif not is_running and current_phase == "training" and _ts_done:
            # Entraînement terminé mais crash pendant save/merge → relancer pour compléter
            st.warning("⚠️ Steps terminés — reprise du save/merge.")
            if st.button("▶ Compléter (save + merge)", type="primary"):
                _launch_phase("Save/merge relancé")

        elif not is_running and current_phase == "merging":
            st.success("✅ Entraînement terminé — adapter sauvegardé.")
            if st.button("▶️ Merger le modèle", type="primary"):
                _launch_phase("Merge démarré")

        elif not is_running and current_phase == "testing":
            st.success("✅ Modèle mergé — prêt pour le test.")
            if st.button("▶️ Démarrer le test", type="primary"):
                _launch_phase("Phase test démarrée")

        elif not is_running and current_phase == "validation":
            st.success("✅ Test terminé.")
            if st.button("▶️ Démarrer la validation", type="primary"):
                _launch_phase("Phase validation démarrée")

        elif is_running:
            if st.button("⏹ Arrêter l'entraînement"):
                try:
                    pid = int(pid_file.read_text())
                    import psutil
                    psutil.Process(pid).terminate()
                    pid_file.unlink(missing_ok=True)
                    st.warning("Entraînement interrompu.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
            # Barre de progression
            train_steps_running = [m for m in metrics_history if m.get("type") == "train" and m.get("step") is not None]
            if train_steps_running:
                last_r = train_steps_running[-1]
                step_r   = last_r.get("step", 0)
                total_r  = last_r.get("total_steps", 1) or 1
                pct_r    = min(step_r / total_r, 1.0)
                eta_r    = last_r.get("eta_s", 0)
                st.progress(pct_r, text=f"Step {step_r}/{total_r} — ETA {eta_r//60:.0f}m{eta_r%60:02.0f}s")
            else:
                init_entries = [m for m in metrics_history if m.get("type") == "init"]
                init_msg = init_entries[-1].get("msg", "Initialisation du modèle…") if init_entries else "Initialisation du modèle…"
                st.progress(0.0, text=init_msg)
            time.sleep(3)
            st.rerun()

        elif metrics_history:
            last = metrics_history[-1]
            train_entries = [m for m in metrics_history if m.get("type") == "train" and m.get("status") != "started"]
            if last.get("status") == "done":
                st.success("✅ Entraînement terminé !")
            elif not train_entries:
                # Crashé avant le 1er step — afficher stderr + bouton reset
                st.error("❌ Le process a crashé avant le démarrage de l'entraînement.")
                stderr_log = LOGS_DIR / f"ft_{run_id}_stderr.log"
                if stderr_log.exists() and stderr_log.stat().st_size > 0:
                    with st.expander("🔍 Détail de l'erreur"):
                        st.code(stderr_log.read_text(encoding="utf-8", errors="replace")[-3000:], language="text")
                if st.button("🔄 Réinitialiser ce run"):
                    log_file.unlink(missing_ok=True)
                    pid_file.unlink(missing_ok=True)
                    S.ft_run_id = None
                    S.ft_step   = 2
                    st.rerun()
            else:
                last_t = train_entries[-1]
                st.info(f"Dernière étape connue : step {last_t.get('step', '?')}")

    with col_status:
        if metrics_history:
            train_steps  = [m for m in metrics_history if m.get("type") == "train"]
            val_epochs   = [m for m in metrics_history if m.get("type") == "val"]

            # Métriques résumées
            m1, m2, m3, m4 = st.columns(4)
            if train_steps:
                last_t = train_steps[-1]
                with m1:
                    _loss = last_t.get("loss")
                    _loss_str = f"{_loss:.3f}" if isinstance(_loss, (int, float)) and _loss > 0 else "—"
                    st.markdown(f'<div class="metric-card"><div class="metric-val">{_loss_str}</div><div class="metric-lab">Train Loss<br>dernier step</div></div>', unsafe_allow_html=True)
                with m2:
                    elapsed = last_t.get("elapsed_s", 0)
                    st.markdown(f'<div class="metric-card"><div class="metric-val">{elapsed//60:.0f}m</div><div class="metric-lab">Temps écoulé</div></div>', unsafe_allow_html=True)
                with m3:
                    step_n = last_t.get("step", 0)
                    total  = last_t.get("total_steps", 1)
                    pct    = int(step_n / max(total, 1) * 100)
                    st.markdown(f'<div class="metric-card"><div class="metric-val">{pct}%</div><div class="metric-lab">Progression<br>step {step_n}/{total}</div></div>', unsafe_allow_html=True)
                with m4:
                    remaining = last_t.get("eta_s", 0)
                    st.markdown(f'<div class="metric-card"><div class="metric-val">{remaining//60:.0f}m</div><div class="metric-lab">ETA</div></div>', unsafe_allow_html=True)

            if train_steps:
                st.markdown("**Courbe de loss (train)**")
                import pandas as pd
                df_train = pd.DataFrame(train_steps)[["step", "loss"]].dropna()
                st.line_chart(df_train.set_index("step"), height=180)

            if val_epochs:
                st.markdown("**Loss validation par epoch**")
                df_val = pd.DataFrame(val_epochs)[["epoch", "val_loss"]].dropna()
                st.bar_chart(df_val.set_index("epoch"), height=140)

            # Résultats test / validation si disponibles
            test_res = training_state.get("test_results")
            val_res  = training_state.get("val_results")
            if test_res or val_res:
                st.markdown("**Résultats d'évaluation**")
                rc1, rc2 = st.columns(2)
                if test_res:
                    with rc1:
                        st.markdown(f'<div class="metric-card"><div class="metric-val">{test_res.get("field_f1", 0):.3f}</div><div class="metric-lab">F1 — Test<br>({test_res.get("n_evaluated",0)} ex.)</div></div>', unsafe_allow_html=True)
                if val_res:
                    with rc2:
                        st.markdown(f'<div class="metric-card"><div class="metric-val">{val_res.get("field_f1", 0):.3f}</div><div class="metric-lab">F1 — Validation<br>({val_res.get("n_evaluated",0)} ex.)</div></div>', unsafe_allow_html=True)
        else:
            # Vérifier si un crash stderr est disponible
            stderr_log = LOGS_DIR / f"ft_{run_id}_stderr.log"
            if stderr_log.exists() and stderr_log.stat().st_size > 0:
                st.error("❌ L'entraînement a crashé — stderr :")
                st.code(stderr_log.read_text(encoding="utf-8", errors="replace")[-3000:], language="text")
            else:
                st.markdown('<div class="callout-info">Les métriques s\'afficheront ici dès le démarrage de l\'entraînement.</div>', unsafe_allow_html=True)

    # ── Avancer si tout terminé (phase "done") ──
    if training_state.get("phase") == "done":
        S.ft_adapter_path = training_state.get("adapter_path", "")
        if S.ft_step == 3:
            if st.button("✅ Passer à l'évaluation", type="primary"):
                S.ft_step = 4
                st.rerun()
    elif metrics_history and any(m.get("status") == "done" for m in metrics_history):
        last_done = next(m for m in reversed(metrics_history) if m.get("status") == "done")
        S.ft_adapter_path = last_done.get("adapter_path", "")
        if S.ft_step == 3:
            if st.button("✅ Passer à l'évaluation", type="primary"):
                S.ft_step = 4
                st.rerun()

    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 — Évaluation & validation
# ══════════════════════════════════════════════════════════════════════════════
if S.ft_step >= 4:
    st.markdown('<div class="section-header">📊 Étape 4 — Évaluation & validation</div>', unsafe_allow_html=True)

    eval_file = LOGS_DIR / f"ft_{S.ft_run_id}_eval.json" if S.ft_run_id else None

    if eval_file and eval_file.exists():
        eval_data = json.loads(eval_file.read_text(encoding="utf-8"))

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.markdown("**Métriques globales**")
            me1, me2, me3 = st.columns(3)
            with me1:
                v = eval_data.get("parse_success_rate", 0)
                st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{"#22c55e" if v>0.9 else "#f59e0b"};">{v:.0%}</div><div class="metric-lab">Parse<br>success rate</div></div>', unsafe_allow_html=True)
            with me2:
                v = eval_data.get("field_f1", 0)
                st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{"#22c55e" if v>0.8 else "#f59e0b"};">{v:.2f}</div><div class="metric-lab">Field F1<br>moyen</div></div>', unsafe_allow_html=True)
            with me3:
                v = eval_data.get("val_loss", 0)
                st.markdown(f'<div class="metric-card"><div class="metric-val">{v:.3f}</div><div class="metric-lab">Val Loss<br>finale</div></div>', unsafe_allow_html=True)

            # F1 par champ
            field_f1 = eval_data.get("field_f1_per_key", {})
            if field_f1:
                st.markdown("**F1 par champ :**")
                import pandas as pd
                df_f1 = pd.DataFrame({"champ": list(field_f1.keys()), "F1": list(field_f1.values())})
                st.bar_chart(df_f1.set_index("champ"), height=200)

        with col_e2:
            st.markdown("**Comparaison base vs fine-tuné**")
            base_f1 = eval_data.get("base_field_f1", None)
            ft_f1   = eval_data.get("field_f1", 0)
            if base_f1:
                delta = ft_f1 - base_f1
                color = "#22c55e" if delta > 0 else "#ef4444"
                st.markdown(
                    f'<div class="callout-ok">'
                    f'Base model F1 : <strong>{base_f1:.2f}</strong><br>'
                    f'Fine-tuné F1  : <strong>{ft_f1:.2f}</strong><br>'
                    f'Δ : <strong style="color:{color};">{delta:+.2f}</strong>'
                    f'</div>', unsafe_allow_html=True
                )

            # Exemples val
            val_examples = eval_data.get("val_examples", [])
            if val_examples:
                st.markdown("**Prédictions sur set de validation :**")
                for ex in val_examples[:3]:
                    with st.expander(f"📄 {ex.get('image','?')} — {ex.get('doc_type','?')}"):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**Attendu**")
                            st.json(ex.get("expected", {}), expanded=True)
                        with c2:
                            st.markdown("**Prédit**")
                            st.json(ex.get("predicted", {}), expanded=True)

        # Décision
        f1_ok = eval_data.get("field_f1", 0) >= 0.75
        parse_ok = eval_data.get("parse_success_rate", 0) >= 0.9
        if f1_ok and parse_ok:
            st.success("✅ Qualité suffisante — modèle validé pour conversion GGUF.")
        else:
            st.warning("⚠ Qualité insuffisante. Conseils : augmenter les exemples ou le rang r.")

    else:
        st.markdown('<div class="callout-info">L\'évaluation sera disponible après la fin de l\'entraînement.<br>Le script calcule automatiquement F1, parse rate et exemples de validation.</div>', unsafe_allow_html=True)

    if S.ft_step == 4:
        if st.button("✅ Déployer dans le Model Registry →", type="primary"):
            S.ft_step = 5
            st.rerun()

    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 5 — Deploy → MLflow Model Registry
# ══════════════════════════════════════════════════════════════════════════════
if S.ft_step >= 5:
    st.markdown('<div class="section-header">🗂️ Étape 5 — Deploy → MLflow Model Registry</div>', unsafe_allow_html=True)

    _reg_run_id   = S.ft_run_id or "run"
    _reg_state    = {}
    _reg_sf       = FT_DIR / _reg_run_id / "training_state.json"
    if _reg_sf.exists():
        try:
            _reg_state = json.loads(_reg_sf.read_text(encoding="utf-8"))
        except Exception:
            pass
    _merged_dir   = FT_DIR / _reg_run_id / "merged"
    _mlflow_run_id = _reg_state.get("mlflow_run_id", "")
    _reg_version  = _reg_state.get("mlflow_model_version")
    _tracking_uri = (LOGS_DIR / "mlruns").as_uri()

    col_r1, col_r2 = st.columns([1, 1])
    with col_r1:
        st.markdown("**Modèle source :** `merged/` (HuggingFace FP16)")
        st.markdown("**Registry name :** `qwen_vl_finetuned`")
        st.markdown("**Tracking URI :**")
        st.code(str(LOGS_DIR / "mlruns"), language="text")

        if _reg_version:
            st.success(f"✅ Déjà enregistré — version **v{_reg_version}**")
        elif not _merged_dir.exists() or not (_merged_dir / "config.json").exists():
            st.warning("⚠️ Le modèle mergé n'est pas disponible. Exécutez d'abord la phase Merge.")
        elif not _mlflow_run_id:
            st.warning("⚠️ MLflow run ID introuvable dans training_state.json.")
        else:
            stage = st.selectbox("Stage initial", ["None", "Staging", "Production"], index=1)
            model_name = st.text_input("Nom dans le registry", value="qwen_vl_finetuned")
            if st.button("🚀 Déployer dans le Model Registry", type="primary"):
                import mlflow
                try:
                    with st.spinner("Enregistrement dans le Model Registry…"):
                        mlflow.set_tracking_uri(_tracking_uri)
                        client = mlflow.MlflowClient()
                        # Créer le registered model s'il n'existe pas
                        try:
                            client.create_registered_model(
                                model_name,
                                tags={"base_model": _reg_state.get("model_id", "Qwen2.5-VL-3B-Instruct")},
                            )
                        except Exception:
                            pass  # existe déjà
                        # Créer une version pointant vers le chemin local merged/
                        # (pas de copie — référence directe au dossier)
                        mv = client.create_model_version(
                            name=model_name,
                            source=str(_merged_dir),
                            run_id=_mlflow_run_id,
                            tags={
                                "run_id":     _reg_run_id,
                                "base_model": _reg_state.get("model_id", "Qwen2.5-VL-3B-Instruct"),
                                "lora_r":     str(S.ft_config.get("lora_r", 4)),
                                "n_train":    str(_reg_state.get("n_train", 0)),
                                "field_f1":   str(_reg_state.get("test_results", {}).get("field_f1", "?")),
                            },
                        )
                        if stage != "None":
                            client.transition_model_version_stage(
                                name=model_name, version=mv.version, stage=stage
                            )
                        # Persister version dans training_state
                        _reg_state["mlflow_model_version"] = mv.version
                        _reg_sf.write_text(json.dumps(_reg_state, indent=2, ensure_ascii=False), encoding="utf-8")
                        st.success(f"✅ Enregistré : **{model_name} v{mv.version}** → {stage}")
                        st.rerun()
                except Exception as _e:
                    st.error(f"Erreur : {_e}")

    with col_r2:
        st.markdown("**Workflow réel (production)**")
        st.markdown("""
```
Training ──→ merged/
                │
                ▼
         MLflow Registry
         qwen_vl_finetuned
          ├── Staging  (test A/B)
          └── Production ←── serving charge ici
                │
                ▼
         mlflow models serve
         # ou vLLM / TorchServe
         # → GET /invocations
```
Pour llama-server (GGUF) :
```
Registry (HF FP16)
    ↓ pull + convert
convert_hf_to_gguf.py
    ↓
llama-server --model *.gguf
```
        """)
        st.markdown("**Ouvrir MLflow UI :**")
        st.code(f"mlflow ui --backend-store-uri {LOGS_DIR / 'mlruns'}", language="bash")

    if S.ft_step == 5:
        if st.button("✅ Passer à la conversion GGUF →", type="primary"):
            S.ft_step = 6
            st.rerun()

    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 6 — Conversion GGUF
# ══════════════════════════════════════════════════════════════════════════════
if S.ft_step >= 6:
    st.markdown('<div class="section-header">📦 Étape 6 — Conversion GGUF</div>', unsafe_allow_html=True)

    run_id         = S.ft_run_id or "run"
    merged_dir     = FT_DIR / run_id / "merged"
    gguf_dir       = FT_DIR / run_id / "gguf"
    gguf_f16       = gguf_dir / f"qwen3b_ft_{run_id}_f16.gguf"
    gguf_q4        = gguf_dir / f"qwen3b_ft_{run_id}_q4_k_m.gguf"
    convert_script = ROOT.parent / "llama.cpp" / "convert_hf_to_gguf.py"
    # llama-quantize binary (optionnel — nécessite build cmake)
    quantize_bin   = ROOT.parent / "llama.cpp" / "build" / "bin" / "llama-quantize.exe"

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown(f"**Source :** `finetuning/{run_id}/merged/`")

        if not convert_script.exists():
            st.error("⚠️ llama.cpp introuvable — cloner le repo d'abord :")
            st.code(
                f"cd {ROOT.parent}\n"
                "git clone https://github.com/ggerganov/llama.cpp.git\n"
                "pip install -r llama.cpp/requirements.txt",
                language="bash",
            )
            st.caption(f"Chemin attendu : `{convert_script}`")

        # ── Étape 1 : HF → GGUF f16 ──────────────────────────────────────
        st.markdown("**Étape 1 — Conversion HF → GGUF f16** (~6 GB)")
        if gguf_f16.exists():
            st.success(f"✅ `{gguf_f16.name}` ({gguf_f16.stat().st_size/1e9:.1f} GB)")
        elif merged_dir.exists() and convert_script.exists():
            if st.button("▶ Convertir en f16", type="primary", key="btn_convert_f16"):
                gguf_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable, str(convert_script),
                    str(merged_dir),
                    "--outtype", "f16",
                    "--outfile", str(gguf_f16),
                ]
                with st.spinner("Conversion f16 en cours (~5–10 min)…"):
                    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
                if result.returncode == 0:
                    S.ft_gguf_path = str(gguf_f16)
                    st.success(f"✅ GGUF f16 créé : {gguf_f16.name}")
                    st.rerun()
                else:
                    st.error("Erreur conversion :")
                    st.code(result.stderr[-2000:])
        elif not merged_dir.exists():
            st.markdown('<div class="callout-warn">Le répertoire merged/ sera créé automatiquement à la fin du merge.</div>', unsafe_allow_html=True)

        # ── Étape 2 : f16 → q4_k_m (optionnel, nécessite llama-quantize) ──
        st.markdown("**Étape 2 — Quantification q4_k_m** (~1.9 GB, optionnel)")
        if gguf_q4.exists():
            st.success(f"✅ `{gguf_q4.name}` ({gguf_q4.stat().st_size/1e9:.1f} GB)")
            S.ft_gguf_path = str(gguf_q4)
        elif gguf_f16.exists():
            if quantize_bin.exists():
                if st.button("▶ Quantifier en q4_k_m", key="btn_quantize"):
                    cmd_q = [str(quantize_bin), str(gguf_f16), str(gguf_q4), "q4_k_m"]
                    with st.spinner("Quantification en cours (~2–5 min)…"):
                        result_q = subprocess.run(cmd_q, capture_output=True, text=True)
                    if result_q.returncode == 0:
                        S.ft_gguf_path = str(gguf_q4)
                        st.success(f"✅ {gguf_q4.name}")
                        st.rerun()
                    else:
                        st.error("Erreur quantification :")
                        st.code(result_q.stderr[-1000:])
            else:
                st.warning("💡 `llama-quantize.exe` non trouvé — télécharger les binaires pré-compilés :")
                st.caption(f"Attendu : `{quantize_bin}`")
                if st.button("⬇️ Télécharger llama-quantize (Windows x64, pré-compilé)", key="btn_dl_quantize"):
                    import urllib.request, zipfile, io as _io
                    with st.spinner("Téléchargement des binaires llama.cpp depuis GitHub Releases…"):
                        try:
                            _api_url = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
                            with urllib.request.urlopen(_api_url, timeout=15) as _r:
                                _release = json.loads(_r.read())
                            # Asset Windows sans CUDA (AVX2 = compatible la plupart des CPU/GPU)
                            _asset = next(
                                (a for a in _release["assets"]
                                 if "win" in a["name"].lower() and "avx2" in a["name"].lower()
                                 and a["name"].endswith(".zip")),
                                None,
                            ) or next(
                                (a for a in _release["assets"]
                                 if "win" in a["name"].lower() and a["name"].endswith(".zip")),
                                None,
                            )
                            if not _asset:
                                st.error("Aucun binaire Windows trouvé dans la dernière release.")
                            else:
                                st.info(f"Asset : `{_asset['name']}` ({_asset['size']//1024//1024} MB) — téléchargement…")
                                with urllib.request.urlopen(_asset["browser_download_url"], timeout=120) as _r:
                                    _zip_data = _r.read()
                                _dest = ROOT.parent / "llama.cpp" / "build" / "bin"
                                _dest.mkdir(parents=True, exist_ok=True)
                                with zipfile.ZipFile(_io.BytesIO(_zip_data)) as _z:
                                    _extracted = [n for n in _z.namelist() if "llama-quantize" in n and n.endswith(".exe")]
                                    if _extracted:
                                        _z.extract(_extracted[0], str(_dest))
                                        # Si extrait dans un sous-dossier, remonter
                                        _found = list(_dest.rglob("llama-quantize.exe"))
                                        if _found and _found[0] != _dest / "llama-quantize.exe":
                                            import shutil as _sh
                                            _sh.move(str(_found[0]), str(_dest / "llama-quantize.exe"))
                                if quantize_bin.exists():
                                    st.success("✅ `llama-quantize.exe` installé — rechargez la page.")
                                    st.rerun()
                                else:
                                    st.error("Extraction échouée — `llama-quantize.exe` non trouvé dans le zip.")
                        except Exception as _e:
                            st.error(f"Erreur téléchargement : {_e}")
        else:
            st.caption("Disponible après l'étape 1.")

    with col_g2:
        # Afficher le meilleur GGUF disponible
        _best_gguf = gguf_q4 if gguf_q4.exists() else (gguf_f16 if gguf_f16.exists() else None)
        if _best_gguf:
            size_gb = _best_gguf.stat().st_size / 1e9
            st.markdown(f'<div class="callout-ok">✓ GGUF disponible<br><strong>{_best_gguf.name}</strong><br>Taille : {size_gb:.2f} GB</div>', unsafe_allow_html=True)
            st.markdown("**Utiliser avec llama-server :**")
            st.code(f"""cp finetuning/{run_id}/gguf/{_best_gguf.name} models/
pwsh scripts/start_server.ps1""", language="bash")

    if S.ft_step == 6:
        if st.button("✅ Passer à la mise en service", type="primary"):
            S.ft_step = 7
            st.rerun()

    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 7 — Servir le modèle dans un repo
# ══════════════════════════════════════════════════════════════════════════════
if S.ft_step >= 7:
    st.markdown('<div class="section-header">🗄️ Étape 7 — Servir le modèle fine-tuné</div>', unsafe_allow_html=True)

    run_id    = S.ft_run_id or "run"
    gguf_path = Path(S.ft_gguf_path) if S.ft_gguf_path else None
    if not gguf_path or not gguf_path.exists():
        # chercher dans finetuning/
        candidates = list((FT_DIR / run_id / "gguf").glob("*.gguf")) if (FT_DIR / run_id / "gguf").exists() else []
        if candidates:
            gguf_path = candidates[0]

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("#### Destination du modèle")
        dest_options = {
            "served_models/ (repo dédié)": str(SERVED_DIR / run_id),
            "models/ (remplacer modèle actuel)": str(MODELS_DIR),
        }
        dest_label = st.radio("Où servir le modèle ?", list(dest_options.keys()))
        dest_dir   = Path(dest_options[dest_label])

        st.markdown("#### Métadonnées du run")
        model_version = st.text_input("Version", f"v1.0-{run_id}")
        model_notes   = st.text_area("Notes / contexte", f"Fine-tuning LoRA r=4 sur {S.ft_n_train} exemples · XPU · {datetime.now().strftime('%Y-%m-%d')}", height=80)

        if gguf_path and gguf_path.exists():
            if st.button("🚀 Publier le modèle dans le repo", type="primary"):
                dest_dir.mkdir(parents=True, exist_ok=True)
                dst_gguf = dest_dir / gguf_path.name
                shutil.copy2(str(gguf_path), str(dst_gguf))

                # Copier mmproj si présent
                mmproj_src = list(MODELS_DIR.glob("mmproj*.gguf"))
                if mmproj_src:
                    shutil.copy2(str(mmproj_src[0]), str(dest_dir / mmproj_src[0].name))

                # Écrire manifest.json
                manifest = {
                    "run_id": run_id,
                    "version": model_version,
                    "notes": model_notes,
                    "published_at": datetime.now().isoformat(),
                    "gguf_file": gguf_path.name,
                    "gguf_path": str(dst_gguf),
                    "base_model": "Qwen2.5-VL-3B-Instruct",
                    "lora_config": {
                        "r": S.ft_config.get("lora_r", 4),
                        "alpha": S.ft_config.get("lora_alpha", 8),
                        "target_modules": S.ft_config.get("target_modules", []),
                    },
                    "dataset": {
                        "n_train": S.ft_n_train,
                        "n_val": S.ft_n_val,
                        "path": S.ft_dataset_path,
                    },
                    "size_gb": round(gguf_path.stat().st_size / 1e9, 2),
                }
                (dest_dir / "manifest.json").write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                st.success(f"✅ Modèle publié dans `{dest_dir}`")
                st.balloons()
        else:
            st.markdown('<div class="callout-warn">Aucun GGUF disponible — complétez l\'étape 5 d\'abord.</div>', unsafe_allow_html=True)

    with col_s2:
        st.markdown("#### Repo served_models/")
        served_runs = [d for d in SERVED_DIR.iterdir() if d.is_dir()] if SERVED_DIR.exists() else []
        if served_runs:
            for run_dir in sorted(served_runs, reverse=True):
                manifest_file = run_dir / "manifest.json"
                if manifest_file.exists():
                    m = json.loads(manifest_file.read_text(encoding="utf-8"))
                    with st.expander(f"📦 {m.get('version','?')} — {m.get('run_id','?')} · {m.get('size_gb','?')} GB"):
                        st.markdown(f"**Publié :** {m.get('published_at','?')[:16]}")
                        st.markdown(f"**Notes :** {m.get('notes','')}")
                        st.markdown(f"**Dataset :** {m.get('dataset',{}).get('n_train','?')} exemples")
                        gguf_f = Path(m.get("gguf_path", ""))
                        if gguf_f.exists():
                            if st.button(f"⚡ Activer ce modèle dans models/", key=f"activate_{run_dir.name}"):
                                dst = MODELS_DIR / gguf_f.name
                                shutil.copy2(str(gguf_f), str(dst))
                                st.success(f"✅ {gguf_f.name} copié dans models/ — relancez llama-server.")
                                st.page_link("pages/02_Pipeline.py", label="Aller au pipeline →")
        else:
            st.info("Aucun modèle publié pour l'instant.")

        st.markdown("#### Commande de démarrage")
        st.code(f"""# Servir le modèle fine-tuné
pwsh scripts/start_server.ps1 -Backend cpu

# Vérifier qu'il répond
curl http://localhost:8080/v1/models""", language="bash")

    # ── Réinitialiser ──
    st.divider()
    if st.button("↩ Nouveau fine-tuning"):
        for k in list(st.session_state.keys()):
            if k.startswith("ft_"):
                del st.session_state[k]
        _init()
        st.rerun()

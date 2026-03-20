"""
Streamlit UI — Document Intelligence with Qwen2.5-VL
=====================================================
Process flow (5 steps, visible in sidebar):
  1. Configuration  — server URL, model params
  2. Upload         — drag-drop image or PDF
  3. Classification — detect document type
  4. Extraction     — structured field/table extraction
  5. Results        — JSON, CSV export, history
"""
from __future__ import annotations
import base64
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

# ── styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Step badges */
.step-badge {
    display: inline-block;
    background: #e0e7ff;
    color: #3730a3;
    border-radius: 50%;
    width: 28px; height: 28px;
    line-height: 28px;
    text-align: center;
    font-weight: 700;
    font-size: 13px;
    margin-right: 8px;
}
.step-badge.done  { background: #d1fae5; color: #065f46; }
.step-badge.active{ background: #3730a3; color: #fff; }
.step-badge.error { background: #fee2e2; color: #991b1b; }

/* KV grid */
.kv-grid { display: grid; grid-template-columns: 160px 1fr; gap: 6px 12px; }
.kv-key  { font-size: 12px; color: #6b7280; font-weight: 600; text-transform: uppercase; }
.kv-val  { font-size: 14px; color: #111827; }

/* confidence bar */
.conf-bar-wrap { background:#e5e7eb; border-radius:6px; height:10px; width:100%; }
.conf-bar      { background:#4f46e5; border-radius:6px; height:10px; }

/* doc type chip */
.chip {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 9999px;
    font-size: 13px;
    font-weight: 600;
    background: #ede9fe;
    color: #5b21b6;
    margin-right: 6px;
}
.chip.financial { background:#dbeafe; color:#1e40af; }
.chip.legal     { background:#fef3c7; color:#92400e; }

/* status pill */
.status-ok  { background:#d1fae5; color:#065f46; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; }
.status-err { background:#fee2e2; color:#991b1b; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; }
.status-off { background:#f3f4f6; color:#6b7280; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ── Persistent state file (survit navigation + rechargement Streamlit) ────────
_STATE_FILE = ROOT / "logs" / "pipeline_state.json"
_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

def _load_persistent_state() -> dict:
    if _STATE_FILE.exists():
        try:
            return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def _save_persistent_state():
    try:
        state = {
            "classify_result": st.session_state.get("classify_result"),
            "extract_result":  st.session_state.get("extract_result"),
            "history":         st.session_state.get("history", []),
            "raw_response":    st.session_state.get("raw_response", ""),
            "step":            st.session_state.get("step", 1),
        }
        _STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

# Session state
# ═══════════════════════════════════════════════════════════════════════════════
def _init_state():
    persisted = _load_persistent_state()
    defaults = {
        "step":            1,  # toujours démarrer à la Configuration (step persisté ignoré)
        "server_ok":       False,
        "uploaded_images": [],
        "current_idx":     0,
        "classify_result": persisted.get("classify_result"),
        "extract_result":  persisted.get("extract_result"),
        "history":         persisted.get("history", []),
        "raw_response":    persisted.get("raw_response", ""),
        "processing":      False,
        "_mlflow_run_id":  None,
        "classify_tokens": {},
        "extract_tokens":  {},
        "current_img_meta": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img_rgb = img.convert("RGB") if img.mode != "RGB" else img
    img_rgb.save(buf, format="JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _resize(img: Image.Image, max_dim: int = 1024) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    scale = max_dim / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def _check_server(url: str, api_key: str) -> tuple[bool, str]:
    """Returns (ok, diagnostic_message)."""
    import urllib.parse, socket
    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 80
        # 1. TCP reachability first (fast, clear error)
        s = socket.create_connection((host, port), timeout=2)
        s.close()
    except socket.timeout:
        return False, f"Timeout — le serveur ne répond pas sur {host}:{port}"
    except ConnectionRefusedError:
        return False, (
            f"Connexion refusée sur {host}:{port} — le serveur n'est pas démarré.\n\n"
            "**Lancez dans un terminal :**\n"
            "```powershell\npwsh scripts/start_server.ps1\n```"
        )
    except OSError as e:
        return False, f"Erreur réseau : {e}"
    try:
        from openai import OpenAI
        c = OpenAI(api_key=api_key, base_url=url)
        c.models.list()
        return True, "OK"
    except Exception as e:
        err = str(e)
        if "404" in err:
            return False, f"Port {port} actif mais `/v1/models` introuvable — vérifiez que c'est bien un serveur llama.cpp (et non une autre app)."
        if "401" in err or "403" in err:
            return False, "Authentification refusée — essayez n'importe quelle valeur non-vide pour l'API key."
        return False, f"Serveur TCP accessible mais erreur API : {err[:200]}"


def _call_vlm(img: Image.Image, system: str, prompt: str, url: str, api_key: str,
              model: str, temperature: float, max_tokens: int) -> tuple[str, int, dict]:
    """Returns (content, latency_ms, token_info)."""
    from openai import OpenAI
    image_b64 = _pil_to_b64(_resize(img))
    client = OpenAI(api_key=api_key, base_url=url)
    t0 = time.monotonic()
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_b64}},
                {"type": "text", "text": prompt},
            ]},
        ],
    )
    latency_ms = int((time.monotonic() - t0) * 1000)
    usage = resp.usage
    tokens = {
        "prompt_tokens":     getattr(usage, "prompt_tokens",     0) if usage else 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
        "total_tokens":      getattr(usage, "total_tokens",      0) if usage else 0,
    }
    return resp.choices[0].message.content or "", latency_ms, tokens


def _parse_json(text: str) -> dict:
    import re
    m = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if m:
        text = m.group(1)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m2 = re.search(r"\{[\s\S]+\}", text)
        if m2:
            return json.loads(m2.group())
        raise ValueError(f"Cannot parse JSON:\n{text[:400]}")


# ── Correction helpers ────────────────────────────────────────────────────────
_CORRECTIONS_FILE = ROOT / "logs" / "classification_corrections.json"
_ALL_DOC_TYPES = [
    "invoice", "contract", "bank_statement", "report", "form",
    "receipt", "id_document", "letter", "cv", "presentation", "other"
]

def _load_corrections() -> list:
    if _CORRECTIONS_FILE.exists():
        try:
            return json.loads(_CORRECTIONS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []

def _save_correction(filename: str, predicted: str, corrected: str):
    corrections = _load_corrections()
    corrections = [c for c in corrections
                   if not (c["filename"] == filename and c["corrected"] == corrected)]
    corrections.append({
        "filename": filename,
        "predicted": predicted,
        "corrected": corrected,
        "timestamp": datetime.utcnow().isoformat(),
    })
    _CORRECTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CORRECTIONS_FILE.write_text(
        json.dumps(corrections, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def _build_classify_prompt() -> str:
    types_str = "|".join(_ALL_DOC_TYPES)
    prompt = f"""Classify this document.

Return JSON with exactly these fields:
{{
  "document_type": "<{types_str}>",
  "category": "<financial|legal|administrative|technical|personal|other>",
  "confidence": <float 0.0-1.0>,
  "language": "<fr|en|de|es|other>",
  "notes": "<short observation>"
}}"""
    corrections = _load_corrections()
    if corrections:
        seen: set = set()
        examples = []
        for c in reversed(corrections):
            key = (c["predicted"], c["corrected"])
            if key not in seen:
                seen.add(key)
                examples.append(c)
            if len(examples) >= 6:
                break
        if examples:
            ex_lines = "\n".join(
                f'- predicted "{e["predicted"]}" but correct type was "{e["corrected"]}"'
                for e in examples
            )
            prompt += f"\n\nPast corrections to avoid repeating mistakes:\n{ex_lines}"
    return prompt


def _img_info(img: Image.Image) -> dict:
    """Returns resolution, estimated JPEG size, megapixels, complexity label."""
    w, h = img.size
    buf = io.BytesIO()
    img_rgb = img.convert("RGB") if img.mode != "RGB" else img
    img_rgb.save(buf, format="JPEG", quality=92)
    size_kb = round(buf.tell() / 1024, 1)
    mp = round(w * h / 1_000_000, 2)
    if mp > 4:
        complexity = "very_high_res"
    elif mp > 1.5:
        complexity = "high_res"
    elif mp > 0.3:
        complexity = "standard"
    else:
        complexity = "low_res"
    return {"width": w, "height": h, "size_kb": size_kb, "megapixels": mp, "complexity": complexity}


def _confidence_html(conf: float) -> str:
    pct = int(conf * 100)
    color = "#4f46e5" if conf > 0.7 else "#f59e0b" if conf > 0.4 else "#ef4444"
    return (
        f'<div class="conf-bar-wrap"><div class="conf-bar" '
        f'style="width:{pct}%; background:{color};"></div></div>'
        f'<span style="font-size:13px;color:{color};font-weight:600;">{pct}%</span>'
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar — navigation + config
# ═══════════════════════════════════════════════════════════════════════════════
STEPS = [
    (1, "Configuration",  "⚙️"),
    (2, "Upload",         "📄"),
    (3, "Classification", "🏷️"),
    (4, "Extraction",     "🔬"),
    (5, "Résultats",      "📊"),
]

with st.sidebar:
    st.markdown("## 🔍 Document Intelligence")
    st.markdown("**Qwen2.5-VL — Pipeline local**")
    st.divider()

    st.markdown("### Étapes")
    for num, label, icon in STEPS:
        current = st.session_state.step
        if num < current:
            badge_cls = "done"
            badge_icon = "✓"
        elif num == current:
            badge_cls = "active"
            badge_icon = str(num)
        else:
            badge_cls = ""
            badge_icon = str(num)

        active_style = "font-weight:700;" if num == current else "color:#6b7280;"
        st.markdown(
            f'<div style="{active_style} margin: 4px 0;">'
            f'<span class="step-badge {badge_cls}">{badge_icon}</span>'
            f'{icon} {label}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Server status — auto TCP probe if not already confirmed
    if not st.session_state.server_ok:
        import socket as _socket, urllib.parse as _urlparse
        _srv_url = st.session_state.get("cfg_url", "http://localhost:8080/v1")
        try:
            _p = _urlparse.urlparse(_srv_url)
            _sock = _socket.create_connection((_p.hostname or "localhost", _p.port or 8080), timeout=1)
            _sock.close()
            st.session_state.server_ok = True
        except Exception:
            pass
    if st.session_state.server_ok:
        st.markdown('<span class="status-ok">● Serveur connecté :8080</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-off">● Serveur hors ligne :8080</span>', unsafe_allow_html=True)
        _backend = st.radio(
            "Backend",
            options=["auto", "vulkan", "cpu"],
            index=0,
            horizontal=True,
            help="vulkan = Intel Arc GPU · cpu = CPU uniquement",
            key="llama_backend",
            label_visibility="collapsed",
        )
        _backend_labels = {"auto": "🔍 auto", "vulkan": "⚡ Vulkan (Intel Arc)", "cpu": "🖥 CPU"}
        st.caption(_backend_labels.get(_backend, _backend))

        if st.button("▶ Démarrer llama.cpp", use_container_width=True):
            import subprocess as _sp
            _script = ROOT / "scripts" / "start_server.ps1"
            try:
                _args = f"'-NoExit', '-File', '{_script}', '-Backend', '{_backend}'"
                _sp.Popen(
                    ["powershell", "-Command",
                     f"Start-Process pwsh -ArgumentList {_args} -WorkingDirectory '{ROOT}'"],
                    creationflags=_sp.CREATE_NO_WINDOW,
                )
                _icon = "⚡" if _backend == "vulkan" else "🖥" if _backend == "cpu" else "⏳"
                st.toast(f"Démarrage [{_backend}] — chargement ~60s", icon=_icon)
            except Exception as _e:
                st.error(f"Impossible de lancer le serveur : {_e}")

    if st.session_state.uploaded_images:
        st.markdown(f"**Documents:** {len(st.session_state.uploaded_images)} chargé(s)")

    st.divider()

    # ── Monitoring temps réel ─────────────────────────────────────────────────
    st.markdown("### 📊 Monitoring")
    col_ref, _ = st.columns([1, 1])
    with col_ref:
        if st.button("↻", use_container_width=True, help="Rafraîchir les métriques"):
            st.rerun()

    try:
        from src.utils.monitoring import get_metrics, color_for_pct
        _m = get_metrics()

        # CPU
        _cpu_c = color_for_pct(_m.cpu_percent)
        st.markdown(
            f'<span style="font-size:12px;font-weight:600;">CPU</span> '
            f'<span style="color:{_cpu_c};font-weight:700;">{_m.cpu_percent:.0f}%</span>'
            + (f' <span style="font-size:10px;color:#9ca3af;">{_m.cpu_freq_mhz:.0f} MHz</span>' if _m.cpu_freq_mhz else ""),
            unsafe_allow_html=True,
        )
        st.progress(min(_m.cpu_percent / 100, 1.0))

        # RAM
        _ram_c = color_for_pct(_m.ram_percent)
        st.markdown(
            f'<span style="font-size:12px;font-weight:600;">RAM</span> '
            f'<span style="color:{_ram_c};font-weight:700;">{_m.ram_percent:.0f}%</span>'
            f' <span style="font-size:10px;color:#9ca3af;">{_m.ram_used_gb:.1f}/{_m.ram_total_gb:.1f} GB</span>',
            unsafe_allow_html=True,
        )
        st.progress(min(_m.ram_percent / 100, 1.0))

        # GPU Intel Arc
        if _m.gpu_name:
            _gpu_label = _m.gpu_name.replace("Intel(R) ", "").replace("(TM) ", "")
            _gpu_c = color_for_pct(_m.gpu_util_percent)
            st.markdown(
                f'<span style="font-size:12px;font-weight:600;">GPU</span> '
                f'<span style="color:{_gpu_c};font-weight:700;">{_m.gpu_util_percent:.0f}%</span>'
                f' <span style="font-size:10px;color:#9ca3af;">{_gpu_label}</span>',
                unsafe_allow_html=True,
            )
            st.progress(min(_m.gpu_util_percent / 100, 1.0))
            if _m.gpu_mem_used_mb > 0:
                st.caption(f"VRAM: {_m.gpu_mem_used_mb:.0f} MB utilisés")

        # NPU Intel AI Boost
        if _m.npu_present:
            st.markdown(
                f'<span style="font-size:12px;font-weight:600;color:#8b5cf6;">NPU</span> '
                f'<span style="font-size:10px;color:#8b5cf6;">AI Boost {_m.npu_tops} TOPS ✓</span>',
                unsafe_allow_html=True,
            )
            st.caption(_m.npu_name)

    except Exception as _e:
        st.caption(f"Monitoring: {_e}")

    st.divider()

    if st.button("↩ Recommencer", use_container_width=True):
        for k in ["step","uploaded_images","classify_result","extract_result","raw_response",
                  "current_idx","_mlflow_run_id","classify_tokens","extract_tokens","current_img_meta"]:
            if k in ("step",):
                st.session_state[k] = 1
            elif k in ("uploaded_images",):
                st.session_state[k] = []
            elif k in ("classify_tokens","extract_tokens","current_img_meta"):
                st.session_state[k] = {}
            else:
                st.session_state[k] = None
        _save_persistent_state()
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Configuration
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.step == 1:
    st.markdown("## ⚙️ Étape 1 — Configuration du serveur")
    st.markdown(
        "Configurez la connexion au serveur **llama.cpp** qui héberge Qwen2.5-VL.  \n"
        "Démarrez le serveur si ce n'est pas encore fait :"
    )

    with st.expander("📋 Commandes de démarrage du serveur", expanded=False):
        st.code("pwsh poc_qwen/scripts/start_server.ps1", language="powershell")
        st.code("bash poc_qwen/scripts/start_server.sh", language="bash")
        st.code(
            "# Ou directement avec llama-cpp-python:\n"
            "python -m llama_cpp.server --model models/qwen2.5-vl-7b-instruct-q4_k_m.gguf "
            "--n_gpu_layers 35 --port 8080",
            language="bash",
        )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Connexion")
        server_url = st.text_input(
            "URL du serveur",
            value=os.getenv("MODEL_SERVER_URL", "http://localhost:8080/v1"),
            help="URL de base OpenAI-compatible (llama.cpp)",
        )
        api_key = st.text_input(
            "API Key",
            value=os.getenv("OPENAI_API_KEY", "local-llama-key"),
            type="password",
            help="Valeur quelconque — llama.cpp accepte tout",
        )
        model_name = st.text_input(
            "Nom du modèle",
            value=os.getenv("MODEL_NAME", "qwen2.5-vl-7b-q4_k_m"),
        )

    with col2:
        st.markdown("### Paramètres d'inférence")
        temperature = st.slider("Température", 0.0, 1.0, 0.1, 0.05,
                                help="0.1 = déterministe (recommandé pour JSON)")
        max_tokens  = st.slider("Max tokens", 256, 4096, 2048, 256)
        st.markdown(
            "**Recommandations:**\n"
            "- Température ≤ 0.15 pour extraction JSON\n"
            "- Max tokens ≥ 1024 pour documents complexes"
        )

    st.divider()

    col_test, col_next = st.columns([1, 2])
    with col_test:
        if st.button("🔌 Tester la connexion", use_container_width=True):
            with st.spinner("Test en cours…"):
                ok, diag = _check_server(server_url, api_key)
            st.session_state.server_ok = ok
            if ok:
                st.success("✓ Serveur accessible et opérationnel")
            else:
                st.error("✗ Serveur inaccessible")
                st.markdown(diag)
                with st.expander("💡 Comment démarrer le serveur", expanded=True):
                    st.markdown(
                        "**Option 1 — Script automatique (recommandé) :**\n"
                        "```powershell\n# Dans un nouveau terminal, depuis poc_qwen/\n"
                        "pwsh scripts/start_server.ps1\n```\n\n"
                        "**Option 2 — Python direct :**\n"
                        "```powershell\npython -m llama_cpp.server \\\n"
                        "  --model models/qwen2.5-vl-7b-instruct-q4_k_m.gguf \\\n"
                        "  --n_gpu_layers 35 \\\n"
                        "  --n_ctx 8192 \\\n"
                        "  --port 8080\n```\n\n"
                        "**Option 3 — Pas de modèle encore ?**\n"
                        "```powershell\npython scripts/download_model.py\n```\n\n"
                        "> Attendez le message `llama server listening at http://0.0.0.0:8080` "
                        "puis retestez la connexion."
                    )

    with col_next:
        btn_label = "Continuer sans vérification →" if not st.session_state.server_ok else "Suivant : Upload →"
        if st.button(btn_label, type="primary", use_container_width=True):
            # Store config in session
            st.session_state["cfg_url"]         = server_url
            st.session_state["cfg_api_key"]     = api_key
            st.session_state["cfg_model"]       = model_name
            st.session_state["cfg_temperature"] = temperature
            st.session_state["cfg_max_tokens"]  = max_tokens
            st.session_state.step = 2
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Upload
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 2:
    st.markdown("## 📄 Étape 2 — Chargement des documents")
    st.markdown("Glissez-déposez des images ou des PDFs. Chaque page sera traitée séparément.")

    uploaded = st.file_uploader(
        "Documents (jpg, png, pdf)",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    _PDF_MAX_MB = 10
    _PDF_MAX_BYTES = _PDF_MAX_MB * 1024 * 1024

    images: list[tuple[str, Image.Image]] = []

    if uploaded:
        for f in uploaded:
            if f.type == "application/pdf":
                pdf_bytes = f.read()
                if len(pdf_bytes) > _PDF_MAX_BYTES:
                    st.error(
                        f"⚠ PDF trop volumineux : **{f.name}** "
                        f"({len(pdf_bytes)/1024/1024:.1f} MB > {_PDF_MAX_MB} MB max).  \n"
                        "Compressez le PDF ou réduisez sa résolution avant de le charger."
                    )
                    continue
                try:
                    from src.utils.image_utils import pdf_bytes_to_images
                    pages = pdf_bytes_to_images(pdf_bytes, dpi=200)
                    for i, page in enumerate(pages):
                        images.append((f"{f.name} — p.{i+1}", page.convert("RGB")))
                    st.success(f"PDF '{f.name}' — {len(pages)} page(s) extraite(s)")
                except Exception as e:
                    st.error(f"Erreur PDF {f.name}: {e}")
            else:
                img_bytes = f.read()
                if len(img_bytes) > _PDF_MAX_BYTES:
                    st.error(
                        f"⚠ Image trop volumineuse : **{f.name}** "
                        f"({len(img_bytes)/1024/1024:.1f} MB > {_PDF_MAX_MB} MB max).  \n"
                        "Réduisez la résolution ou compressez l'image avant de la charger."
                    )
                    continue
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                images.append((f.name, img))

    if images:
        st.markdown(f"**{len(images)} image(s) prête(s)**")
        cols = st.columns(min(len(images), 4))
        for i, (name, img) in enumerate(images[:8]):
            with cols[i % 4]:
                st.image(img, caption=name, use_container_width=True)
        if len(images) > 8:
            st.caption(f"… et {len(images)-8} autre(s) non affichée(s)")

        st.divider()
        col_back, col_next = st.columns(2)
        with col_back:
            if st.button("← Configuration"):
                st.session_state.step = 1
                st.rerun()
        with col_next:
            if st.button("Suivant : Classification →", type="primary", use_container_width=True):
                st.session_state.uploaded_images = images
                st.session_state.current_idx = 0
                # Reset previous results so old notes/type don't bleed into new doc
                st.session_state.classify_result  = None
                st.session_state.extract_result   = None
                st.session_state.raw_response     = ""
                st.session_state.classify_tokens  = {}
                st.session_state.extract_tokens   = {}
                st.session_state.current_img_meta = {}
                st.session_state["_mlflow_run_id"] = None
                st.session_state.step = 3
                st.rerun()
    else:
        st.info("Chargez au moins un document pour continuer.")
        if st.button("← Configuration"):
            st.session_state.step = 1
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Classification
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 3:
    st.markdown("## 🏷️ Étape 3 — Classification du document")
    st.markdown(
        "Le modèle analyse l'image et détermine le **type de document** "
        "(facture, contrat, relevé bancaire…) avant l'extraction des données."
    )

    images = st.session_state.uploaded_images
    if not images:
        st.warning("Aucun document chargé. Retournez à l'étape Upload.")
        if st.button("← Upload"):
            st.session_state.step = 2
            st.rerun()
        st.stop()
    idx = min(st.session_state.current_idx, len(images) - 1)
    st.session_state.current_idx = idx
    name, img = images[idx]

    # Document selector if multiple
    if len(images) > 1:
        names = [n for n, _ in images]
        sel = st.selectbox("Document sélectionné", names, index=idx)
        idx = names.index(sel)
        st.session_state.current_idx = idx
        name, img = images[idx]

    col_img, col_result = st.columns([1, 1])

    with col_img:
        st.markdown(f"**{name}**")
        st.image(img, use_container_width=True)

    with col_result:
        CLASSIFY_SYSTEM = (
            "You are a document classification expert. "
            "Analyze the document image and classify it precisely. "
            "Always respond with valid JSON only — no prose, no markdown fences."
        )

        if st.button("🚀 Lancer la classification", type="primary", use_container_width=True):
            _prog = st.progress(0, text="Initialisation…")
            try:
                _prog.progress(15, text="Encodage de l'image…")
                _prog.progress(30, text="Envoi au modèle Qwen2.5-VL…")
                raw, latency, cls_tokens = _call_vlm(
                    img,
                    CLASSIFY_SYSTEM,
                    _build_classify_prompt(),
                    st.session_state.get("cfg_url", "http://localhost:8080/v1"),
                    st.session_state.get("cfg_api_key", "local-llama-key"),
                    st.session_state.get("cfg_model", "qwen2.5-vl-7b-q4_k_m"),
                    st.session_state.get("cfg_temperature", 0.1),
                    st.session_state.get("cfg_max_tokens", 2048),
                )
                _prog.progress(90, text="Parsing JSON…")
                data = _parse_json(raw)
                data["_latency_ms"] = latency
                st.session_state.classify_result = data
                st.session_state.classify_tokens = cls_tokens
                st.session_state.raw_response = raw
                _save_persistent_state()
                _prog.progress(97, text="Enregistrement MLflow…")
                # ── MLflow — Step 3 (classification) ──────────────────────────
                try:
                    import mlflow
                    _mlruns = ROOT / "logs" / "mlruns"
                    _mlruns.mkdir(parents=True, exist_ok=True)
                    mlflow.set_tracking_uri(_mlruns.as_uri())
                    mlflow.set_experiment("qwen_vl_pipeline")
                    _img_meta_3 = _img_info(img)
                    _run = mlflow.start_run(run_name=f"doc:{name}")
                    mlflow.log_params({
                        "source_file":      name,
                        "n_pages":          len(st.session_state.uploaded_images),
                        "model":            st.session_state.get("cfg_model", ""),
                        "temperature":      st.session_state.get("cfg_temperature", 0.1),
                        "max_tokens":       st.session_state.get("cfg_max_tokens", 2048),
                        "server_url":       st.session_state.get("cfg_url", ""),
                        "img_complexity":   _img_meta_3["complexity"],
                        "img_resolution":   f"{_img_meta_3['width']}x{_img_meta_3['height']}",
                    })
                    mlflow.log_metrics({
                        "classify_confidence":        float(data.get("confidence", 0)),
                        "classify_latency_ms":        float(latency),
                        "classify_prompt_tokens":     float(cls_tokens.get("prompt_tokens", 0)),
                        "classify_completion_tokens": float(cls_tokens.get("completion_tokens", 0)),
                        "img_size_kb":                float(_img_meta_3["size_kb"]),
                        "img_megapixels":             float(_img_meta_3["megapixels"]),
                    })
                    mlflow.set_tags({
                        "doc_type":   data.get("document_type", ""),
                        "language":   data.get("language", ""),
                        "category":   data.get("category", ""),
                        "step3_done": "true",
                    })
                    st.session_state["_mlflow_run_id"] = _run.info.run_id
                    mlflow.end_run()
                    st.toast("📈 Étape 3 trackée dans MLflow", icon="📈")
                except Exception as _me:
                    st.warning(f"MLflow (non bloquant) : {_me}")
                _prog.progress(100, text=f"Classification terminée ✓ ({latency} ms)")
            except Exception as e:
                _prog.empty()
                err = str(e)
                if "Connection error" in err or "ConnectionRefused" in err or "connect" in err.lower():
                    st.error(
                        "**Serveur llama.cpp inaccessible** — lancez-le dans un terminal :\n"
                        "```powershell\npwsh poc_qwen/scripts/start_server.ps1\n```"
                    )
                else:
                    st.error(f"Erreur: {e}")
                st.session_state.classify_result = None

        result = st.session_state.classify_result

        if result:
            st.markdown("### Résultat")

            doc_type = result.get("document_type", "—")
            category = result.get("category", "")
            conf     = float(result.get("confidence", 0))
            lang     = result.get("language", "")
            notes    = result.get("notes", "")
            latency  = result.get("_latency_ms", 0)

            st.markdown(
                f'<span class="chip">{doc_type}</span>'
                f'<span class="chip {category}">{category}</span>'
                f'<span style="font-size:12px;color:#6b7280;">🌐 {lang}</span>',
                unsafe_allow_html=True,
            )
            st.markdown("**Confiance**")
            st.markdown(_confidence_html(conf), unsafe_allow_html=True)

            if notes:
                st.caption(f"💬 {notes}")
            st.caption(f"⏱ {latency} ms")

            with st.expander("Réponse brute du modèle"):
                st.code(st.session_state.raw_response, language="json")

            # ── Correction manuelle ────────────────────────────────────────────
            st.divider()
            st.markdown("**✏️ Type incorrect ? Corrigez-le :**")
            _type_idx = _ALL_DOC_TYPES.index(doc_type) if doc_type in _ALL_DOC_TYPES else len(_ALL_DOC_TYPES) - 1
            _corrected = st.selectbox(
                "Type réel",
                _ALL_DOC_TYPES,
                index=_type_idx,
                key="correction_select",
                label_visibility="collapsed",
            )
            if _corrected != doc_type:
                if st.button("✓ Confirmer la correction", type="secondary", use_container_width=True):
                    _save_correction(name, doc_type, _corrected)
                    st.session_state.classify_result["document_type"] = _corrected
                    _save_persistent_state()
                    st.success(f"✓ **{doc_type}** → **{_corrected}** mémorisé — les prochaines classifications en tiendront compte.")
                    st.rerun()
            else:
                _n_corr = len(_load_corrections())
                st.caption(f"✓ Type confirmé · {_n_corr} correction(s) mémorisée(s)" if _n_corr else "✓ Type confirmé")

    st.divider()
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Upload"):
            st.session_state.step = 2
            st.rerun()
    with col_next:
        disabled = st.session_state.classify_result is None
        if st.button(
            "Suivant : Extraction →",
            type="primary",
            use_container_width=True,
            disabled=disabled,
        ):
            st.session_state.step = 4
            st.rerun()
        if disabled:
            st.caption("Lancez d'abord la classification")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Extraction
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 4:
    st.markdown("## 🔬 Étape 4 — Extraction des données")

    cls = st.session_state.classify_result or {}
    doc_type = cls.get("document_type", "default")

    st.markdown(
        f"Type détecté : **{doc_type}** — le prompt d'extraction est adapté automatiquement.  \n"
        "Vous pouvez modifier le prompt si nécessaire avant de lancer l'extraction."
    )

    images = st.session_state.uploaded_images
    if not images:
        st.warning("Aucun document chargé. Retournez à l'étape Upload.")
        if st.button("← Upload"):
            st.session_state.step = 2
            st.rerun()
        st.stop()
    idx = min(st.session_state.current_idx, len(images) - 1)
    st.session_state.current_idx = idx
    name, img = images[idx]

    # Prompt templates
    PROMPTS = {
        "invoice": """{
  "document_type": "invoice",
  "confidence": <float>,
  "fields": {
    "invoice_number": "",
    "date": "", "due_date": "",
    "vendor": "", "vendor_address": "",
    "client": "", "client_address": "",
    "total_amount": "", "vat_amount": "", "currency": ""
  },
  "line_items": [{"description":"","qty":"","unit_price":"","total":""}],
  "raw_text_snippet": "<first 200 chars>"
}""",
        "bank_statement": """{
  "document_type": "bank_statement",
  "confidence": <float>,
  "fields": {
    "account_holder": "", "iban": "", "bank_name": "",
    "period_start": "", "period_end": "",
    "opening_balance": "", "closing_balance": "", "currency": ""
  },
  "line_items": [{"description":"<label>","qty":"<date>","unit_price":"<debit>","total":"<credit>"}],
  "raw_text_snippet": "<first 200 chars>"
}""",
        "contract": """{
  "document_type": "contract",
  "confidence": <float>,
  "fields": {
    "contract_title": "", "contract_date": "",
    "effective_date": "", "expiry_date": "",
    "parties": [], "governing_law": "", "contract_value": ""
  },
  "line_items": [],
  "raw_text_snippet": "<first 200 chars>"
}""",
        "cv": """{
  "document_type": "cv",
  "confidence": <float>,
  "fields": {
    "full_name": "", "email": "", "phone": "",
    "location": "", "linkedin": "", "github": "",
    "current_title": "", "years_experience": "",
    "education": "", "languages": ""
  },
  "line_items": [{"description":"<job title>","qty":"<company>","unit_price":"<dates>","total":"<location>"}],
  "tables": [],
  "raw_text_snippet": "<first 200 chars>"
}""",
        "default": """{
  "document_type": "<type>",
  "confidence": <float>,
  "fields": {"<key>": "<value>"},
  "line_items": [{"description":"","qty":"","unit_price":"","total":""}],
  "tables": [],
  "raw_text_snippet": "<first 200 chars>"
}""",
    }

    base_schema = PROMPTS.get(doc_type, PROMPTS["default"])

    col_img, col_prompt = st.columns([1, 1])

    with col_img:
        st.markdown(f"**{name}**")
        st.image(img, use_container_width=True)

    with col_prompt:
        st.markdown("**Prompt d'extraction** *(éditable)*")
        system_prompt = st.text_area(
            "Système",
            value="You are an expert data extraction engine. Extract all structured information from the document. Respond only with valid JSON.",
            height=70,
            label_visibility="collapsed",
        )
        user_prompt = st.text_area(
            "Schéma JSON attendu",
            value=f"Extract all data from this {doc_type}. Return JSON:\n{base_schema}",
            height=280,
            key=f"extract_prompt_{doc_type}",
        )

        if st.button("🚀 Lancer l'extraction", type="primary", use_container_width=True):
            _prog = st.progress(0, text="Initialisation…")
            try:
                _prog.progress(10, text="Encodage de l'image…")
                _prog.progress(25, text="Envoi au modèle — extraction en cours (5-30s)…")
                raw, latency, ext_tokens = _call_vlm(
                    img, system_prompt, user_prompt,
                    st.session_state.get("cfg_url", "http://localhost:8080/v1"),
                    st.session_state.get("cfg_api_key", "local-llama-key"),
                    st.session_state.get("cfg_model", "qwen2.5-vl-7b-q4_k_m"),
                    st.session_state.get("cfg_temperature", 0.1),
                    st.session_state.get("cfg_max_tokens", 2048),
                )
                _prog.progress(85, text="Parsing JSON…")
                data = _parse_json(raw)
                data["_latency_ms"] = latency
                data["_source"]     = name
                data["_timestamp"]  = datetime.utcnow().isoformat()
                _img_meta_4 = _img_info(img)
                st.session_state.extract_result   = data
                st.session_state.extract_tokens   = ext_tokens
                st.session_state.current_img_meta = _img_meta_4
                st.session_state.raw_response     = raw
                # Save to history
                st.session_state.history.append({
                    "file": name,
                    "doc_type": doc_type,
                    "classify_confidence": float(cls.get("confidence", 0)),
                    "extract_confidence": float(data.get("confidence", 0)),
                    "n_fields": len(data.get("fields", {})),
                    "n_line_items": len(data.get("line_items", [])),
                    "classify_latency_ms": int(cls.get("_latency_ms", 0)),
                    "latency_ms": latency,
                    "timestamp": data["_timestamp"],
                    "classify_tokens": st.session_state.get("classify_tokens", {}),
                    "extract_tokens": ext_tokens,
                    "img_meta": _img_meta_4,
                    "result": data,
                })
                # Persist to file-based state
                _save_persistent_state()
                _prog.progress(92, text="Sauvegarde base vectorielle…")
                # Persist to ChromaDB vector store
                try:
                    from src.utils.vector_store import save_document as _vs_save
                    _doc_id = _vs_save(
                        filename=name,
                        doc_type=doc_type,
                        fields=data.get("fields", {}),
                        line_items=data.get("line_items", []),
                        raw_text=data.get("raw_text_snippet", ""),
                        confidence=float(data.get("confidence", 0)),
                    )
                    st.toast(f"Sauvegardé en base vectorielle (id: {_doc_id[:8]}…)", icon="🗄️")
                except Exception as _ve:
                    st.warning(f"Base vectorielle: {_ve}")
                # ── MLflow — Step 4 (extraction) : reprend le run de l'étape 3 ──
                try:
                    import mlflow, tempfile
                    _mlruns = ROOT / "logs" / "mlruns"
                    _mlruns.mkdir(parents=True, exist_ok=True)
                    mlflow.set_tracking_uri(_mlruns.as_uri())
                    mlflow.set_experiment("qwen_vl_pipeline")
                    _existing_run_id = st.session_state.get("_mlflow_run_id")
                    _run_kwargs = {"run_id": _existing_run_id} if _existing_run_id else {"run_name": f"doc:{name}"}
                    with mlflow.start_run(**_run_kwargs):
                        # Params de base — uniquement si pas déjà loggés à l'étape 3
                        if not _existing_run_id:
                            mlflow.log_params({
                                "source_file": name,
                                "n_pages":     len(st.session_state.uploaded_images),
                                "model":       st.session_state.get("cfg_model", ""),
                                "temperature": st.session_state.get("cfg_temperature", 0.1),
                                "max_tokens":  st.session_state.get("cfg_max_tokens", 2048),
                                "server_url":  st.session_state.get("cfg_url", ""),
                            })
                        # Params extraction (clés nouvelles — pas de conflit avec l'étape 3)
                        mlflow.log_params({
                            "doc_type":       doc_type,
                            "language":       cls.get("language", ""),
                            "category":       cls.get("category", ""),
                            "img_complexity": _img_meta_4["complexity"],
                            "img_resolution": f"{_img_meta_4['width']}x{_img_meta_4['height']}",
                        })
                        _cls_tok = st.session_state.get("classify_tokens", {})
                        # Metrics — classification + extraction + tokens + image
                        mlflow.log_metrics({
                            "classify_confidence":        float(cls.get("confidence", 0)),
                            "classify_latency_ms":        float(cls.get("_latency_ms", 0)),
                            "classify_prompt_tokens":     float(_cls_tok.get("prompt_tokens", 0)),
                            "classify_completion_tokens": float(_cls_tok.get("completion_tokens", 0)),
                            "extract_confidence":         float(data.get("confidence", 0)),
                            "extract_latency_ms":         float(latency),
                            "extract_prompt_tokens":      float(ext_tokens.get("prompt_tokens", 0)),
                            "extract_completion_tokens":  float(ext_tokens.get("completion_tokens", 0)),
                            "total_prompt_tokens":        float(_cls_tok.get("prompt_tokens", 0) + ext_tokens.get("prompt_tokens", 0)),
                            "total_completion_tokens":    float(_cls_tok.get("completion_tokens", 0) + ext_tokens.get("completion_tokens", 0)),
                            "n_fields":                   float(len(data.get("fields", {}))),
                            "n_line_items":               float(len(data.get("line_items", []))),
                            "n_tables":                   float(len(data.get("tables", []))),
                            "img_size_kb":                float(_img_meta_4["size_kb"]),
                            "img_megapixels":             float(_img_meta_4["megapixels"]),
                            "img_width_px":               float(_img_meta_4["width"]),
                            "img_height_px":              float(_img_meta_4["height"]),
                        })
                        # Tags
                        mlflow.set_tags({
                            "parse_success": "true",
                            "step4_done":    "true",
                        })
                        # Artifact — JSON résultat complet
                        _artifact = {
                            "classification": {k: v for k, v in cls.items() if not k.startswith("_")},
                            "extraction":     {k: v for k, v in data.items() if not k.startswith("_")},
                        }
                        with tempfile.NamedTemporaryFile(
                            mode="w", suffix=".json", delete=False, encoding="utf-8"
                        ) as _f:
                            json.dump(_artifact, _f, ensure_ascii=False, indent=2)
                            _tmp_path = _f.name
                        mlflow.log_artifact(_tmp_path, artifact_path="results")
                        Path(_tmp_path).unlink(missing_ok=True)
                    st.toast("📈 Étape 4 trackée — run MLflow complet", icon="📈")
                except Exception as _me:
                    st.warning(f"MLflow (non bloquant) : {_me}")
                _prog.progress(100, text=f"Extraction terminée ✓ ({latency} ms)")
            except Exception as e:
                _prog.empty()
                err = str(e)
                if "Connection error" in err or "ConnectionRefused" in err or "connect" in err.lower():
                    st.error(
                        "**Serveur llama.cpp inaccessible** — lancez-le dans un terminal :\n"
                        "```powershell\npwsh poc_qwen/scripts/start_server.ps1\n```"
                    )
                else:
                    st.error(f"Erreur extraction : {e}")

    st.divider()
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Classification"):
            st.session_state.step = 3
            st.rerun()
    with col_next:
        disabled = st.session_state.extract_result is None
        if st.button(
            "Voir les résultats →",
            type="primary",
            use_container_width=True,
            disabled=disabled,
        ):
            st.session_state.step = 5
            st.rerun()
        if disabled:
            st.caption("Lancez d'abord l'extraction")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Results
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 5:
    st.markdown("## 📊 Étape 5 — Résultats")

    data = st.session_state.extract_result or {}
    cls  = st.session_state.classify_result or {}

    tab_struct, tab_json, tab_analytics, tab_history, tab_vector = st.tabs(
        ["📋 Données structurées", "{ } JSON brut", "📊 Analytics", "🕘 Historique", "🗄️ Base Vectorielle"]
    )

    # ── Tab 1: Structured view ────────────────────────────────────────────────
    with tab_struct:
        col_meta, col_conf = st.columns([2, 1])
        with col_meta:
            doc_type = data.get("document_type", cls.get("document_type", "—"))
            category = cls.get("category", "")
            lang     = cls.get("language", "")
            source   = data.get("_source", "")
            ts       = data.get("_timestamp", "")[:19].replace("T", " ")
            latency  = data.get("_latency_ms", "—")

            st.markdown(
                f'<div class="kv-grid">'
                f'<span class="kv-key">Type</span>'
                f'<span class="kv-val"><span class="chip">{doc_type}</span>'
                f'<span class="chip {category}">{category}</span></span>'
                f'<span class="kv-key">Langue</span><span class="kv-val">🌐 {lang}</span>'
                f'<span class="kv-key">Source</span><span class="kv-val">{source}</span>'
                f'<span class="kv-key">Horodatage</span><span class="kv-val">{ts}</span>'
                f'<span class="kv-key">Latence</span><span class="kv-val">⏱ {latency} ms</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_conf:
            conf_ext = float(data.get("confidence", 0))
            conf_cls = float(cls.get("confidence", 0))
            st.markdown("**Confiance extraction**")
            st.markdown(_confidence_html(conf_ext), unsafe_allow_html=True)
            st.markdown("**Confiance classification**")
            st.markdown(_confidence_html(conf_cls), unsafe_allow_html=True)

        st.divider()

        # Fields
        fields = data.get("fields", {})
        if fields:
            st.markdown("### Champs extraits")
            rows = [{"Champ": k, "Valeur": str(v)} for k, v in fields.items() if v]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Line items
        line_items = data.get("line_items", [])
        if line_items:
            st.markdown("### Lignes")
            rows = [
                item if isinstance(item, dict) else {"description": str(item)}
                for item in line_items
            ]
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Tables
        tables = data.get("tables", [])
        if tables:
            st.markdown("### Tableaux détectés")
            for i, tbl in enumerate(tables):
                st.markdown(f"*Tableau {i+1}*")
                try:
                    st.dataframe(pd.DataFrame(tbl), use_container_width=True)
                except Exception:
                    st.json(tbl)

        # Raw text snippet
        snippet = data.get("raw_text_snippet", "")
        if snippet:
            with st.expander("Extrait texte brut"):
                st.text(snippet)

        # Exports
        st.divider()
        st.markdown("### Export")
        col_json_dl, col_csv_dl = st.columns(2)

        export_data = {
            "classification": cls,
            "extraction": {k: v for k, v in data.items() if not k.startswith("_")},
        }
        json_bytes = json.dumps(export_data, indent=2, ensure_ascii=False).encode()

        with col_json_dl:
            st.download_button(
                "⬇ Télécharger JSON",
                data=json_bytes,
                file_name=f"result_{doc_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )

        with col_csv_dl:
            if fields:
                csv_rows = [{"field": k, "value": str(v)} for k, v in fields.items()]
                csv_bytes = pd.DataFrame(csv_rows).to_csv(index=False).encode()
                st.download_button(
                    "⬇ Télécharger CSV (champs)",
                    data=csv_bytes,
                    file_name=f"fields_{doc_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    # ── Tab 2: Raw JSON ───────────────────────────────────────────────────────
    with tab_json:
        st.markdown("**Réponse complète du modèle**")
        clean = {k: v for k, v in data.items() if not k.startswith("_")}
        st.json(clean)
        with st.expander("Réponse brute (avant parsing)"):
            st.code(st.session_state.raw_response, language="json")

    # ── Tab 3: Analytics ─────────────────────────────────────────────────────
    with tab_analytics:
        _cls_tok  = st.session_state.get("classify_tokens", {})
        _ext_tok  = st.session_state.get("extract_tokens",  {})
        _img_meta = st.session_state.get("current_img_meta", {})

        st.markdown("### 📊 Tableau de bord d'analyse")

        # ── KPIs — 2 rows × 3 cols (avoid truncation with 6 cols) ────────────
        _total_prompt     = _cls_tok.get("prompt_tokens", 0)     + _ext_tok.get("prompt_tokens", 0)
        _total_completion = _cls_tok.get("completion_tokens", 0) + _ext_tok.get("completion_tokens", 0)
        _total_lat        = int(cls.get("_latency_ms", 0)) + int(data.get("_latency_ms", 0))
        _lat_str = f"{_total_lat/1000:.1f} s" if _total_lat >= 1000 else f"{_total_lat} ms"
        _size_str = f"{_img_meta.get('size_kb', '—')} KB"
        _res_str  = (f"{_img_meta['width']}×{_img_meta['height']}" if _img_meta.get('width') else "—")

        k1, k2, k3 = st.columns(3)
        k1.metric("Tokens prompt",   f"{_total_prompt:,}")
        k2.metric("Tokens réponse",  f"{_total_completion:,}")
        k3.metric("Total tokens",    f"{_total_prompt + _total_completion:,}")

        k4, k5, k6 = st.columns(3)
        k4.metric("Latence totale",  _lat_str)
        k5.metric("Taille (JPEG)",   _size_str)
        k6.metric("Résolution",      _res_str)

        st.divider()

        # ── Token breakdown table ─────────────────────────────────────────────
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("#### Tokens par étape")
            _tok_df = pd.DataFrame({
                "Étape":          ["Classification", "Extraction", "Total"],
                "Prompt":         [
                    _cls_tok.get("prompt_tokens", 0),
                    _ext_tok.get("prompt_tokens", 0),
                    _cls_tok.get("prompt_tokens", 0) + _ext_tok.get("prompt_tokens", 0),
                ],
                "Réponse":        [
                    _cls_tok.get("completion_tokens", 0),
                    _ext_tok.get("completion_tokens", 0),
                    _cls_tok.get("completion_tokens", 0) + _ext_tok.get("completion_tokens", 0),
                ],
                "Latence (ms)":   [
                    int(cls.get("_latency_ms", 0)),
                    int(data.get("_latency_ms", 0)),
                    _total_lat,
                ],
            })
            st.dataframe(_tok_df, use_container_width=True, hide_index=True)

        with c_right:
            st.markdown("#### Document analysé")
            _complexity_labels = {
                "very_high_res": "🔴 Très haute résolution",
                "high_res":      "🟠 Haute résolution",
                "standard":      "🟢 Standard",
                "low_res":       "🔵 Basse résolution",
            }
            _complexity_str = _complexity_labels.get(_img_meta.get("complexity", ""), "—")
            _doc_meta_rows = [
                {"Indicateur": "Type", "Valeur": data.get("document_type", cls.get("document_type", "—"))},
                {"Indicateur": "Langue", "Valeur": cls.get("language", "—")},
                {"Indicateur": "Complexité image", "Valeur": _complexity_str},
                {"Indicateur": "Mégapixels", "Valeur": str(_img_meta.get("megapixels", "—"))},
                {"Indicateur": "Résolution", "Valeur": f"{_img_meta.get('width','?')}×{_img_meta.get('height','?')} px"},
                {"Indicateur": "Taille (JPEG 92)", "Valeur": f"{_img_meta.get('size_kb', '—')} KB"},
                {"Indicateur": "Nb champs extraits", "Valeur": str(len(data.get("fields", {})))},
                {"Indicateur": "Nb lignes", "Valeur": str(len(data.get("line_items", [])))},
            ]
            st.dataframe(pd.DataFrame(_doc_meta_rows), use_container_width=True, hide_index=True)

        st.divider()

        # ── Token bar chart (session history) ────────────────────────────────
        _hist = st.session_state.history
        if len(_hist) > 1:
            st.markdown("#### Évolution de la session — tokens & latences")
            _hist_rows = []
            for _h in _hist:
                _ct = _h.get("classify_tokens", {})
                _et = _h.get("extract_tokens", {})
                _hist_rows.append({
                    "Document":      _h["file"][:20],
                    "Prompt total":  _ct.get("prompt_tokens", 0) + _et.get("prompt_tokens", 0),
                    "Réponse total": _ct.get("completion_tokens", 0) + _et.get("completion_tokens", 0),
                    "Latence cls.":  _h.get("classify_latency_ms", 0),
                    "Latence ext.":  _h.get("latency_ms", 0),
                })
            _hist_df = pd.DataFrame(_hist_rows).set_index("Document")
            st.bar_chart(_hist_df[["Prompt total", "Réponse total"]])
            st.caption("Latences (ms)")
            st.bar_chart(_hist_df[["Latence cls.", "Latence ext."]])

        # ── MLflow link ───────────────────────────────────────────────────────
        st.divider()
        _run_id = st.session_state.get("_mlflow_run_id", "")
        if _run_id:
            st.info(
                f"📈 **Run MLflow** : `{_run_id[:8]}…`  \n"
                "Visualisez tous les runs dans l'UI MLflow : `http://localhost:5000/#/experiments`"
            )

    # ── Tab 4: History ────────────────────────────────────────────────────────
    with tab_history:
        history = st.session_state.history
        if not history:
            st.info("Aucun document traité dans cette session.")
        else:
            st.markdown(f"**{len(history)} document(s) traité(s) cette session**")
            summary_rows = [
                {
                    "Fichier":    h["file"],
                    "Type":       h["doc_type"],
                    "Conf. class.": f"{h['classify_confidence']:.0%}",
                    "Conf. extrac.": f"{h['extract_confidence']:.0%}",
                    "Champs":     h["n_fields"],
                    "Lignes":     h["n_line_items"],
                    "Latence":    f"{h['latency_ms']} ms",
                    "Heure":      h["timestamp"][11:19],
                }
                for h in history
            ]
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

            # Export all history
            all_json = json.dumps(
                [{"file": h["file"], **h["result"]} for h in history],
                indent=2, ensure_ascii=False,
            ).encode()
            st.download_button(
                "⬇ Exporter toute la session (JSON)",
                data=all_json,
                file_name=f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

    # ── Tab 4: Vector Store ───────────────────────────────────────────────────
    with tab_vector:
        st.markdown("### 🗄️ Base Vectorielle Persistée (ChromaDB)")
        st.markdown(
            "Les documents extraits sont sauvegardés en local dans `data/vectorstore/`.  \n"
            "Cette base **persiste après fermeture de Streamlit** et permet la recherche sémantique."
        )
        try:
            from src.utils.vector_store import get_all_documents, search_similar, count as vs_count, clear_all

            _total = vs_count()
            col_cnt, col_clr = st.columns([2, 1])
            with col_cnt:
                st.metric("Documents persistés", _total)
            with col_clr:
                if st.button("🗑 Vider la base", use_container_width=True):
                    clear_all()
                    st.success("Base vectorielle vidée.")
                    st.rerun()

            if _total > 0:
                # Search
                st.markdown("#### Recherche sémantique")
                _query = st.text_input("Rechercher un document (ex: 'facture EDF 2024')", key="vs_query")
                if _query:
                    _results = search_similar(_query, n_results=5)
                    if _results:
                        _rows = [
                            {
                                "Fichier": r["filename"],
                                "Type": r["doc_type"],
                                "Confiance": f"{(r['confidence'] or 0):.0%}",
                                "Distance": f"{(r['distance'] or 0):.3f}",
                                "Champs": r["n_fields"],
                                "Date": (r["timestamp"] or "")[:19].replace("T", " "),
                            }
                            for r in _results
                        ]
                        st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)
                    else:
                        st.info("Aucun résultat.")

                st.markdown("#### Tous les documents")
                _all_docs = get_all_documents()
                _all_rows = [
                    {
                        "Fichier": d["filename"],
                        "Type": d["doc_type"],
                        "Champs": d["n_fields"],
                        "Date": (d["timestamp"] or "")[:19].replace("T", " "),
                    }
                    for d in _all_docs
                ]
                st.dataframe(pd.DataFrame(_all_rows), use_container_width=True, hide_index=True)

                with st.expander("Voir les données complètes (JSON)"):
                    st.json(_all_docs[:10])

        except Exception as _ve:
            st.warning(f"Base vectorielle indisponible: {_ve}")
            st.caption("Installez ChromaDB: `pip install chromadb`")

    st.divider()
    col_back, col_new = st.columns(2)
    with col_back:
        if st.button("← Modifier l'extraction"):
            st.session_state.step = 4
            st.rerun()
    with col_new:
        if st.button("🔄 Traiter un autre document", type="primary", use_container_width=True):
            st.session_state.step = 2
            st.session_state.classify_result  = None
            st.session_state.extract_result   = None
            st.session_state.raw_response     = ""
            st.session_state._mlflow_run_id   = None
            st.session_state.classify_tokens  = {}
            st.session_state.extract_tokens   = {}
            st.session_state.current_img_meta = {}
            st.rerun()

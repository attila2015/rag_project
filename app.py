"""
Page d'accueil — Document Intelligence POC
Streamlit multi-page entry point.
"""
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

SERVER_PORT = 8080
_PID_FILE   = ROOT / ".server.pid"


def _server_responding() -> bool:
    try:
        urllib.request.urlopen(f"http://localhost:{SERVER_PORT}/v1/models", timeout=2)
        return True
    except Exception:
        return False


def _port_open() -> bool:
    try:
        with socket.create_connection(("localhost", SERVER_PORT), timeout=1):
            return True
    except OSError:
        return False


def _server_process_alive() -> bool:
    if not _PID_FILE.exists():
        return False
    try:
        pid = int(_PID_FILE.read_text().strip())
        import psutil
        return psutil.pid_exists(pid)
    except Exception:
        return False


def _start_server():
    script = ROOT / "scripts" / "start_server.ps1"
    proc = subprocess.Popen(
        ["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", str(script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    _PID_FILE.write_text(str(proc.pid))
    return proc.pid


def ensure_server():
    """Start llama.cpp server if not already running. Returns (is_up, status_msg)."""
    if _server_responding():
        return True, "en ligne"

    if _port_open():
        # Port open but not llama.cpp — skip (Docker etc.)
        return False, "port occupé par un autre processus"

    if _server_process_alive():
        return False, "démarrage en cours…"

    # Launch
    _start_server()
    return False, "démarrage en cours…"


# ── Auto-start (runs once per Streamlit worker process) ───────────────────────
if "server_boot_attempted" not in st.session_state:
    st.session_state.server_boot_attempted = True
    if not _server_responding() and not _port_open():
        _start_server()

# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Document Intelligence — Qwen VL",
    page_icon="🔍",
    layout="centered",
)


def home():
    server_up, srv_msg = ensure_server()

    model_files = [f for f in (ROOT / "models").glob("*.gguf") if "mmproj" not in f.name.lower()]

    st.markdown("""
<div style="text-align:center; padding: 40px 0 20px 0;">
  <div style="font-size:64px;">🔍</div>
  <h1 style="margin:0; font-size:32px;">Document Intelligence</h1>
  <p style="color:#6b7280; font-size:16px; margin-top:8px;">
    Pipeline local · Qwen2.5-VL · llama.cpp
  </p>
</div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        if model_files:
            st.success(f"✓ Modèle : {model_files[0].name[:30]}")
        else:
            st.error("✗ Aucun modèle GGUF — lancez le setup")
    with c2:
        if server_up:
            st.success(f"✓ Serveur : {srv_msg} :{SERVER_PORT}")
        else:
            st.warning(f"⏳ Serveur : {srv_msg}")
            st.caption("Rafraîchissez dans quelques secondes (chargement du modèle ~30 s)")
    with c3:
        st.info("🔁 Prêt à traiter des documents")

    st.divider()

    col_guide, col_pipeline = st.columns(2)
    with col_guide:
        st.markdown("""
### 📖 Guide technique
Tout ce qu'un data scientist doit savoir :
- Architecture et choix techniques
- Setup pas à pas
- Prompt engineering
- Évaluation et métriques
- Fine-tuning Qwen2.5-VL
- Troubleshooting
""")
        st.page_link("pages/01_Guide.py", label="Ouvrir le guide →", icon="📖")

    with col_pipeline:
        st.markdown("""
### 🔬 Pipeline interactif
5 étapes guidées :
1. Configuration serveur
2. Upload document / PDF
3. Classification automatique
4. Extraction structurée
5. Export JSON / CSV
""")
        st.page_link("pages/02_Pipeline.py", label="Lancer le pipeline →", icon="🚀")

    st.caption("POC — local, aucune donnée envoyée vers le cloud")


pg = st.navigation([
    st.Page(home,                       title="Accueil",         icon="🏠"),
    st.Page("pages/01_Guide.py",        title="Guide technique", icon="📖"),
    st.Page("pages/02_Pipeline.py",     title="Pipeline",        icon="🚀"),
    st.Page("pages/03_FineTuning.py",   title="Fine-Tuning",     icon="🧪"),
])
pg.run()

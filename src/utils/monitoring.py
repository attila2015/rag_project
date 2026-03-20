"""
Monitoring système en temps réel : CPU, RAM, Intel Arc GPU, Intel NPU.
Fonctionne sans dépendances propriétaires — psutil + WMI subprocess.
"""
from __future__ import annotations
import subprocess
import time
from dataclasses import dataclass, field


@dataclass
class SystemMetrics:
    cpu_percent: float = 0.0
    cpu_freq_mhz: float = 0.0
    ram_percent: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    gpu_name: str = ""
    gpu_mem_used_mb: float = 0.0
    gpu_mem_total_mb: float = 0.0
    gpu_util_percent: float = 0.0
    npu_present: bool = False
    npu_name: str = ""
    npu_tops: float = 0.0
    inference_active: bool = False


def get_metrics() -> SystemMetrics:
    import psutil
    m = SystemMetrics()

    # ── CPU ─────────────────────────────────────────────────────────────────
    m.cpu_percent = psutil.cpu_percent(interval=0.2)
    freq = psutil.cpu_freq()
    if freq:
        m.cpu_freq_mhz = freq.current

    # ── RAM ─────────────────────────────────────────────────────────────────
    ram = psutil.virtual_memory()
    m.ram_percent = ram.percent
    m.ram_used_gb = ram.used / 1e9
    m.ram_total_gb = ram.total / 1e9

    # ── Intel Arc GPU via WMI ────────────────────────────────────────────────
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "Get-CimInstance Win32_VideoController | "
             "Select-Object Name, AdapterRAM | ConvertTo-Json"],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            if isinstance(data, dict):
                data = [data]
            for gpu in data:
                name = gpu.get("Name", "")
                if "Intel" in name or "Arc" in name:
                    m.gpu_name = name
                    vram = gpu.get("AdapterRAM", 0) or 0
                    m.gpu_mem_total_mb = vram / 1e6
                    break
    except Exception:
        pass

    # ── Intel Arc GPU utilisation via PDH (Windows Performance Counter) ──────
    try:
        ps_cmd = (
            "(Get-Counter '\\GPU Engine(*engtype_3D)\\Utilization Percentage' "
            "-ErrorAction SilentlyContinue).CounterSamples | "
            "Where-Object {$_.InstanceName -match 'Intel'} | "
            "Measure-Object CookedValue -Average | "
            "Select-Object -ExpandProperty Average"
        )
        r = subprocess.run(
            ["powershell", "-Command", ps_cmd],
            capture_output=True, text=True, timeout=3
        )
        val = r.stdout.strip()
        if val and val.replace(".", "").replace(",", "").isdigit():
            m.gpu_util_percent = float(val.replace(",", "."))
    except Exception:
        pass

    # ── Intel Arc GPU mémoire via DXGI/PDH ──────────────────────────────────
    try:
        ps_mem = (
            "(Get-Counter '\\GPU Local Adapter Memory(*)\\Local Usage' "
            "-ErrorAction SilentlyContinue).CounterSamples | "
            "Measure-Object CookedValue -Sum | "
            "Select-Object -ExpandProperty Sum"
        )
        r = subprocess.run(
            ["powershell", "-Command", ps_mem],
            capture_output=True, text=True, timeout=3
        )
        val = r.stdout.strip()
        if val and val.replace(".", "").replace(",", "").isdigit():
            m.gpu_mem_used_mb = float(val.replace(",", ".")) / 1e6
    except Exception:
        pass

    # ── Intel NPU (Core Ultra — Meteor Lake / Arrow Lake) ────────────────────
    try:
        r = subprocess.run(
            ["powershell", "-Command",
             "Get-PnpDevice | Where-Object {$_.FriendlyName -match 'NPU|Neural' "
             "-and $_.Status -eq 'OK'} | Select-Object FriendlyName | ConvertTo-Json"],
            capture_output=True, text=True, timeout=3
        )
        if r.returncode == 0 and r.stdout.strip() and r.stdout.strip() != "null":
            import json
            npu_data = json.loads(r.stdout)
            if isinstance(npu_data, dict):
                npu_data = [npu_data]
            if npu_data:
                m.npu_present = True
                m.npu_name = npu_data[0].get("FriendlyName", "Intel NPU")
                # Core Ultra 5 125H = 11.5 TOPS, Core Ultra 7 = 13 TOPS
                m.npu_tops = 11.5

    except Exception:
        pass

    return m


def color_for_pct(pct: float) -> str:
    if pct < 60:
        return "#22c55e"   # green
    if pct < 85:
        return "#f59e0b"   # amber
    return "#ef4444"       # red

#!/usr/bin/env python3
"""
capture_chart_dataset.py
Génère 40 screenshots TradingView CAC40 (1Y / 1D) + labels.json

Prérequis :
    pip install playwright yfinance pillow
    playwright install chromium

Sortie :
    finetuning/dataset/images/chart_cac40_YYYY-MM-DD.jpg  (×40)
    finetuning/dataset/labels.json
"""

import asyncio, json, time, io, sys
from datetime import datetime, timedelta
from pathlib import Path

# ── Auto-install des dépendances ───────────────────────────────────────────────
import subprocess as _sp

def _ensure(pkg: str, import_name: str | None = None):
    mod = import_name or pkg
    try:
        __import__(mod)
    except ImportError:
        print(f"[setup] Installation de {pkg}…")
        _sp.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

_ensure("yfinance")
_ensure("playwright", "playwright.async_api")
_ensure("Pillow", "PIL")

# Vérifier que chromium est installé pour Playwright
try:
    from playwright.sync_api import sync_playwright as _spw
    with _spw() as _p:
        _p.chromium.executable_path  # lève une exception si absent
except Exception:
    print("[setup] Installation chromium pour Playwright…")
    _sp.check_call([sys.executable, "-m", "playwright", "install", "chromium"])

# ── Imports définitifs ─────────────────────────────────────────────────────────
import yfinance as yf
from playwright.async_api import async_playwright
from PIL import Image

# ── Constantes ─────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
IMAGES_DIR  = ROOT / "finetuning" / "dataset" / "images"
LABELS_FILE = ROOT / "finetuning" / "dataset" / "labels.json"

N_SAMPLES   = int(sys.argv[1]) if len(sys.argv) > 1 else 40
YF_SYMBOL   = "^FCHI"            # Yahoo Finance : indice CAC40
TV_SYMBOL   = "TVC:CAC40"        # label interne TradingView
TV_URL      = "https://fr.tradingview.com/chart/?symbol=TVC%3ACAC40"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# ── Étape 1 : récupérer 40 points de données via yfinance ─────────────────────

def get_sample_data(n: int = 40) -> list[dict]:
    """Retourne n points hebdomadaires sur 1 an de données CAC40."""
    print("[yfinance] Téléchargement données CAC40 1 an…")
    ticker = yf.Ticker(YF_SYMBOL)
    hist = ticker.history(period="1y", interval="1d")
    hist = hist[hist["Volume"] > 0].dropna()

    # ~252 jours de bourse / 40 ≈ tous les 6-7 jours
    step = max(1, len(hist) // n)
    rows = hist.iloc[::step][:n]

    samples = []
    for dt, row in rows.iterrows():
        samples.append({
            "date":   dt.strftime("%Y-%m-%d"),
            "open":   round(float(row["Open"]),   2),
            "high":   round(float(row["High"]),   2),
            "low":    round(float(row["Low"]),    2),
            "close":  round(float(row["Close"]),  2),
            "volume": int(row["Volume"]),
        })

    print(f"  → {len(samples)} dates sélectionnées ({samples[0]['date']} → {samples[-1]['date']})")
    return samples


# ── Étape 2 : Playwright — screenshots TradingView ────────────────────────────

async def setup_chart(page):
    """Ouvre TradingView, passe en plein écran, configure 1D / 1Y."""

    print(f"[Browser] Ouverture {TV_URL}")
    await page.goto(TV_URL, wait_until="networkidle", timeout=90_000)
    await page.wait_for_timeout(4_000)

    # Fermer toute popup visible (cookies, login, promo…)
    popup_selectors = [
        '[aria-label="Fermer"]', '[aria-label="Close"]',
        'button.toast-closeButton', '.js-dialog__close',
        '[data-name="close"]', '.tv-dialog__close',
        'button:has-text("Accepter")', 'button:has-text("Tout accepter")',
        'button:has-text("Continuer")',
    ]
    for sel in popup_selectors:
        try:
            el = await page.query_selector(sel)
            if el and await el.is_visible():
                await el.click()
                await page.wait_for_timeout(600)
        except Exception:
            pass

    # Plein écran : touche 'F' (raccourci TradingView)
    await page.keyboard.press("f")
    await page.wait_for_timeout(1_000)

    # Sélectionner timeframe 1D
    tf_selectors = [
        '[data-value="1D"]',
        'button[class*="button"]:has-text("1J")',
        'button[class*="button"]:has-text("1D")',
    ]
    for sel in tf_selectors:
        try:
            btn = await page.query_selector(sel)
            if btn:
                await btn.click()
                await page.wait_for_timeout(1_000)
                break
        except Exception:
            pass

    # Sélectionner range 1 an via le sélecteur de plage en bas du chart
    range_selectors = ['button:has-text("1A")', 'button:has-text("1Y")', '[data-value="12M"]']
    for sel in range_selectors:
        try:
            btn = await page.query_selector(sel)
            if btn:
                await btn.click()
                await page.wait_for_timeout(1_500)
                break
        except Exception:
            pass

    await page.wait_for_timeout(2_000)
    print("  [chart] Configuré : 1D / 1Y / plein écran")


def _draw_annotations(img: Image.Image, x: int, close: float,
                       price_min: float, price_max: float) -> Image.Image:
    """
    Dessine sur l'image :
      • une ligne verticale rouge à x (date sélectionnée)
      • une ligne horizontale rouge au prix de clôture
    Zone chart supposée : y ∈ [CHART_TOP, CHART_BOT] en pixels.
    """
    from PIL import ImageDraw
    # Marges estimées de TradingView en plein écran (px sur 1080p)
    CHART_TOP = 60    # barre d'outils haute
    CHART_BOT = 980   # axe des dates en bas

    w, h = img.size
    # Recalibrer si la hauteur diffère de 1080
    scale = h / 1080
    top = int(CHART_TOP * scale)
    bot = int(CHART_BOT * scale)

    # Position y du prix de clôture (axe inversé : haut = prix max)
    price_range = price_max - price_min or 1.0
    price_frac  = (price_max - close) / price_range          # 0 = haut, 1 = bas
    y_price     = int(top + price_frac * (bot - top))

    draw = ImageDraw.Draw(img)
    RED  = (220, 30, 30)

    # Ligne verticale (date)
    draw.line([(x, top), (x, bot)], fill=RED, width=2)

    # Ligne horizontale (close)
    draw.line([(0, y_price), (w, y_price)], fill=RED, width=2)

    # Petit carré d'intersection pour souligner le point exact
    r = 6
    draw.ellipse([(x - r, y_price - r), (x + r, y_price + r)],
                 outline=RED, width=2)

    return img


async def hover_date_and_screenshot(page, sample: dict, img_path: Path,
                                    price_min: float, price_max: float) -> bool:
    """
    Positionne le curseur sur la bougie correspondant à `sample['date']`,
    prend un screenshot plein écran puis dessine les annotations prix/date.
    """
    date_str = sample["date"]
    target   = datetime.strptime(date_str, "%Y-%m-%d")
    today    = datetime.today()

    # ── Position x (date) ─────────────────────────────────────────────────────
    vp = page.viewport_size or {"width": 1920, "height": 1080}
    w, h = vp["width"], vp["height"]

    days_ago = (today - target).days
    fraction = 1.0 - (days_ago / 365.0)
    fraction = max(0.03, min(0.97, fraction))

    x = int(w * fraction)
    y = int(h * 0.50)

    # ── Déplacer la souris + attendre le tooltip ──────────────────────────────
    await page.mouse.move(x, y)
    await page.wait_for_timeout(900)

    # ── Screenshot ────────────────────────────────────────────────────────────
    png_bytes = await page.screenshot(type="png", full_page=False)

    img = Image.open(io.BytesIO(png_bytes))

    # ── Annoter : ligne verticale (date) + ligne horizontale (close) ──────────
    img = _draw_annotations(img, x, float(sample["close"]), price_min, price_max)

    img.save(str(img_path), "JPEG", quality=92)
    return True


async def _get_chart_price_range(page, samples: list[dict]) -> tuple[float, float]:
    """
    Tente de lire la plage de prix visible depuis le widget TradingView via JS.
    Fallback sur min(low)/max(high) du dataset si l'API interne n'est pas accessible.
    """
    try:
        result = await page.evaluate("""
            () => {
                // Chercher l'instance du widget TradingView dans le scope global
                const keys = Object.keys(window);
                for (const k of keys) {
                    const obj = window[k];
                    if (obj && typeof obj.activeChart === 'function') {
                        try {
                            const chart = obj.activeChart();
                            const pane  = chart.getPanes()[0];
                            const scale = pane.getRightPriceScale();
                            const range = scale.getVisiblePriceRange();
                            return { min: range.minValue, max: range.maxValue, source: 'tvWidget' };
                        } catch (_) {}
                    }
                }
                // Deuxième tentative : lire les labels de l'axe des prix (texte DOM)
                const labels = Array.from(
                    document.querySelectorAll('[class*="priceAxisLabel"], [class*="price-axis"] span')
                ).map(el => parseFloat(el.textContent.replace(/[^0-9.]/g, '')))
                 .filter(v => !isNaN(v) && v > 100);
                if (labels.length >= 2) {
                    return { min: Math.min(...labels), max: Math.max(...labels), source: 'dom' };
                }
                return null;
            }
        """)
        if result and result.get("min") and result.get("max"):
            src = result["source"]
            lo, hi = float(result["min"]), float(result["max"])
            print(f"  [range] Prix visible depuis TradingView ({src}) : {lo:.0f} – {hi:.0f}")
            return lo, hi
    except Exception as e:
        print(f"  [range] JS eval échoué ({e}), fallback dataset")

    # Fallback : min/max du dataset
    lo = min(s["low"]  for s in samples)
    hi = max(s["high"] for s in samples)
    print(f"  [range] Prix depuis dataset : {lo:.0f} – {hi:.0f}")
    return lo, hi


async def capture_all(samples: list[dict]) -> list[dict]:
    labels = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,          # headed = moins de détection bot
            args=[
                "--start-maximized",
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
            ],
        )
        ctx = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            locale="fr-FR",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/132.0.0.0 Safari/537.36"
            ),
        )
        page = await ctx.new_page()
        await setup_chart(page)

        # Plage de prix : tenter de lire depuis le chart TradingView,
        # sinon fallback sur min/max du dataset.
        price_min, price_max = await _get_chart_price_range(page, samples)

        n = len(samples)
        for i, sample in enumerate(samples):
            date_str = sample["date"]
            img_name = f"chart_cac40_{date_str}.jpg"
            img_path = IMAGES_DIR / img_name

            print(f"  [{i+1:02d}/{n}] {date_str}  O={sample['open']}  H={sample['high']}  "
                  f"L={sample['low']}  C={sample['close']}")

            ok = await hover_date_and_screenshot(page, sample, img_path, price_min, price_max)
            if not ok:
                print(f"  ⚠  screenshot échoué pour {date_str}")
                continue

            labels.append({
                "image":    f"images/{img_name}",
                "doc_type": "chart",
                "expected": {
                    "symbol":     TV_SYMBOL,
                    "chart_date": date_str,
                    "timeframe":  "1D",
                    "open":       str(sample["open"]),
                    "high":       str(sample["high"]),
                    "low":        str(sample["low"]),
                    "close":      str(sample["close"]),
                    "volume":     str(sample["volume"]),
                    "chart_type": "candlestick",
                },
            })

            # Petite pause entre captures pour ne pas saturer
            await page.wait_for_timeout(300)

        await browser.close()

    return labels


# ── Main ───────────────────────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("  CAC40 Chart Dataset — 40 screenshots + labels.json")
    print("=" * 60)

    samples = get_sample_data(N_SAMPLES)

    print("\n[Playwright] Capture des screenshots TradingView…")
    labels = await capture_all(samples)

    # Sauvegarder (fusion avec labels existants si présent)
    existing = []
    if LABELS_FILE.exists():
        try:
            existing = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    # Dédupliquer sur image
    seen = {e["image"] for e in existing}
    merged = existing + [l for l in labels if l["image"] not in seen]

    LABELS_FILE.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n✅  {len(labels)} screenshots capturés")
    print(f"    Images  : {IMAGES_DIR}")
    print(f"    Labels  : {LABELS_FILE}  ({len(merged)} exemples total)")


if __name__ == "__main__":
    asyncio.run(main())

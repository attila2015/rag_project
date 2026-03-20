"""
Image utilities: loading, resizing, base64 encoding for VLM input.
"""
import base64
import io
from pathlib import Path

from PIL import Image, ImageOps

MAX_DIMENSION = 1024   # Qwen2.5-VL works best ≤ 1280px; 1024 is safe
JPEG_QUALITY  = 92


def load_image(path: str | Path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)          # fix phone rotation
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img


def resize_for_vlm(img: Image.Image, max_dim: int = MAX_DIMENSION) -> Image.Image:
    """Resize so neither width nor height exceeds max_dim, preserving aspect ratio."""
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    scale = max_dim / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)


def image_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    """Encode PIL image to base64 data URI."""
    buf = io.BytesIO()
    if fmt == "JPEG" and img.mode == "RGBA":
        img = img.convert("RGB")
    img.save(buf, format=fmt, quality=JPEG_QUALITY)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


def load_and_encode(path: str | Path, max_dim: int = MAX_DIMENSION) -> str:
    """Load, resize, and base64-encode an image file in one call."""
    img = load_image(path)
    img = resize_for_vlm(img, max_dim)
    return image_to_base64(img)


def _find_poppler_path() -> str | None:
    """Auto-detect poppler bin directory (bundled or system PATH)."""
    import shutil
    # 1. Bundled in project bin/poppler/
    project_root = Path(__file__).parent.parent.parent
    for candidate in project_root.rglob("pdftoppm.exe"):
        return str(candidate.parent)
    for candidate in project_root.rglob("pdftoppm"):
        return str(candidate.parent)
    # 2. System PATH
    if shutil.which("pdftoppm"):
        return None  # pdf2image finds it automatically
    return None


def pdf_to_images(pdf_path: str | Path, dpi: int = 200) -> list[Image.Image]:
    """Convert all pages of a PDF to PIL images."""
    try:
        from pdf2image import convert_from_path
        poppler_path = _find_poppler_path()
        kwargs = {"dpi": dpi}
        if poppler_path:
            kwargs["poppler_path"] = poppler_path
        return convert_from_path(str(pdf_path), **kwargs)
    except ImportError:
        raise SystemExit("pdf2image not installed. Run: pip install pdf2image")
    except Exception as e:
        raise RuntimeError(f"PDF conversion failed for {pdf_path}: {e}") from e


def pdf_bytes_to_images(pdf_bytes: bytes, dpi: int = 200) -> list[Image.Image]:
    """Convert PDF bytes (from st.file_uploader) to PIL images."""
    try:
        from pdf2image import convert_from_bytes
        poppler_path = _find_poppler_path()
        kwargs = {"dpi": dpi}
        if poppler_path:
            kwargs["poppler_path"] = poppler_path
        return convert_from_bytes(pdf_bytes, **kwargs)
    except ImportError:
        raise SystemExit("pdf2image not installed. Run: pip install pdf2image")
    except Exception as e:
        raise RuntimeError(f"PDF bytes conversion failed: {e}") from e

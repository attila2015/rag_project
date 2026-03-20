"""
Convert PDF files to images ready for Qwen VL inference.

Usage:
    python scripts/convert_pdf.py --input data/raw/doc.pdf
    python scripts/convert_pdf.py --input data/raw/  # batch
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.image_utils import pdf_to_images, resize_for_vlm

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).parent.parent / "data" / "processed"


def convert(src: Path) -> list[Path]:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    pages = pdf_to_images(src)
    results = []
    for i, img in enumerate(pages):
        img = resize_for_vlm(img)
        dest = PROC_DIR / f"{src.stem}_page{i+1:03d}.jpg"
        img.save(dest, "JPEG", quality=92)
        print(f"  [{i+1}/{len(pages)}] {dest.name}")
        results.append(dest)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="PDF file or directory")
    args = parser.parse_args()

    src = Path(args.input)
    if src.is_dir():
        pdfs = list(src.glob("**/*.pdf"))
        print(f"Found {len(pdfs)} PDFs in {src}")
        for pdf in pdfs:
            print(f"\n→ {pdf.name}")
            convert(pdf)
    elif src.suffix.lower() == ".pdf":
        convert(src)
    else:
        print(f"[skip] Not a PDF: {src}")

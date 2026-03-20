"""
Minimal example: load a document image, send to Qwen2.5-VL, get structured JSON.

Usage:
    python example_inference.py
    python example_inference.py --image path/to/your/document.jpg
"""
import argparse
import json
import sys
from pathlib import Path

# ── Bootstrap ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv()

from src.utils.image_utils import load_and_encode
from src.pipeline.infer import call_vlm, parse_json_from_response

# ── 1. Load and encode the document image ────────────────────────────────────
def load_document(image_path: str) -> str:
    """Resize to ≤ 1024px and return a base64 data URI."""
    print(f"[1/3] Loading image: {image_path}")
    return load_and_encode(image_path)


# ── 2. Send to Qwen2.5-VL ────────────────────────────────────────────────────
SYSTEM = "You are an expert document analysis engine. Respond only with valid JSON."

PROMPT = """Analyze this document image and extract all available information.

Return a single JSON object with:
{
  "document_type": "<type>",
  "language": "<lang>",
  "confidence": <0.0-1.0>,
  "fields": { "<key>": "<value>" },
  "line_items": [{ "description": "", "amount": "" }],
  "summary": "<one sentence describing the document>"
}"""


def query_vlm(image_b64: str) -> tuple[str, int]:
    print("[2/3] Sending to Qwen2.5-VL …")
    return call_vlm(image_b64, SYSTEM, PROMPT)


# ── 3. Parse and return structured JSON ──────────────────────────────────────
def parse_result(raw_text: str) -> dict:
    print("[3/3] Parsing structured output …")
    return parse_json_from_response(raw_text)


# ── Main ─────────────────────────────────────────────────────────────────────
def run(image_path: str) -> dict:
    image_b64 = load_document(image_path)
    raw, latency_ms = query_vlm(image_b64)
    result = parse_result(raw)
    result["_latency_ms"] = latency_ms
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="data/raw/sample_invoice.jpg",
                        help="Path to document image (jpg/png/pdf page)")
    args = parser.parse_args()

    image = Path(args.image)
    if not image.exists():
        print(f"\n[NOTE] File not found: {image}")
        print("       Place a document image in data/raw/ and re-run,")
        print("       or pass --image path/to/your/document.jpg\n")
        sys.exit(0)

    output = run(str(image))

    print("\n── Structured JSON output ──────────────────────────────────")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nLatency: {output.get('_latency_ms', '?')} ms")

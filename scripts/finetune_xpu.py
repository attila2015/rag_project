"""
Fine-tuning Qwen2.5-VL-3B avec LoRA r=4 sur Intel Arc (IPEX XPU)
Lancé par 03_FineTuning.py via subprocess.

┌──────────────────────────────────────────────────────────────────────┐
│  Architecture fonctionnelle                                          │
│                                                                      │
│  prompts/registry.json                                               │
│        │                                                             │
│        ▼                                                             │
│  PromptRegistry ──── get()         → Prompt   (system, user, hash)  │
│                 └─── get_schema()  → Schema   (eval_fields, ...)    │
│                                                                      │
│  finetune_xpu.py                                                     │
│    ├── DocumentDataset  → même prompts que classify.py/extract.py   │
│    │       ×2 items/image : tâche classify + tâche extract          │
│    ├── MetricsCallback  → train_loss, val_loss → MLflow + .jsonl    │
│    ├── evaluate_model() → F1 sur eval_fields du registry par type   │
│    └── main()           → MLflow run complet (params + métriques    │
│                            + artifacts) dans "qwen_vl_finetuning"   │
│                                                                      │
│  Sorties :                                                           │
│    finetuning/<run>/adapter/    → LoRA weights (PEFT)               │
│    finetuning/<run>/merged/     → HF FP16 standalone                │
│    logs/ft_<run>.jsonl          → métriques pas-à-pas               │
│    logs/ft_<run>_eval.json      → F1 + exemples de validation       │
└──────────────────────────────────────────────────────────────────────┘

Usage :
    python scripts/finetune_xpu.py --config finetuning/<run>_config.json
"""
import argparse, json, os, sys, time, math, warnings
from pathlib import Path
from datetime import datetime, timezone

# Supprimer les warnings non-bloquants connus
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")
warnings.filterwarnings("ignore", category=DeprecationWarning)

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# ─── Prompts depuis le registry (source unique de vérité) ────────────────────
try:
    from src.pipeline.prompt_registry import (
        registry as _prompt_registry,
        get_classify_prompts,
        get_extract_prompts,
        CLASSIFY_SYS, CLASSIFY_USER,
        EXTRACT_SYSTEM as EXTRACT_SYS,
        EXTRACT_TEMPLATES,
        get_eval_fields as _get_eval_fields,
    )
    EXTRACT_PROMPTS = {k: v["user"] for k, v in EXTRACT_TEMPLATES.items()}
    _PROMPT_VERSIONS = _prompt_registry.get_active_versions()
    print(f"[INFO] Prompt registry chargé — versions actives : {_PROMPT_VERSIONS}")
except ImportError as e:
    print(f"[WARN] Prompt registry indisponible ({e}) — prompts génériques")
    CLASSIFY_SYS  = "You are a document classification expert. Return only valid JSON."
    CLASSIFY_USER = 'Classify this document. Return JSON: {"document_type":"","confidence":0.0}'
    EXTRACT_SYS   = "You are an expert data extraction engine. Return only valid JSON."
    EXTRACT_PROMPTS = {"default": 'Extract fields. Return JSON: {"fields":{}}'}
    _PROMPT_VERSIONS = {}
    _get_eval_fields = lambda doc_type: []


# ─── Helpers logging ──────────────────────────────────────────────────────────
def _log(log_path: Path, record: dict):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─── Parsing args ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Chemin vers le JSON de config")
parser.add_argument("--one-phase", action="store_true",
                    help="Exécuter une seule phase puis quitter (contrôle manuel depuis l'UI)")
args = parser.parse_args()

cfg      = json.loads(Path(args.config).read_text(encoding="utf-8"))
run_id   = cfg["run_name"]
log_path = LOGS_DIR / f"ft_{run_id}.jsonl"
eval_path = LOGS_DIR / f"ft_{run_id}_eval.json"
out_dir  = Path(cfg["output_dir"])
out_dir.mkdir(parents=True, exist_ok=True)


# ─── Imports lourds ───────────────────────────────────────────────────────────
import torch

# IPEX XPU si disponible
device = cfg.get("device", "cpu")
if device == "xpu":
    try:
        import intel_extension_for_pytorch as ipex  # noqa
        if not torch.xpu.is_available():
            print("[WARN] XPU non disponible — fallback CPU")
            device = "cpu"
    except ImportError:
        print("[WARN] IPEX non installé — fallback CPU")
        device = "cpu"

print(f"[INFO] Device : {device}")

from transformers import (
    Qwen2_5_VLProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


# ─── Dataset ──────────────────────────────────────────────────────────────────
def _load_image(img_path: Path, max_px: int = 1024) -> Image.Image:
    """Charge et redimensionne une image en préservant le ratio."""
    if img_path.exists():
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        scale = min(max_px / max(w, h), 1.0)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return img
    return Image.new("RGB", (224, 224), color=(200, 200, 200))


def _build_messages(image: Image.Image, sys_prompt: str, user_prompt: str, expected_json: str) -> list:
    """Construit les messages ChatML au format Qwen2.5-VL."""
    return [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": user_prompt},
            ],
        },
        {"role": "assistant", "content": expected_json},
    ]


class DocumentDataset(Dataset):
    """
    Deux tâches par image (pipeline production) :
      1. Classification  → prompts de classify.py + réponse classification JSON
      2. Extraction      → prompts de extract.py  + réponse extraction JSON
    Cela double la taille effective du dataset et aligne train/inférence.
    """

    def __init__(self, examples: list, processor, max_seq_len: int = 1024):
        self.processor   = processor
        self.max_seq_len = max_seq_len
        # Construire la liste plate : 2 items par exemple
        self.items = []
        for ex in examples:
            doc_type = ex.get("doc_type", "default")
            expected = ex.get("expected", {})
            img_rel  = ex.get("image", "")

            # ── Tâche 1 : Classification ─────────────────────────────────────
            classify_answer = json.dumps({
                "document_type": doc_type,
                "category":      ex.get("category", "financial"),
                "confidence":    ex.get("confidence", 0.9),
                "language":      ex.get("language", "fr"),
                "notes":         ex.get("notes", ""),
            }, ensure_ascii=False)
            self.items.append({
                "image":    img_rel,
                "sys":      CLASSIFY_SYS,
                "user":     CLASSIFY_USER,
                "answer":   classify_answer,
                "task":     "classify",
            })

            # ── Tâche 2 : Extraction ─────────────────────────────────────────
            extract_user   = EXTRACT_PROMPTS.get(doc_type, EXTRACT_PROMPTS.get("default", "Extract fields. JSON only."))
            extract_answer = json.dumps({
                "document_type": doc_type,
                "confidence":    ex.get("confidence", 0.9),
                "fields":        expected,
                "line_items":    ex.get("line_items", []),
                "raw_text_snippet": ex.get("raw_text_snippet", ""),
            }, ensure_ascii=False)
            self.items.append({
                "image":  img_rel,
                "sys":    EXTRACT_SYS,
                "user":   extract_user,
                "answer": extract_answer,
                "task":   "extract",
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item     = self.items[idx]
        img_path = Path(cfg["dataset_path"]) / item["image"]
        image    = _load_image(img_path)
        messages = _build_messages(image, item["sys"], item["user"], item["answer"])

        text   = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
        )
        out = {k: v.squeeze(0) for k, v in inputs.items()}
        labels = out["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        out["labels"] = labels
        return out


# ─── Callback métriques ───────────────────────────────────────────────────────
class MetricsCallback(TrainerCallback):
    def __init__(self, total_steps: int, wall_start: float, prev_elapsed_s: int = 0):
        self.total_steps    = total_steps
        self.wall_start     = wall_start
        self.prev_elapsed_s = prev_elapsed_s

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        import mlflow
        elapsed = self.prev_elapsed_s + (time.time() - self.wall_start)
        step    = state.global_step
        eta     = (elapsed / max(step, 1)) * (self.total_steps - step)
        loss    = round(logs.get("loss", 0), 4)
        record  = {
            "type":        "train",
            "step":        step,
            "total_steps": self.total_steps,
            "loss":        loss,
            "lr":          logs.get("learning_rate", 0),
            "elapsed_s":   int(elapsed),
            "eta_s":       int(eta),
            "ts":          _utcnow(),
        }
        _log(log_path, record)
        mlflow.log_metrics({"train_loss": loss, "learning_rate": record["lr"]}, step=step)
        print(f"[STEP {step}/{self.total_steps}] loss={loss:.4f} · elapsed={elapsed//60:.0f}m · ETA={eta//60:.0f}m")

    def on_epoch_end(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            import mlflow
            val_loss = round(metrics.get("eval_loss", 0), 4)
            record = {
                "type":      "val",
                "epoch":     state.epoch,
                "val_loss":  val_loss,
                "ts":        _utcnow(),
            }
            _log(log_path, record)
            mlflow.log_metric("val_loss", val_loss, step=int(state.epoch))

    def on_evaluate(self, args, state, control, **kwargs):
        """Vider le cache XPU avant chaque évaluation pour éviter l'OOM."""
        if torch.xpu.is_available():
            torch.xpu.empty_cache()
            torch.xpu.synchronize()


# ─── Callback checkpoint (persistance de l'état entre restarts) ───────────────
class CheckpointCallback(TrainerCallback):
    def __init__(self, state_file: Path, state: dict):
        self.state_file = state_file
        self.state = state

    def on_save(self, args, state, control, **kwargs):
        import mlflow, shutil, tempfile
        self.state["completed_steps"] = state.global_step
        ckpt_dir = Path(args.output_dir)
        checkpoints = sorted(ckpt_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        if checkpoints:
            latest = checkpoints[-1]
            self.state["checkpoint_dir"] = str(latest)

            # Logguer dans MLflow en mode annule-et-remplace :
            # on copie le checkpoint dans un dossier temp nommé "latest_checkpoint"
            # pour que le artifact_path soit toujours le même → remplace le précédent.
            try:
                tmp = Path(tempfile.mkdtemp())
                dst = tmp / "latest_checkpoint"
                shutil.copytree(str(latest), str(dst))
                mlflow.log_artifacts(str(dst), artifact_path="checkpoint/latest")
                shutil.rmtree(str(tmp), ignore_errors=True)
            except Exception as e:
                print(f"[WARN] MLflow artifact checkpoint : {e}")

        self.state_file.write_text(json.dumps(self.state, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[CHECKPOINT] Step {state.global_step} → MLflow artifact/checkpoint/latest")


# ─── Évaluation post-entraînement ─────────────────────────────────────────────
def evaluate_model(model, processor, val_examples: list, device: str) -> dict:
    """Calcule F1 champ-niveau et parse rate sur le set de validation."""
    model.eval()
    parse_ok   = 0
    all_f1     = []
    field_hits = {}
    field_tot  = {}
    val_preds  = []

    for ex in val_examples:
        img_path = Path(cfg["dataset_path"]) / ex["image"]
        if not img_path.exists():
            continue
        image = Image.open(img_path).convert("RGB").resize((512, 512), Image.LANCZOS)
        messages = [
            {"role": "system", "content": "You are a document extraction assistant. Return only valid JSON."},
            {"role": "user",   "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": f"Extract fields from this {ex.get('doc_type','document')}. JSON only."},
            ]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=False)
        decoded = processor.decode(out[0], skip_special_tokens=True)
        # Extraire JSON
        predicted = {}
        try:
            import re
            m = re.search(r'\{[\s\S]+\}', decoded)
            if m:
                predicted = json.loads(m.group())
                parse_ok += 1
        except Exception:
            pass

        expected  = ex.get("expected", {})
        doc_type  = ex.get("doc_type", "default")
        # Champs à évaluer : eval_fields du registry, ou tous si non définis
        eval_keys = _get_eval_fields(doc_type) or list(expected.keys())
        ex_f1 = []
        for k in eval_keys:
            if k not in expected:
                continue
            field_tot[k] = field_tot.get(k, 0) + 1
            pred_fields = predicted.get("fields", predicted)  # supporte fields imbriqués
            v_pred = pred_fields.get(k) if isinstance(pred_fields, dict) else None
            v_exp  = expected[k]
            if v_pred is not None and str(v_pred).strip() == str(v_exp).strip():
                field_hits[k] = field_hits.get(k, 0) + 1
                ex_f1.append(1.0)
            else:
                ex_f1.append(0.0)
        if ex_f1:
            all_f1.append(sum(ex_f1) / len(ex_f1))

        val_preds.append({
            "image": ex["image"],
            "doc_type": ex.get("doc_type", ""),
            "expected": expected,
            "predicted": predicted,
        })

    n = len(val_examples)
    field_f1_per_key = {k: round(field_hits.get(k, 0) / field_tot[k], 3) for k in field_tot}
    result = {
        "parse_success_rate": round(parse_ok / max(n, 1), 3),
        "field_f1":           round(sum(all_f1) / max(len(all_f1), 1), 3),
        "field_f1_per_key":   field_f1_per_key,
        "val_examples":       val_preds[:5],
        "n_evaluated":        n,
        "ts":                 _utcnow(),
    }
    return result


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    import mlflow

    # ── State file persistant (survit aux crashes + restarts Streamlit) ──
    state_file = out_dir / "training_state.json"
    if state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        print(f"[RESUME] Phase : {state['phase']} · step {state.get('completed_steps',0)}/{state.get('total_steps','?')}")
        is_resume = True
    else:
        state = {"phase": "training", "completed_steps": 0, "elapsed_train_s": 0}
        is_resume = False

    _log(log_path, {
        "type": "start", "run_id": run_id,
        "phase": state["phase"], "resumed": is_resume,
        "config": cfg, "ts": _utcnow(),
    })

    # ── MLflow : reprendre le run existant ou en créer un nouveau ──
    mlflow.set_tracking_uri((ROOT / "logs" / "mlruns").as_uri())
    mlflow.set_experiment("qwen_vl_finetuning")
    existing_mlflow_id = state.get("mlflow_run_id")
    if existing_mlflow_id:
        try:
            mlflow_run = mlflow.start_run(run_id=existing_mlflow_id)
        except Exception:
            mlflow_run = mlflow.start_run(run_name=run_id)
            state["mlflow_run_id"] = mlflow_run.info.run_id
    else:
        mlflow_run = mlflow.start_run(run_name=run_id)
        state["mlflow_run_id"] = mlflow_run.info.run_id

    print(f"[INFO] MLflow run : {mlflow_run.info.run_id}")

    if not is_resume:
        mlflow.log_params({
            "model_id":       cfg.get("model_id", ""),
            "device":         cfg.get("device", "cpu"),
            "lora_r":         cfg.get("lora_r", 4),
            "lora_alpha":     cfg.get("lora_alpha", 8),
            "target_modules": ",".join(cfg.get("target_modules", [])),
            "n_epochs":       cfg.get("n_epochs", 3),
            "learning_rate":  cfg.get("learning_rate", 2e-4),
            "batch_size":     cfg.get("batch_size", 1),
            "max_seq_length": cfg.get("max_seq_length", 1024),
            "quant_out":      cfg.get("quant_out", "q4_k_m"),
        })
        mlflow.log_params({f"prompt_{k}": v for k, v in _PROMPT_VERSIONS.items()})

    # ── Dataset split — utilise pct_test/pct_val depuis config, sinon 70/15/15 ──
    labels_file = Path(cfg["dataset_path"]) / "labels.json"
    labels    = json.loads(labels_file.read_text(encoding="utf-8"))
    n_samples = cfg.get("n_samples", len(labels))
    labels    = labels[:n_samples]          # limiter à l'échantillon choisi
    n         = len(labels)
    pct_test  = cfg.get("pct_test", 15)
    pct_val  = cfg.get("pct_val",  15)
    n_test   = max(1, int(n * pct_test / 100))
    n_val    = max(1, int(n * pct_val  / 100))
    n_train  = max(1, n - n_test - n_val)
    print(f"[INFO] Split : {n_train} train ({100-pct_test-pct_val}%) · {n_val} val ({pct_val}%) · {n_test} test ({pct_test}%)")
    train_data = labels[:n_train]
    val_data   = labels[n_train:n_train + n_val]
    test_data  = labels[n_train + n_val:]
    state.setdefault("n_train", n_train)
    state.setdefault("n_val",   n_val)
    state.setdefault("n_test",  n_test)
    state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Dataset : {n_train} train · {n_val} val · {n_test} test")
    if not is_resume:
        mlflow.log_params({"n_train": n_train, "n_val": n_val, "n_test": n_test})
    _log(log_path, {"type": "init", "msg": f"📂 Dataset prêt — {n_train} train · {n_val} val · {n_test} test", "ts": time.time()})

    # ── Charger processor + modèle (toujours nécessaire) ──
    local_path = Path(cfg.get("local_model", ""))
    model_src  = str(local_path) if local_path.exists() else cfg["model_id"]
    print(f"[INFO] Chargement modèle : {model_src}")
    _log(log_path, {"type": "init", "msg": "⚙️ Chargement du processor Qwen2.5-VL…", "ts": time.time()})

    import transformers.models.auto.video_processing_auto as _vpau
    import transformers.processing_utils as _pu
    class _DummyVideoAuto:
        @classmethod
        def from_pretrained(cls, *a, **kw): return None
    _vpau.video_processor_class_from_name = lambda *a, **kw: None
    _vpau.AutoVideoProcessor = _DummyVideoAuto
    _orig_check = _pu.ProcessorMixin.check_argument_for_proper_class
    def _patched_check(self, argument_name, argument):
        if argument_name == "video_processor":
            return type(None)
        return _orig_check(self, argument_name, argument)
    _pu.ProcessorMixin.check_argument_for_proper_class = _patched_check

    # Patch get_attributes sur Qwen2_5_VLProcessor pour exclure video_processor.
    # save_pretrained (transformers 5.x) utilise get_attributes() qui inspecte __init__
    # et trouve video_processor → getattr crash car l'objet est None / non-sérialisable.
    _orig_get_attrs = Qwen2_5_VLProcessor.get_attributes.__func__
    @classmethod
    def _patched_get_attrs(cls):
        return [a for a in _orig_get_attrs(cls) if a != "video_processor"]
    Qwen2_5_VLProcessor.get_attributes = _patched_get_attrs

    processor = Qwen2_5_VLProcessor.from_pretrained(model_src, trust_remote_code=True)
    _log(log_path, {"type": "init", "msg": "🧠 Chargement du modèle Qwen2.5-VL-3B…", "ts": time.time()})
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_src,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if device == "xpu":
        import intel_extension_for_pytorch as ipex
        _log(log_path, {"type": "init", "msg": "🔧 Transfert sur XPU + optimisation IPEX…", "ts": time.time()})
        model = model.to("xpu")
        model = ipex.optimize(model, dtype=torch.float16)
    else:
        model = model.to(device)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1 — ENTRAÎNEMENT
    # ══════════════════════════════════════════════════════════════════
    if state["phase"] == "training":
        lora_cfg = LoraConfig(
            r             = cfg.get("lora_r", 4),
            lora_alpha    = cfg.get("lora_alpha", 8),
            target_modules= cfg.get("target_modules", ["q_proj", "v_proj"]),
            lora_dropout  = 0.05,
            bias          = "none",
            task_type     = TaskType.CAUSAL_LM,
        )
        lora_r = cfg.get("lora_r", 4)
        _log(log_path, {"type": "init", "msg": f"🔗 Application LoRA r={lora_r}…", "ts": time.time()})
        model = get_peft_model(model, lora_cfg)
        trainable, total = model.get_nb_trainable_parameters()
        print(f"[INFO] Params entraînables : {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")
        _log(log_path, {"type": "info", "trainable_params": trainable, "total_params": total})

        max_seq     = cfg.get("max_seq_length", 1024)
        n_epochs    = cfg.get("n_epochs", 3)
        batch_sz    = cfg.get("batch_size", 1)
        lr          = cfg.get("learning_rate", 2e-4)
        _log(log_path, {"type": "init", "msg": "📋 Construction des datasets…", "ts": time.time()})
        ds_train    = DocumentDataset(train_data, processor, max_seq)
        ds_val      = DocumentDataset(val_data,   processor, max_seq)
        total_steps = math.ceil(len(ds_train) / (batch_sz * 4)) * n_epochs
        state["total_steps"] = total_steps
        state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"[INFO] Items : {len(ds_train)} train · {len(ds_val)} val · {total_steps} steps")
        _log(log_path, {"type": "init", "msg": f"📋 {len(ds_train)} train · {len(ds_val)} val — {total_steps} steps au total", "ts": time.time()})

        training_args = TrainingArguments(
            output_dir                  = str(out_dir / "checkpoints"),
            num_train_epochs            = n_epochs,
            per_device_train_batch_size = batch_sz,
            gradient_accumulation_steps = 4,
            learning_rate               = lr,
            warmup_steps                = max(1, int(total_steps * 0.05)),
            lr_scheduler_type           = "cosine",
            fp16                        = (device != "cpu"),
            gradient_checkpointing      = True,
            logging_steps               = 5,
            eval_strategy               = "no",       # évaluation en phase 2 (évite OOM)
            save_strategy               = "steps",
            save_steps                  = 10,
            save_total_limit            = 5,
            load_best_model_at_end      = False,
            remove_unused_columns       = False,
            dataloader_num_workers      = 0,
            report_to                   = "none",
            use_cpu                     = (device == "cpu"),
        )

        prev_elapsed = state.get("elapsed_train_s", 0)
        wall_start   = time.time()
        callbacks = [
            MetricsCallback(total_steps, wall_start, prev_elapsed),
            CheckpointCallback(state_file, state),
        ]

        trainer = Trainer(
            model         = model,
            args          = training_args,
            train_dataset = ds_train,
            callbacks     = callbacks,
        )

        # Reprise depuis dernier checkpoint si disponible
        checkpoint_dir = state.get("checkpoint_dir")
        if checkpoint_dir and Path(checkpoint_dir).exists():
            print(f"[RESUME] Reprise depuis checkpoint : {checkpoint_dir}")
            resume = checkpoint_dir
        else:
            # Chercher automatiquement dans le dossier checkpoints
            ckpt_root = out_dir / "checkpoints"
            existing  = sorted(ckpt_root.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])) if ckpt_root.exists() else []
            resume    = str(existing[-1]) if existing else None
            if resume:
                print(f"[RESUME] Checkpoint auto-détecté : {resume}")

        adapter_dir = out_dir / "adapter"
        adapter_weights = adapter_dir / "adapter_model.safetensors"
        training_already_done = (
            state.get("completed_steps", 0) >= total_steps
            and adapter_weights.exists()
        )

        if training_already_done:
            # Steps déjà terminés mais merge non fait → passer directement à "merging"
            print("[SKIP] Entraînement déjà terminé — passage en phase merge.")
            state["phase"] = "merging"
            state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
            _log(log_path, {"type": "phase", "phase": "merging", "ts": _utcnow()})
            del model
            if device == "xpu":
                torch.xpu.empty_cache()
            if args.one_phase:
                print("[ONE-PHASE] Prêt pour le merge.")
                mlflow.end_run()
                sys.exit(0)
        else:
            _log(log_path, {"type": "init", "msg": "🚀 Démarrage de l'entraînement…", "ts": time.time()})
            _log(log_path, {
                "type": "train", "status": "started",
                "step": state.get("completed_steps", 0),
                "total_steps": total_steps,
                "resumed_from": resume,
                "ts": _utcnow(),
            })

            trainer.train(resume_from_checkpoint=resume)
            elapsed_train = prev_elapsed + int(time.time() - wall_start)
            state["elapsed_train_s"] = elapsed_train
            print(f"[INFO] Entraînement terminé en {elapsed_train/60:.1f} min")

            # Sauvegarder adapter LoRA
            adapter_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(adapter_dir))
            processor.save_pretrained(str(adapter_dir))
            state["adapter_path"] = str(adapter_dir)
            print(f"[INFO] Adapter sauvegardé : {adapter_dir}")

            state["phase"] = "merging"
            state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
            _log(log_path, {"type": "phase", "phase": "merging", "ts": _utcnow()})

            del model
            if device == "xpu":
                torch.xpu.empty_cache()

            if args.one_phase:
                print("[ONE-PHASE] Entraînement terminé — en attente du merge.")
                mlflow.end_run()
                sys.exit(0)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1b — MERGE LoRA → modèle standalone
    # ══════════════════════════════════════════════════════════════════
    if state["phase"] == "merging":
        print("\n[PHASE MERGE] Chargement adapter + merge…")
        _log(log_path, {"type": "init", "msg": "🔀 Chargement adapter pour merge…", "ts": time.time()})

        if device == "xpu" and torch.xpu.is_available():
            torch.xpu.empty_cache()

        # Recharger le modèle base (libéré après training)
        from peft import PeftModel
        merge_base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_src,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            trust_remote_code=True,
        ).to(device)
        merge_base.config.use_cache = False

        adapter_dir = out_dir / "adapter"
        peft_model  = PeftModel.from_pretrained(merge_base, str(adapter_dir))

        _log(log_path, {"type": "init", "msg": "🔀 Merge LoRA → modèle standalone…", "ts": time.time()})
        merged = peft_model.merge_and_unload()
        merged_dir = out_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(merged_dir), safe_serialization=True)

        # Sauvegarder processor si manquant
        if not (adapter_dir / "tokenizer.json").exists():
            processor.save_pretrained(str(adapter_dir))
        processor.save_pretrained(str(merged_dir))

        state["merged_path"] = str(merged_dir)
        print(f"[INFO] Modèle mergé : {merged_dir}")

        state["phase"] = "testing"
        state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        _log(log_path, {"type": "phase", "phase": "testing", "ts": _utcnow()})

        del peft_model, merged, merge_base
        if device == "xpu" and torch.xpu.is_available():
            torch.xpu.empty_cache()

        if args.one_phase:
            print("[ONE-PHASE] Merge terminé — en attente du lancement du test.")
            mlflow.end_run()
            sys.exit(0)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2 — TEST (jeu de données jamais vu pendant l'entraînement)
    # ══════════════════════════════════════════════════════════════════
    if state["phase"] == "testing":
        print(f"\n[PHASE TEST] {len(test_data)} exemples holdout…")
        merged_dir = Path(state.get("merged_path", out_dir / "merged"))

        if device == "xpu" and torch.xpu.is_available():
            torch.xpu.empty_cache()

        test_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(merged_dir),
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            trust_remote_code=True,
        ).to(device)

        test_results = evaluate_model(test_model, processor, test_data, device)
        test_results["set"] = "test"
        state["test_results"] = test_results
        test_path = LOGS_DIR / f"ft_{run_id}_test.json"
        test_path.write_text(json.dumps(test_results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[TEST] F1={test_results['field_f1']:.3f} · parse_rate={test_results['parse_success_rate']:.3f}")
        mlflow.log_metrics({
            "test_field_f1":           test_results["field_f1"],
            "test_parse_success_rate": test_results["parse_success_rate"],
        }, step=state.get("total_steps", 0))
        mlflow.log_artifact(str(test_path), artifact_path="evaluation")

        del test_model
        if device == "xpu" and torch.xpu.is_available():
            torch.xpu.empty_cache()

        state["phase"] = "validation"
        state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        _log(log_path, {"type": "phase", "phase": "validation", "ts": _utcnow()})

        if args.one_phase:
            print("[ONE-PHASE] Test terminé — en attente du lancement de la validation.")
            mlflow.end_run()
            sys.exit(0)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3 — VALIDATION (jeu de validation vu pendant l'entraînement)
    # ══════════════════════════════════════════════════════════════════
    if state["phase"] == "validation":
        print(f"\n[PHASE VALIDATION] {len(val_data)} exemples…")
        merged_dir = Path(state.get("merged_path", out_dir / "merged"))

        if device == "xpu" and torch.xpu.is_available():
            torch.xpu.empty_cache()

        val_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(merged_dir),
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            trust_remote_code=True,
        ).to(device)

        val_results = evaluate_model(val_model, processor, val_data, device)
        val_results["set"] = "val"
        state["val_results"] = val_results
        val_path = LOGS_DIR / f"ft_{run_id}_val.json"
        val_path.write_text(json.dumps(val_results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[VAL] F1={val_results['field_f1']:.3f} · parse_rate={val_results['parse_success_rate']:.3f}")
        mlflow.log_metrics({
            "val_field_f1":           val_results["field_f1"],
            "val_parse_success_rate": val_results["parse_success_rate"],
        }, step=state.get("total_steps", 0))
        mlflow.log_artifact(str(val_path), artifact_path="evaluation")

        del val_model
        if device == "xpu" and torch.xpu.is_available():
            torch.xpu.empty_cache()

        state["phase"] = "done"
        state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    # ══════════════════════════════════════════════════════════════════
    # FINALISATION
    # ══════════════════════════════════════════════════════════════════
    elapsed_total = state.get("elapsed_train_s", 0)
    mlflow.log_metrics({"elapsed_min": round(elapsed_total / 60, 1)})
    mlflow.log_artifact(str(args.config), artifact_path="config")
    mlflow.log_artifact(str(log_path),    artifact_path="logs")
    mlflow.end_run()

    adapter_path = state.get("adapter_path", "")
    merged_path  = state.get("merged_path", "")
    test_f1  = state.get("test_results", {}).get("field_f1", 0)
    val_f1   = state.get("val_results",  {}).get("field_f1", 0)
    _log(log_path, {
        "type":         "train",
        "status":       "done",
        "step":         state.get("total_steps", 0),
        "total_steps":  state.get("total_steps", 0),
        "elapsed_s":    elapsed_total,
        "eta_s":        0,
        "adapter_path": adapter_path,
        "merged_path":  merged_path,
        "test_f1":      test_f1,
        "val_f1":       val_f1,
        "ts":           _utcnow(),
    })
    print(f"[DONE] Run {run_id} — test_f1={test_f1:.3f} · val_f1={val_f1:.3f} · {elapsed_total/60:.1f} min")


if __name__ == "__main__":
    main()

"""
Prompt & Schema Registry — source unique de vérité pour le pipeline Document Intelligence.

┌─────────────────────────────────────────────────────────────────────────┐
│  Architecture fonctionnelle                                             │
│                                                                         │
│  prompts/registry.json                                                  │
│        │                                                                │
│        ▼                                                                │
│  PromptRegistry ──── get(name)   → Prompt (system, user, hash)         │
│                 └─── get_schema(name) → Schema (output_schema,         │
│                                          required_fields, eval_fields)  │
│                                                                         │
│  Consommateurs :                                                        │
│    classify.py   → get_classify_prompts()   → call_vlm()               │
│    extract.py    → get_extract_prompts()    → call_vlm()               │
│    finetune_xpu.py → mêmes prompts + get_eval_fields() pour F1         │
│    03_FineTuning.py → affiche versions + eval_fields dans l'UI         │
│                                                                         │
│  Versioning : MAJOR.MINOR.PATCH (semver)                               │
│    MAJOR = rupture schéma · MINOR = ajout champ · PATCH = reformulation│
│  Immutabilité : une version enregistrée ne peut plus être modifiée     │
│  Intégrité   : SHA-256 calculé sur (system + user) à la création       │
└─────────────────────────────────────────────────────────────────────────┘

Stockage : poc_qwen/prompts/registry.json (versionnable avec git)
"""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import NamedTuple

_REGISTRY_PATH = Path(__file__).parent.parent.parent / "prompts" / "registry.json"


# ─── Types ────────────────────────────────────────────────────────────────────
class Prompt(NamedTuple):
    name:        str
    version:     str
    system:      str
    user:        str
    description: str
    created_at:  str
    hash:        str


class Schema(NamedTuple):
    name:            str
    version:         str
    description:     str
    output_schema:   dict   # définitions de champs typés
    required_fields: list   # champs obligatoires à la sortie
    eval_fields:     list   # champs utilisés pour le calcul F1


# ─── PromptRegistry ───────────────────────────────────────────────────────────
class PromptRegistry:
    """
    Registry de prompts versionné et persisté dans prompts/registry.json.

    Usage rapide :
        from src.pipeline.prompt_registry import registry
        sys, user = registry.get("classify")
        sys, user = registry.get("extract.invoice", version="1.0.0")
    """

    def __init__(self, path: Path = _REGISTRY_PATH):
        self._path = path
        self._data = self._load()

    # ── I/O ──────────────────────────────────────────────────────────────────
    def _load(self) -> dict:
        if self._path.exists():
            return json.loads(self._path.read_text(encoding="utf-8"))
        return {"prompts": {}}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── Hash ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _hash(system: str, user: str) -> str:
        content = f"system:{system}\nuser:{user}"
        return "sha256:" + hashlib.sha256(content.encode()).hexdigest()[:16]

    # ── Lecture ──────────────────────────────────────────────────────────────
    def get(self, name: str, version: str | None = None) -> Prompt:
        """
        Retourne le prompt (system, user) pour `name`.
        Si version=None, retourne la version active.

        Raises KeyError si le prompt ou la version n'existe pas.
        """
        entry = self._data["prompts"].get(name)
        if entry is None:
            raise KeyError(f"Prompt '{name}' introuvable dans le registry.")

        ver = version or entry["active"]
        v   = entry["versions"].get(ver)
        if v is None:
            raise KeyError(f"Version '{ver}' introuvable pour le prompt '{name}'.")

        return Prompt(
            name        = name,
            version     = ver,
            system      = v["system"],
            user        = v["user"],
            description = v.get("description", ""),
            created_at  = v.get("created_at", ""),
            hash        = v.get("hash", ""),
        )

    def list_versions(self, name: str) -> list[dict]:
        """Liste toutes les versions d'un prompt avec métadonnées."""
        entry = self._data["prompts"].get(name, {})
        active = entry.get("active", "")
        return [
            {
                "version":     ver,
                "active":      ver == active,
                "created_at":  info.get("created_at", ""),
                "description": info.get("description", ""),
                "hash":        info.get("hash", ""),
            }
            for ver, info in entry.get("versions", {}).items()
        ]

    def list_prompts(self) -> list[str]:
        """Liste tous les noms de prompts enregistrés."""
        return list(self._data["prompts"].keys())

    def get_active_versions(self) -> dict[str, str]:
        """Retourne {prompt_name: active_version} — utile pour MLflow params."""
        return {
            name: entry.get("active", "")
            for name, entry in self._data["prompts"].items()
        }

    # ── Schémas ──────────────────────────────────────────────────────────────
    def get_schema(self, name: str, version: str | None = None) -> Schema:
        """
        Retourne le schéma JSON de sortie pour `name`.
        Si version=None, retourne la version active.

        Raises KeyError si le schéma ou la version n'existe pas.
        """
        section = self._data.get("schemas", {})
        entry = section.get(name)
        if entry is None:
            raise KeyError(f"Schéma '{name}' introuvable dans le registry.")

        ver = version or entry.get("active", "")
        v   = entry.get("versions", {}).get(ver)
        if v is None:
            raise KeyError(f"Version '{ver}' introuvable pour le schéma '{name}'.")

        return Schema(
            name            = name,
            version         = ver,
            description     = v.get("description", ""),
            output_schema   = v.get("output_schema", {}),
            required_fields = v.get("required_fields", []),
            eval_fields     = v.get("eval_fields", []),
        )

    def get_eval_fields(self, doc_type: str) -> list[str]:
        """
        Retourne les eval_fields pour un type de document — utilisé par
        finetune_xpu.py pour calculer le F1 sur les bons champs.

        Fallback sur liste vide si le schéma n'existe pas.
        """
        name = f"extract.{doc_type}"
        try:
            return self.get_schema(name).eval_fields
        except KeyError:
            try:
                return self.get_schema("extract.default").eval_fields
            except KeyError:
                return []

    def list_schemas(self) -> list[str]:
        """Liste tous les noms de schémas enregistrés."""
        return list(self._data.get("schemas", {}).keys())

    # ── Écriture ─────────────────────────────────────────────────────────────
    def register(
        self,
        name:        str,
        version:     str,
        system:      str,
        user:        str,
        description: str = "",
        set_active:  bool = True,
    ) -> Prompt:
        """
        Enregistre une nouvelle version d'un prompt.
        La version doit être supérieure à toutes les versions existantes.
        Lève ValueError si la version existe déjà (immuabilité).
        """
        if name not in self._data["prompts"]:
            self._data["prompts"][name] = {"active": "", "versions": {}}

        entry = self._data["prompts"][name]
        if version in entry["versions"]:
            raise ValueError(
                f"Version '{version}' de '{name}' existe déjà — les versions sont immuables. "
                f"Utilisez une nouvelle version (ex: {self._next_patch(version)})."
            )

        h = self._hash(system, user)
        entry["versions"][version] = {
            "created_at":  str(date.today()),
            "description": description,
            "hash":        h,
            "system":      system,
            "user":        user,
        }
        if set_active:
            entry["active"] = version

        self._save()
        return self.get(name, version)

    def set_active(self, name: str, version: str) -> None:
        """Promeut une version existante comme version active."""
        entry = self._data["prompts"].get(name)
        if entry is None:
            raise KeyError(f"Prompt '{name}' introuvable.")
        if version not in entry["versions"]:
            raise KeyError(f"Version '{version}' introuvable pour '{name}'.")
        entry["active"] = version
        self._save()

    def diff(self, name: str, v1: str, v2: str) -> dict:
        """Retourne les différences textuelles entre deux versions."""
        import difflib
        p1 = self.get(name, v1)
        p2 = self.get(name, v2)
        return {
            "name": name,
            "v1": v1,
            "v2": v2,
            "system_diff": list(difflib.unified_diff(
                p1.system.splitlines(), p2.system.splitlines(),
                fromfile=f"{name}@{v1}", tofile=f"{name}@{v2}", lineterm=""
            )),
            "user_diff": list(difflib.unified_diff(
                p1.user.splitlines(), p2.user.splitlines(),
                fromfile=f"{name}@{v1}", tofile=f"{name}@{v2}", lineterm=""
            )),
        }

    # ── Helpers internes ─────────────────────────────────────────────────────
    @staticmethod
    def _next_patch(version: str) -> str:
        """Incrémente le patch d'une version semver."""
        try:
            parts = version.split(".")
            parts[-1] = str(int(parts[-1]) + 1)
            return ".".join(parts)
        except Exception:
            return version + ".1"


# ── Singleton global ───────────────────────────────────────────────────────────
registry = PromptRegistry()


# ── Recalcul des hashes au chargement (intégrité) ────────────────────────────
def _ensure_hashes() -> None:
    """Ajoute les hashes manquants (migration depuis anciens fichiers)."""
    changed = False
    for name, entry in registry._data["prompts"].items():
        for ver, info in entry.get("versions", {}).items():
            if not info.get("hash"):
                info["hash"] = PromptRegistry._hash(info.get("system", ""), info.get("user", ""))
                changed = True
    if changed:
        registry._save()

_ensure_hashes()


# ── Accesseurs de commodité ───────────────────────────────────────────────────
def get_schema(name: str, version: str | None = None) -> "Schema":
    """Raccourci global : registry.get_schema(name, version)."""
    return registry.get_schema(name, version)


def get_eval_fields(doc_type: str) -> list[str]:
    """Raccourci global : registry.get_eval_fields(doc_type)."""
    return registry.get_eval_fields(doc_type)


def get_classify_prompts(version: str | None = None) -> tuple[str, str]:
    """Retourne (system, user) pour la classification (version active si None)."""
    p = registry.get("classify", version)
    return p.system, p.user


def get_extract_prompts(doc_type: str = "default", version: str | None = None) -> tuple[str, str]:
    """Retourne (system, user) pour l'extraction du type donné."""
    name = f"extract.{doc_type}"
    # Fallback sur default si le type n'est pas dans le registry
    if name not in registry.list_prompts():
        name = "extract.default"
    p = registry.get(name, version)
    return p.system, p.user


# ── Exports rétrocompatibles (classify.py, extract.py) ───────────────────────
CLASSIFY_SYS,  CLASSIFY_USER  = get_classify_prompts()
EXTRACT_SYSTEM                 = registry.get("extract.default").system
EXTRACT_TEMPLATES: dict[str, dict] = {
    name.replace("extract.", ""): {
        "version": registry.get(name).version,
        "user":    registry.get(name).user,
    }
    for name in registry.list_prompts()
    if name.startswith("extract.")
}

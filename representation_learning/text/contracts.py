"""`TextEvidenceItem`: the validated input contract of the text encoders (Phase 3, G2).

Source of truth: docs/data_contracts.md section 3 (JSON Schema `text_evidence_item.schema.json`).
This module re-implements that schema in plain Python (no `jsonschema` at run time; the tests parse
the schema from the document and cross-check both) and adds the encoder-boundary rules below.
Nothing here imports torch/transformers or touches the network.

Rules beyond the JSON Schema (each one is deliberate and documented in docs/phase3_text_encoder.md):
  * `original_text_sha256` must equal SHA-256 of `original_text` encoded as UTF-8 (no Unicode or
    newline normalisation). The contract says only "hash of original_text"; UTF-8/no-normalisation is
    the assumption made here.
  * `translation.text` gets the same 20000-character bound as `original_text` (the schema leaves it
    unbounded; untrusted text is length-limited before tokenization).
  * `original_text` / `translation.text` must not be blank, must be encodable as UTF-8 (no lone
    surrogates) and must not contain NUL. Text is never modified, trimmed or normalised.
  * `evidence_id` is generated UPSTREAM. The encoder validates its form only: exactly `te_` followed
    by 16 lowercase hexadecimal characters (`^te_[0-9a-f]{16}$`, the agreed format; stricter than the
    schema's `minLength: 8`). It does not verify, and does not claim to verify, how the id was derived.
  * `language` is preserved verbatim; it must be a non-blank string. There is NO language allow-list
    and no BCP-47 syntax check (the schema only says "string"); it is metadata and never selects a
    model, a prefix or a code path.
  * Unknown top-level keys are rejected (the schema has `additionalProperties: false`);
    `source`, `nlp_context`, `location` may carry extra keys (the schema allows them).
  * Timestamps must parse as ISO-8601 date-times with an explicit UTC offset.
"""

from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional

from .errors import InvalidInputError

SCHEMA_VERSION = "0.1"
SOURCE_KINDS = ("article", "advisory", "event_description", "snippet")
TEXT_VARIANTS = ("original_text", "translation")
MAX_TEXT_CHARS = 20000                       # schema: original_text maxLength
EVIDENCE_ID_PATTERN = r"te_[0-9a-f]{16}"     # agreed upstream format (schema only says minLength 8)

_EVIDENCE_ID_RE = re.compile(EVIDENCE_ID_PATTERN)     # always used with fullmatch (no `$` newline quirk)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_TOP_KEYS = {"schema_version", "evidence_id", "source_kind", "source", "original_text",
             "original_text_sha256", "language", "translation", "truncated", "nlp_context",
             "location", "sensitivity"}
_REQUIRED = ("schema_version", "evidence_id", "source_kind", "source", "original_text",
             "original_text_sha256", "language", "truncated", "sensitivity")
_TRANSLATION_KEYS = {"target_language", "text", "text_sha256", "method", "model_version"}


def text_sha256(text: str) -> str:
    """SHA-256 (hex) of `text` as UTF-8. The hash the encoder expects in `original_text_sha256`
    and `translation.text_sha256`. Raises UnicodeEncodeError for lone surrogates."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_str(v: Any) -> bool:
    return isinstance(v, str)


def _nullable_str(v: Any) -> bool:
    return v is None or isinstance(v, str)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _datetime_problem(label: str, v: Any) -> Optional[str]:
    if not isinstance(v, str) or not v:
        return f"{label} must be an ISO-8601 date-time string"
    # datetime.fromisoformat accepts a trailing 'Z' only from Python 3.11; normalise exactly one
    # uppercase 'Z' (RFC 3339 UTC) so Python 3.10 behaves like 3.11+. Nothing else is loosened.
    # Only a 'Z' directly after a digit counts ('...ZZ' and '... Z' stay malformed: Python 3.10 would
    # otherwise accept them once the 'Z' is rewritten).
    parse = re.sub(r"(?<=\d)Z\Z", "+00:00", v)
    try:
        dt = datetime.fromisoformat(parse)
    except ValueError:
        return f"{label} is not a valid ISO-8601 date-time: {v!r}"
    if dt.tzinfo is None or dt.utcoffset() is None:
        return f"{label} must include a UTC offset (e.g. 'Z' or '+00:00'): {v!r}"
    return None


def _text_problems(label: str, v: Any) -> List[str]:
    if not isinstance(v, str):
        return [f"{label} must be a string"]
    out: List[str] = []
    if not v.strip():
        out.append(f"{label} is empty or whitespace-only")
    if len(v) > MAX_TEXT_CHARS:      # schema limit for original_text; applied to translation.text too
        out.append(f"{label} has {len(v)} characters (maximum {MAX_TEXT_CHARS}); "
                   "it is rejected, never truncated")
    if "\x00" in v:
        out.append(f"{label} contains a NUL character")
    try:
        v.encode("utf-8")
    except UnicodeEncodeError:
        out.append(f"{label} is not encodable as UTF-8 (lone surrogate)")
    return out


def _id_problems(label: str, v: Any) -> List[str]:
    if not isinstance(v, str):
        return [f"{label} must be a string"]
    if _EVIDENCE_ID_RE.fullmatch(v) is None:
        shown = v if len(v) <= 40 else v[:40] + "..."
        return [f"{label} must be 'te_' followed by 16 lowercase hexadecimal characters "
                f"(got {shown!r})"]
    return []


def text_evidence_problems(d: Any) -> List[str]:
    """Every contract violation of `d` (empty list = valid). Never raises."""
    if not isinstance(d, Mapping):
        return ["text evidence must be a JSON object (dict)"]
    out: List[str] = []
    extra = sorted(set(d) - _TOP_KEYS, key=str)
    if extra:
        out.append(f"unknown keys {extra} (the contract allows no additional properties)")
    for k in _REQUIRED:
        if k not in d:
            out.append(f"missing required key {k!r}")
    if "schema_version" in d and d["schema_version"] != SCHEMA_VERSION:
        out.append(f"schema_version must be {SCHEMA_VERSION!r}")
    if "evidence_id" in d:
        out.extend(_id_problems("evidence_id", d["evidence_id"]))
    if "source_kind" in d and d["source_kind"] not in SOURCE_KINDS:
        out.append(f"source_kind must be one of {list(SOURCE_KINDS)}")

    src = d.get("source")
    if "source" in d:
        if not isinstance(src, Mapping):
            out.append("source must be an object")
        else:
            if not _is_str(src.get("source_name")) or not src.get("source_name", "").strip():
                out.append("source.source_name must be a non-empty string")
            if "retrieved_at" not in src:
                out.append("source.retrieved_at is required")
            else:
                p = _datetime_problem("source.retrieved_at", src["retrieved_at"])
                if p:
                    out.append(p)
            for k in ("source_url", "source_item_id"):
                if k in src and not _nullable_str(src[k]):
                    out.append(f"source.{k} must be a string or null")
            if src.get("published_at") is not None and "published_at" in src:
                p = _datetime_problem("source.published_at", src["published_at"])
                if p:
                    out.append(p)

    if "original_text" in d:
        out.extend(_text_problems("original_text", d["original_text"]))
    if "original_text_sha256" in d:
        h = d["original_text_sha256"]
        if not isinstance(h, str) or not _HEX64.match(h):
            out.append("original_text_sha256 must be 64 lowercase hex characters")
        elif isinstance(d.get("original_text"), str) and not _text_problems_hard(d["original_text"]):
            if text_sha256(d["original_text"]) != h:
                out.append("original_text_sha256 does not match SHA-256 of original_text (UTF-8)")
    if "language" in d:
        lang = d["language"]
        if not isinstance(lang, str) or not lang.strip():
            out.append("language must be a non-blank string (BCP-47 tag, preserved verbatim)")

    if "translation" in d:
        tr = d["translation"]
        if not isinstance(tr, Mapping):
            out.append("translation must be an object when present")
        else:
            miss = sorted(_TRANSLATION_KEYS - set(tr))
            extra_tr = sorted(set(tr) - _TRANSLATION_KEYS, key=str)
            if miss:
                out.append(f"translation is missing keys {miss}")
            if extra_tr:
                out.append(f"translation has unknown keys {extra_tr}")
            for k in ("target_language", "method", "model_version"):
                if k in tr and (not isinstance(tr[k], str) or not tr[k].strip()):
                    out.append(f"translation.{k} must be a non-blank string")
            if "text" in tr:
                out.extend(_text_problems("translation.text", tr["text"]))
            if "text_sha256" in tr:
                h = tr["text_sha256"]
                if not isinstance(h, str) or not _HEX64.match(h):
                    out.append("translation.text_sha256 must be 64 lowercase hex characters")
                elif isinstance(tr.get("text"), str) and not _text_problems_hard(tr["text"]):
                    if text_sha256(tr["text"]) != h:
                        out.append("translation.text_sha256 does not match SHA-256 of translation.text")

    if "truncated" in d and not isinstance(d["truncated"], bool):
        out.append("truncated must be a boolean")
    if "nlp_context" in d and not isinstance(d["nlp_context"], Mapping):
        out.append("nlp_context must be an object when present")
    if "location" in d:
        loc = d["location"]
        if not isinstance(loc, Mapping):
            out.append("location must be an object when present")
        else:
            if "place_label" in loc and not isinstance(loc["place_label"], str):
                out.append("location.place_label must be a string")
            for k in ("lat", "lon"):
                if k in loc and not _is_number(loc[k]):
                    out.append(f"location.{k} must be a number")
    if "sensitivity" in d:
        s = d["sensitivity"]
        if not isinstance(s, Mapping):
            out.append("sensitivity must be an object")
        else:
            for k in ("may_contain_personal_data", "retain_text"):
                if not isinstance(s.get(k), bool):
                    out.append(f"sensitivity.{k} is required and must be a boolean")
    return out


def _text_problems_hard(v: str) -> bool:
    """True if `v` cannot be hashed (not UTF-8 encodable)."""
    try:
        v.encode("utf-8")
    except UnicodeEncodeError:
        return True
    return False


@dataclass(frozen=True)
class TextEvidenceItem:
    """A validated, immutable copy of a Phase 1 `TextEvidenceItem`.

    Build it with `TextEvidenceItem.from_dict` (strict validation). The mapping fields are private
    deep copies, so later changes to the caller's dict cannot alter the item; `to_dict()` returns a
    fresh deep copy equal to the validated input. `original_text` and `language` are held verbatim.
    """

    schema_version: str
    evidence_id: str
    source_kind: str
    source: Dict[str, Any]
    original_text: str
    original_text_sha256: str
    language: str
    truncated: bool
    sensitivity: Dict[str, Any]
    translation: Optional[Dict[str, Any]] = None
    nlp_context: Optional[Dict[str, Any]] = None
    location: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        # Direct construction is validated too, so an instance can never carry an invalid contract.
        probs = text_evidence_problems(self._raw())
        if probs:
            raise InvalidInputError("invalid TextEvidenceItem: " + "; ".join(probs))

    def _raw(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "schema_version": self.schema_version, "evidence_id": self.evidence_id,
            "source_kind": self.source_kind, "source": self.source,
            "original_text": self.original_text,
            "original_text_sha256": self.original_text_sha256, "language": self.language,
            "truncated": self.truncated, "sensitivity": self.sensitivity}
        for k in ("translation", "nlp_context", "location"):
            if getattr(self, k) is not None:
                out[k] = getattr(self, k)
        return out

    @classmethod
    def from_dict(cls, d: Any) -> "TextEvidenceItem":
        probs = text_evidence_problems(d)
        if probs:
            raise InvalidInputError("invalid TextEvidenceItem: " + "; ".join(probs))
        g = copy.deepcopy(dict(d))
        return cls(**{k: g.get(k) for k in _TOP_KEYS if k in g})

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._raw())

    def select(self, text_variant: str) -> "SelectedText":
        """The text of one variant plus the language that describes it. `translation` is valid only
        if the item has one (data_contracts.md section 4, `text_variant`)."""
        if text_variant == "original_text":
            return SelectedText("original_text", self.original_text, self.original_text_sha256,
                                self.language)
        if text_variant == "translation":
            if self.translation is None:
                raise InvalidInputError(
                    "text_variant 'translation' requested but the item has no translation")
            return SelectedText("translation", self.translation["text"],
                                self.translation["text_sha256"],
                                self.translation["target_language"])
        raise InvalidInputError(f"text_variant must be one of {list(TEXT_VARIANTS)}, "
                                f"got {text_variant!r}")


@dataclass(frozen=True)
class SelectedText:
    variant: str
    text: str
    sha256: str
    language: str


def coerce_item(item: Any) -> TextEvidenceItem:
    """Accept a `TextEvidenceItem` or its dict form. Raw strings are refused: every encoded text
    needs an upstream-generated evidence id and provenance (no ad-hoc text)."""
    if isinstance(item, TextEvidenceItem):
        return item
    if isinstance(item, str):
        raise InvalidInputError(
            "a raw string cannot be encoded; pass a TextEvidenceItem (dict) with an upstream "
            "evidence_id and original_text_sha256")
    return TextEvidenceItem.from_dict(item)

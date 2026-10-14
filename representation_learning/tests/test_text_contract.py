"""Phase 3 / G2: `TextEvidenceItem` contract tests. Offline, no model, no torch.
The Phase 1 JSON Schema is parsed from docs/data_contracts.md (single source of truth) and used as
a cross-check: the Python validator may be STRICTER than the schema, never looser."""

import copy
import unicodedata

import pytest
from jsonschema import Draft202012Validator

from representation_learning.text.contracts import (MAX_TEXT_CHARS,
                                                    TextEvidenceItem, coerce_item,
                                                    text_evidence_problems, text_sha256)
from representation_learning.text.errors import InvalidInputError
from representation_learning.testing.text_mocks import make_text_item


@pytest.fixture
def schema_v(schemas):
    return Draft202012Validator(schemas["text_evidence_item.schema.json"])


def mutated(fn, **kw):
    d = make_text_item(translation_text="Example translation", **kw)
    fn(d)
    return d


# ------------------------------------------------------------------ valid items
def test_valid_item_roundtrips_and_matches_schema(schema_v):
    d = make_text_item(translation_text="Hello", nlp_context={"nlp_model_version": "n1", "entities": []},
                       location={"place_label": "X", "lat": 1.5, "lon": -2})
    assert not list(schema_v.iter_errors(d))
    item = TextEvidenceItem.from_dict(d)
    assert item.to_dict() == d and text_evidence_problems(d) == []


@pytest.mark.parametrize("lang", ["en", "ta", "zh-Hant-TW", "und", "x-private", "tlh", "xx-ZZ",
                                  "sr-Latn", "UNKNOWN-LANG"])
def test_language_is_preserved_verbatim_with_no_allow_list(lang):
    item = TextEvidenceItem.from_dict(make_text_item(language=lang))
    assert item.language == lang


@pytest.mark.parametrize("text", [
    "  leading and trailing  \n", "line1\nline2\r\n\ttabbed", "é (NFD) vs é (NFC)",
    "உதாரண உரை", "مرحبا بالعالم", "混合 text with 日本語", "emoji \U0001F6A8 test",
    "query: already looks like a prefix"])
def test_original_text_is_preserved_verbatim(text):
    d = make_text_item(text)
    item = TextEvidenceItem.from_dict(d)
    assert item.original_text == text and item.to_dict()["original_text"] == text
    assert item.original_text_sha256 == text_sha256(text)


def test_no_unicode_normalisation_hash_distinguishes_nfc_and_nfd():
    nfc, nfd = unicodedata.normalize("NFC", "é"), unicodedata.normalize("NFD", "é")
    assert nfc != nfd and text_sha256(nfc) != text_sha256(nfd)
    with pytest.raises(InvalidInputError, match="does not match"):
        TextEvidenceItem.from_dict(make_text_item(nfc, original_text_sha256=text_sha256(nfd)))


def test_item_is_an_isolated_copy_of_the_input():
    d = make_text_item(nlp_context={"k": ["a"]})
    item = TextEvidenceItem.from_dict(d)
    d["source"]["source_name"] = "changed"
    d["nlp_context"]["k"].append("b")
    assert item.source["source_name"] == "synthetic" and item.nlp_context == {"k": ["a"]}
    out = item.to_dict()
    out["source"]["source_name"] = "x"
    assert item.source["source_name"] == "synthetic"
    with pytest.raises(Exception):
        item.language = "fr"                                   # frozen


def test_direct_construction_is_validated_too():
    d = make_text_item()
    d["original_text_sha256"] = "0" * 64
    with pytest.raises(InvalidInputError, match="does not match"):
        TextEvidenceItem(**d)


# ------------------------------------------------------------------ rejected items
BAD = [
    ("not-a-dict", None, "JSON object"),
    ("missing-text", lambda d: d.pop("original_text"), "original_text"),
    ("missing-hash", lambda d: d.pop("original_text_sha256"), "original_text_sha256"),
    ("missing-language", lambda d: d.pop("language"), "language"),
    ("missing-sensitivity", lambda d: d.pop("sensitivity"), "sensitivity"),
    ("missing-truncated", lambda d: d.pop("truncated"), "truncated"),
    ("unknown-top-key", lambda d: d.update(normalized_text="x"), "unknown keys"),
    ("wrong-schema-version", lambda d: d.update(schema_version="9"), "schema_version"),
    ("bad-source-kind", lambda d: d.update(source_kind="tweet"), "source_kind"),
    ("short-id", lambda d: d.update(evidence_id="te_1"), "evidence_id"),
    ("long-id", lambda d: d.update(evidence_id="e" * 257), "evidence_id"),
    ("id-with-space", lambda d: d.update(evidence_id="te 0000001"), "te_"),
    ("id-with-newline", lambda d: d.update(evidence_id="te_0123456789abcdef\n"), "te_"),
    ("id-with-control", lambda d: d.update(evidence_id="te_\x07000001"), "te_"),
    ("id-not-string", lambda d: d.update(evidence_id=12345678), "evidence_id"),
    ("empty-text", lambda d: d.update(original_text=""), "empty"),
    ("blank-text", lambda d: d.update(original_text=" \n\t "), "empty"),
    ("text-too-long", lambda d: d.update(original_text="a" * (MAX_TEXT_CHARS + 1)), "rejected, never truncated"),
    ("text-with-nul", lambda d: d.update(original_text="a\x00b"), "NUL"),
    ("text-lone-surrogate", lambda d: d.update(original_text="a\ud800b"), "UTF-8"),
    ("text-not-string", lambda d: d.update(original_text=["a"]), "string"),
    ("hash-uppercase", lambda d: d.update(original_text_sha256="A" * 64), "64 lowercase hex"),
    ("hash-short", lambda d: d.update(original_text_sha256="ab"), "64 lowercase hex"),
    ("hash-mismatch", lambda d: d.update(original_text_sha256="0" * 64), "does not match"),
    ("blank-language", lambda d: d.update(language="  "), "language"),
    ("language-not-string", lambda d: d.update(language=["en"]), "language"),
    ("truncated-not-bool", lambda d: d.update(truncated="false"), "truncated"),
    ("sensitivity-missing-key", lambda d: d["sensitivity"].pop("retain_text"), "retain_text"),
    ("sensitivity-not-bool", lambda d: d["sensitivity"].update(may_contain_personal_data="no"), "may_contain"),
    ("source-not-object", lambda d: d.update(source="x"), "source must be"),
    ("source-no-name", lambda d: d["source"].pop("source_name"), "source_name"),
    ("source-no-retrieved", lambda d: d["source"].pop("retrieved_at"), "retrieved_at"),
    ("retrieved-not-date", lambda d: d["source"].update(retrieved_at="yesterday"), "date-time"),
    ("retrieved-no-offset", lambda d: d["source"].update(retrieved_at="2026-10-02T00:00:00"), "UTC offset"),
    ("published-bad", lambda d: d["source"].update(published_at="soon"), "published_at"),
    ("source-url-int", lambda d: d["source"].update(source_url=5), "source_url"),
    ("translation-not-object", lambda d: d.update(translation="x"), "translation must be"),
    ("translation-missing-key", lambda d: d["translation"].pop("method"), "missing keys"),
    ("translation-extra-key", lambda d: d["translation"].update(note="x"), "unknown keys"),
    ("translation-hash-mismatch", lambda d: d["translation"].update(text_sha256="0" * 64), "translation.text_sha256"),
    ("translation-too-long", lambda d: d["translation"].update(text="a" * (MAX_TEXT_CHARS + 1),
                                                              text_sha256=text_sha256("a" * (MAX_TEXT_CHARS + 1))), "translation.text has"),
    ("translation-blank", lambda d: d["translation"].update(text=" "), "translation.text"),
    ("nlp-context-list", lambda d: d.update(nlp_context=[1]), "nlp_context"),
    ("location-lat-bool", lambda d: d.update(location={"lat": True}), "location.lat"),
    ("location-label-int", lambda d: d.update(location={"place_label": 3}), "place_label"),
]


@pytest.mark.parametrize("fn,expect", [(b[1], b[2]) for b in BAD], ids=[b[0] for b in BAD])
def test_invalid_item_is_rejected(fn, expect, schema_v):
    d = None if fn is None else mutated(fn)
    probs = text_evidence_problems(d)
    assert probs and any(expect in p for p in probs), probs
    with pytest.raises(InvalidInputError, match="invalid TextEvidenceItem"):
        TextEvidenceItem.from_dict(d)


def test_python_validator_is_never_looser_than_the_phase1_schema(schema_v):
    """Everything the encoder-side validator accepts also satisfies the Phase 1 schema, and every
    BAD case above is rejected by the Python validator (the schema may or may not reject it)."""
    accepted = [make_text_item(), make_text_item(translation_text="t"),
                make_text_item(location={"lat": 1, "lon": 2}, nlp_context={"a": 1}),
                make_text_item(language="und"), make_text_item(truncated=True)]
    for d in accepted:
        assert text_evidence_problems(d) == [] and not list(schema_v.iter_errors(d))


@pytest.mark.parametrize("fn", [
    lambda d: d.update(original_text="   ", original_text_sha256=text_sha256("   ")),
    lambda d: d.update(original_text_sha256="0" * 64),
    lambda d: d.update(language=" "),
    lambda d: d["source"].update(retrieved_at="2026-10-02T00:00:00"),
    lambda d: d.update(evidence_id="te_" + "x" * 16),
    lambda d: d.update(evidence_id="te 0000001"),
    lambda d: d.update(evidence_id="abcdefgh"),
], ids=["blank-text", "hash-mismatch", "blank-language", "no-utc-offset", "id-not-hex", "id-space",
        "id-no-prefix"])
def test_documented_extra_strictness_over_the_schema(fn, schema_v):
    """Rules the Phase 1 schema does not state (listed in the contract module docstring): the
    schema alone would accept these items, the encoder boundary does not."""
    d = mutated(fn)
    assert not list(schema_v.iter_errors(d))
    assert text_evidence_problems(d)


# ------------------------------------------------------------------ variants / coercion
def test_select_original_and_translation():
    item = TextEvidenceItem.from_dict(make_text_item("Original", language="ta", translation_text="Translated"))
    o, t = item.select("original_text"), item.select("translation")
    assert (o.text, o.language, o.sha256) == ("Original", "ta", text_sha256("Original"))
    assert (t.text, t.language, t.sha256) == ("Translated", "en", text_sha256("Translated"))
    assert item.original_text == "Original"                    # the translation never replaces it


def test_translation_variant_requires_a_translation_and_known_variants():
    item = TextEvidenceItem.from_dict(make_text_item())
    with pytest.raises(InvalidInputError, match="no translation"):
        item.select("translation")
    with pytest.raises(InvalidInputError, match="text_variant"):
        item.select("normalized_text")


def test_coerce_item_accepts_dict_and_item_and_refuses_raw_strings():
    d = make_text_item()
    item = coerce_item(d)
    assert coerce_item(item) is item
    with pytest.raises(InvalidInputError, match="raw string"):
        coerce_item("just a string")
    with pytest.raises(InvalidInputError):
        coerce_item(None)
    with pytest.raises(InvalidInputError):
        coerce_item(copy.deepcopy(d) | {"original_text_sha256": "0" * 64})

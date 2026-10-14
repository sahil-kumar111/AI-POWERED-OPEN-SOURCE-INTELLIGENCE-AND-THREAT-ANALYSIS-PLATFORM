"""Valid and invalid examples for the Phase 2 amendments to the Phase 1 contracts
(original_text/language/translation on TextEvidenceItem; optional text_variant on
EmbeddingRecord). Schemas are parsed from docs/data_contracts.md."""

import copy

import pytest
from jsonschema import Draft202012Validator

H = lambda c: c * 64  # noqa: E731

TEXT = {"schema_version": "0.1", "evidence_id": "te_00000001", "source_kind": "advisory",
        "source": {"source_name": "example", "retrieved_at": "2026-10-02T00:00:00Z"},
        "original_text": "உதாரண உரை", "original_text_sha256": H("0"), "language": "ta",
        "truncated": False,
        "sensitivity": {"may_contain_personal_data": False, "retain_text": True}}
TRANSLATION = {"target_language": "en", "text": "Example text", "text_sha256": H("1"),
               "method": "TBD", "model_version": "TBD"}


@pytest.fixture
def text_v(schemas):
    return Draft202012Validator(schemas["text_evidence_item.schema.json"])


@pytest.fixture
def emb_v(schemas):
    return Draft202012Validator(schemas["embedding_record.schema.json"])


@pytest.fixture
def visual_record():
    from representation_learning.testing.mocks import mock_encoder
    import numpy as np
    return mock_encoder("clip").encode(np.zeros((64, 64, 3), np.uint8), color_order="RGB").record


def _text_embedding(rec, **kw):
    r = copy.deepcopy(rec)
    r.update(modality="text", evidence_id="te_00000001", preprocessing={"tokenizer": "TBD"}, **kw)
    return r


def test_text_valid_original_and_translation_separate(text_v):
    assert not list(text_v.iter_errors(TEXT))
    with_tr = dict(TEXT, translation=TRANSLATION)
    assert not list(text_v.iter_errors(with_tr))
    assert with_tr["original_text"] == TEXT["original_text"]       # translation never replaces it


@pytest.mark.parametrize("mutate", [
    lambda t: t.pop("original_text"),
    lambda t: t.pop("original_text_sha256"),
    lambda t: t.pop("language"),
    lambda t: (t.pop("original_text"), t.pop("original_text_sha256"),
               t.update(normalized_text="x", text_sha256=H("0"))),          # legacy fields
    lambda t: t.update(original_language="ta"),                              # removed field
    lambda t: t.update(translation={k: v for k, v in TRANSLATION.items() if k != "text"}),
    lambda t: t.update(translation=dict(TRANSLATION, extra="x")),
], ids=["no-original", "no-hash", "no-language", "legacy-fields", "removed-original_language",
        "translation-no-text", "translation-extra-field"])
def test_text_invalid(text_v, mutate):
    bad = copy.deepcopy(TEXT)
    mutate(bad)
    assert list(text_v.iter_errors(bad))


def test_embedding_visual_without_text_variant_valid(emb_v, visual_record):
    assert "text_variant" not in visual_record
    assert not list(emb_v.iter_errors(visual_record))


@pytest.mark.parametrize("variant", ["original_text", "translation", None])
def test_embedding_text_variant_valid_on_text_records(emb_v, visual_record, variant):
    rec = _text_embedding(visual_record) if variant is None else \
        _text_embedding(visual_record, text_variant=variant)
    assert not list(emb_v.iter_errors(rec))


@pytest.mark.parametrize("variant", ["original_text", "translation"])
def test_embedding_text_variant_rejected_on_visual_records(emb_v, visual_record, variant):
    assert list(emb_v.iter_errors(dict(visual_record, text_variant=variant)))


def test_embedding_text_variant_unknown_value_rejected(emb_v, visual_record):
    assert list(emb_v.iter_errors(_text_embedding(visual_record, text_variant="summary")))

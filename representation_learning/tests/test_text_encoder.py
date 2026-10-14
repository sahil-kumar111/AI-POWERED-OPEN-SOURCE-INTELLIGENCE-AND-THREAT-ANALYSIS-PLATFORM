"""Phase 3 / G2: text encoder interface + both candidate adapters, with MOCK backends only.
No weights, no torch, no network (conftest blocks sockets). These tests verify input formatting,
token-limit behaviour, output shape/normalisation, error handling and the EmbeddingRecord - NOT the
quality of E5 or MiniLM embeddings."""

import json

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from representation_learning.encoders.datatypes import EncodeFailure, EncoderConfig
from representation_learning.testing.text_mocks import (SPECIAL_TOKENS, MockTextBackend,
                                                        make_text_item, mock_text_encoder)
from representation_learning.text.adapters import (E5_SPEC, MINILM_SPEC, E5TextEncoder,
                                                   MiniLMTextEncoder)
from representation_learning.text.contracts import TextEvidenceItem
from representation_learning.text.errors import (EmbeddingValidationError, EncoderIntegrityError,
                                                 EncoderLoadError, InvalidInputError,
                                                 TextTooLongError)

CANDS = ["e5", "minilm"]


def words(n):
    return " ".join(["w"] * n)


def role_for(c):
    return "passage" if c == "e5" else None


@pytest.fixture(params=CANDS)
def cand(request):
    return request.param


def fwd(enc):
    return enc._backend.forward_texts


# ------------------------------------------------------------------ input formatting
def test_e5_prefixes_are_exact_and_follow_the_callers_role():
    enc = mock_text_encoder("e5")
    enc.encode(make_text_item("hello world"), role="query")
    enc.encode(make_text_item("hello world"), role="passage")
    assert fwd(enc) == ["query: hello world", "passage: hello world"]      # note the trailing space
    assert E5_SPEC.role_prefixes == {"query": "query: ", "passage": "passage: "}


def test_e5_text_is_passed_verbatim_after_the_prefix():
    enc = mock_text_encoder("e5")
    t = "  Spaces \n and é and 日本語 \U0001F6A8  "
    enc.encode(make_text_item(t), role="query")
    assert fwd(enc) == ["query: " + t]


@pytest.mark.parametrize("role", [None, "", "Query", "PASSAGE", "document", 1, b"query", {"query"}])
def test_e5_requires_a_valid_role(role):
    enc = mock_text_encoder("e5")
    with pytest.raises(InvalidInputError, match="role"):
        enc.encode(make_text_item(), role=role)
    assert fwd(enc) == []                                                # nothing reached the model


def test_encode_takes_scalars_only():
    enc = mock_text_encoder("e5")
    with pytest.raises(InvalidInputError, match="single role"):
        enc.encode(make_text_item(), role=["query"])
    with pytest.raises(InvalidInputError, match="single role"):
        enc.encode(make_text_item(), role="query", text_variant=["original_text"])


def test_minilm_never_adds_a_prefix_and_role_is_optional():
    enc = mock_text_encoder("minilm")
    enc.encode(make_text_item("hello world"))
    enc.encode(make_text_item("hello world"), role="query")
    enc.encode(make_text_item("hello world"), role="passage")
    assert fwd(enc) == ["hello world"] * 3
    assert MINILM_SPEC.role_prefixes is None and MINILM_SPEC.role_required is False


def test_minilm_still_validates_a_role_it_does_not_use():
    with pytest.raises(InvalidInputError, match="role"):
        mock_text_encoder("minilm").encode(make_text_item(), role="document")


def test_minilm_role_has_no_effect_on_vector_but_is_recorded():
    enc = mock_text_encoder("minilm")
    a, b = enc.encode(make_text_item()), enc.encode(make_text_item(), role="query")
    assert np.array_equal(a.vector, b.vector)
    assert a.record["source_ref"]["input_role"] is None
    assert b.record["source_ref"]["input_role"] == "query"
    assert b.record["source_ref"]["role_prefix_applied"] is False


def test_e5_query_and_passage_differ_but_share_one_space():
    enc = mock_text_encoder("e5")
    q, p = enc.encode(make_text_item(), role="query"), enc.encode(make_text_item(), role="passage")
    assert not np.array_equal(q.vector, p.vector)
    assert q.space_id == p.space_id and q.embedding_id != p.embedding_id
    assert q.record["source_ref"]["role_prefix_applied"] is True


def test_nlp_context_and_metadata_are_never_concatenated_into_the_text(cand):
    enc = mock_text_encoder(cand)
    item = make_text_item("plain text", nlp_context={"keywords": ["SECRETKEYWORD"], "nlp_model_version": "n9"},
                          location={"place_label": "SECRETPLACE"})
    r = enc.encode(item, role=role_for(cand))
    assert all("SECRET" not in t for t in fwd(enc))
    assert r.record["upstream"]["nlp_model_version"] is None          # G2.1: never read from nlp_context


def test_language_is_metadata_only_and_never_changes_the_model_input_or_vector(cand):
    enc = mock_text_encoder(cand)
    en = enc.encode(make_text_item("same text", language="en"), role=role_for(cand))
    ta = enc.encode(make_text_item("same text", language="ta"), role=role_for(cand))
    odd = enc.encode(make_text_item("same text", language="xx-made-up"), role=role_for(cand))
    assert np.array_equal(en.vector, ta.vector) and np.array_equal(en.vector, odd.vector)
    assert len(set(fwd(enc))) == 1
    assert (en.record["source_ref"]["language"], ta.record["source_ref"]["language"],
            odd.record["source_ref"]["language"]) == ("en", "ta", "xx-made-up")


def test_translation_variant_embeds_the_translation_and_says_so(cand):
    enc = mock_text_encoder(cand)
    item = make_text_item("original உரை", language="ta", translation_text="the translation")
    o = enc.encode(item, role=role_for(cand))
    t = enc.encode(item, role=role_for(cand), text_variant="translation")
    assert o.record["text_variant"] == "original_text" and t.record["text_variant"] == "translation"
    assert [x.split(": ", 1)[-1] for x in fwd(enc)] == ["original உரை", "the translation"]
    assert o.record["source_ref"]["encoded_text_language"] == "ta"
    assert t.record["source_ref"]["encoded_text_language"] == "en"
    assert t.record["source_ref"]["language"] == "ta"                  # language of the ORIGINAL
    assert "translation_variant_embedded" in t.record["warnings"]
    assert o.embedding_id != t.embedding_id and not np.array_equal(o.vector, t.vector)


def test_translation_variant_without_a_translation_is_rejected(cand):
    with pytest.raises(InvalidInputError, match="no translation"):
        mock_text_encoder(cand).encode(make_text_item(), role=role_for(cand), text_variant="translation")


@pytest.mark.parametrize("v", [None, "", "normalized_text", "Original_Text"])
def test_unknown_text_variant_is_rejected(cand, v):
    with pytest.raises(InvalidInputError, match="text_variant"):
        mock_text_encoder(cand).encode(make_text_item(), role=role_for(cand), text_variant=v)


# ------------------------------------------------------------------ token limits
def test_spec_limits_are_the_decided_values():
    assert E5_SPEC.max_input_tokens == 512 and MINILM_SPEC.max_input_tokens == 128
    assert E5_SPEC.token_limit_policy == MINILM_SPEC.token_limit_policy == "reject_over_limit"


@pytest.mark.parametrize("cand_,limit,prefix_words", [("e5", 512, 1), ("minilm", 128, 0)])
def test_limit_boundary_exact_ok_one_over_rejected(cand_, limit, prefix_words):
    enc = mock_text_encoder(cand_)
    n_ok = limit - SPECIAL_TOKENS - prefix_words              # model input is exactly `limit` tokens
    r = enc.encode(make_text_item(words(n_ok)), role=role_for(cand_))
    assert r.record["quality"]["token_count"] == limit
    assert r.record["quality"]["max_input_tokens"] == limit
    with pytest.raises(TextTooLongError) as ei:
        enc.encode(make_text_item(words(n_ok + 1)), role=role_for(cand_))
    assert ei.value.token_count == limit + 1 and ei.value.limit == limit
    assert isinstance(ei.value, InvalidInputError)


def test_e5_prefix_counts_toward_the_limit():
    enc = mock_text_encoder("e5")
    text = words(512 - SPECIAL_TOKENS)                         # fits WITHOUT the prefix, not with it
    assert MockTextBackend.token_count(text) == 512
    with pytest.raises(TextTooLongError):
        enc.encode(make_text_item(text), role="query")


def test_over_limit_text_is_rejected_never_truncated_and_never_reaches_the_model(cand):
    enc = mock_text_encoder(cand)
    limit = enc.spec.max_input_tokens
    with pytest.raises(TextTooLongError, match="never truncated"):
        enc.encode(make_text_item(words(limit * 3)), role=role_for(cand))
    assert fwd(enc) == [] and enc._backend.forward_batches == []


def test_over_limit_in_a_batch_fails_only_that_item_and_the_rest_are_encoded_untruncated(cand):
    enc = mock_text_encoder(cand)
    limit = enc.spec.max_input_tokens
    items = [make_text_item("short one", evidence_id="te_0000000000000001"),
             make_text_item(words(limit + 5), evidence_id="te_0000000000000002"),
             make_text_item("short two", evidence_id="te_0000000000000003")]
    res = enc.encode_batch(items, role=role_for(cand))
    assert isinstance(res[1], EncodeFailure) and res[1].index == 1
    assert res[1].error_type == "TextTooLongError"
    assert res[0].record["evidence_id"] == "te_0000000000000001" and res[2].record["evidence_id"] == "te_0000000000000003"
    assert [t.split(": ", 1)[-1] for t in fwd(enc)] == ["short one", "short two"]
    assert res[0].record["inference"]["batch_size"] == 2       # the rejected item is not in the forward batch
    with pytest.raises(TextTooLongError):
        enc.encode_batch(items, role=role_for(cand), on_error="raise")


def test_token_counts_come_from_the_backend_with_special_tokens(cand):
    enc = mock_text_encoder(cand)
    r = enc.encode(make_text_item("a b c"), role=role_for(cand))
    extra = 1 if cand == "e5" else 0
    assert r.record["quality"]["token_count"] == 3 + extra + SPECIAL_TOKENS


# ------------------------------------------------------------------ output
def test_output_dimension_dtype_norm_finite_readonly(cand):
    enc = mock_text_encoder(cand)
    r = enc.encode(make_text_item(), role=role_for(cand))
    assert r.vector.shape == (384,) and r.vector.dtype == np.float32 and r.dim == 384
    assert np.isfinite(r.vector).all() and not r.vector.flags.writeable
    assert abs(float(np.linalg.norm(r.vector.astype(np.float64))) - 1.0) < 1e-5
    assert abs(r.record["norm"] - 1.0) < 1e-5


def test_encoder_normalises_unnormalised_backend_output(cand):
    enc = mock_text_encoder(cand)
    raw = enc._backend.forward(["x y z"])
    assert abs(float(np.linalg.norm(raw)) - 1.0) > 1.0                    # the mock is NOT unit-norm
    r = enc.encode(make_text_item("x y z"), role=role_for(cand))
    assert abs(float(np.linalg.norm(r.vector)) - 1.0) < 1e-5


def test_determinism_and_text_sensitivity(cand):
    enc = mock_text_encoder(cand)
    a = enc.encode(make_text_item("alpha beta"), role=role_for(cand))
    b = enc.encode(make_text_item("alpha beta"), role=role_for(cand))
    c = enc.encode(make_text_item("alpha gamma"), role=role_for(cand))
    assert np.array_equal(a.vector, b.vector) and a.embedding_id == b.embedding_id
    assert not np.array_equal(a.vector, c.vector)


def test_batch_matches_single_and_preserves_order_and_chunking(cand):
    enc = mock_text_encoder(cand)
    items = [make_text_item(f"text number {i}", evidence_id=f"te_{i:016x}") for i in range(5)]
    singles = [enc.encode(it, role=role_for(cand)) for it in items]
    enc2 = mock_text_encoder(cand)
    res = enc2.encode_batch(items, role=role_for(cand), batch_size=2)
    assert enc2._backend.forward_batches == [2, 2, 1]
    assert [r.record["evidence_id"] for r in res] == [it["evidence_id"] for it in items]
    assert [r.record["inference"]["batch_size"] for r in res] == [2, 2, 2, 2, 1]
    for s, r in zip(singles, res):
        assert np.array_equal(s.vector, r.vector)


def test_batch_with_per_item_roles_and_variants():
    enc = mock_text_encoder("e5")
    items = [make_text_item("one", evidence_id="te_0000000000000001"),
             make_text_item("two", evidence_id="te_0000000000000002", translation_text="deux")]
    res = enc.encode_batch(items, role=["query", "passage"], text_variant=["original_text", "translation"])
    assert fwd(enc) == ["query: one", "passage: deux"]
    assert [r.record["text_variant"] for r in res] == ["original_text", "translation"]
    with pytest.raises(InvalidInputError, match="length"):
        enc.encode_batch(items, role=["query"])


def test_empty_batch_is_an_empty_result(cand):
    assert mock_text_encoder(cand).encode_batch([], role=role_for(cand)) == []


# ------------------------------------------------------------------ error handling
@pytest.mark.parametrize("fault,msg", [("nan", "NaN or Inf"), ("inf", "NaN or Inf"),
                                       ("zeros", "zero norm"), ("wrong_shape", "output shape"),
                                       ("raise", "backend forward failed"),
                                       ("int_dtype", "non-floating")])
def test_bad_model_output_is_refused(cand, fault, msg):
    enc = mock_text_encoder(cand, fault=fault)
    with pytest.raises(EmbeddingValidationError, match=msg):
        enc.encode(make_text_item(), role=role_for(cand))
    res = enc.encode_batch([make_text_item(evidence_id="te_0000000000000001"),
                            make_text_item("other", evidence_id="te_0000000000000002")], role=role_for(cand))
    assert all(isinstance(r, EncodeFailure) and r.error_type == "EmbeddingValidationError" for r in res)


@pytest.mark.parametrize("fault,msg", [("raise", "tokenization failed"), ("wrong_len", "positive integers"),
                                       ("zero", "positive integers"), ("float", "positive integers")])
def test_bad_token_counts_are_refused_before_inference(cand, fault, msg):
    enc = mock_text_encoder(cand, count_fault=fault)
    items = [make_text_item(evidence_id="te_0000000000000001"), make_text_item("b", evidence_id="te_0000000000000002")]
    with pytest.raises(EmbeddingValidationError, match=msg):
        enc.encode_batch(items, role=role_for(cand), on_error="raise")
    assert fwd(enc) == []


def test_invalid_items_in_a_batch_are_collected_with_their_index(cand):
    enc = mock_text_encoder(cand)
    bad_hash = make_text_item("x", evidence_id="te_0000000000000002")
    bad_hash["original_text_sha256"] = "0" * 64
    res = enc.encode_batch([make_text_item(evidence_id="te_0000000000000001"), bad_hash, "raw string", None],
                           role=role_for(cand))
    assert not isinstance(res[0], EncodeFailure)
    assert [(r.index, r.error_type) for r in res[1:]] == [(1, "InvalidInputError"), (2, "InvalidInputError"),
                                                          (3, "InvalidInputError")]
    assert "raw string" in res[2].message
    with pytest.raises(InvalidInputError):
        enc.encode_batch([bad_hash], role=role_for(cand), on_error="raise")
    with pytest.raises(ValueError):
        enc.encode_batch([], on_error="skip")


def test_raw_strings_and_missing_evidence_are_refused(cand):
    enc = mock_text_encoder(cand)
    for bad in ("hello", None, {}, {"original_text": "hello"}):
        with pytest.raises(InvalidInputError):
            enc.encode(bad, role=role_for(cand))
    assert fwd(enc) == []


def test_evidence_id_reused_for_different_text_rejects_every_member_of_the_group(cand):
    enc = mock_text_encoder(cand)
    items = [make_text_item("first text", evidence_id="te_dddddddd00000001"),
             make_text_item("second text", evidence_id="te_dddddddd00000001"),
             make_text_item("fine", evidence_id="te_0000000000000009"),
             make_text_item("first text", evidence_id="te_5a5a5a5a00000001"),
             make_text_item("first text", evidence_id="te_5a5a5a5a00000001")]
    res = enc.encode_batch(items, role=role_for(cand))
    assert [isinstance(r, EncodeFailure) for r in res] == [True, True, False, False, False]
    assert "different texts" in res[0].message


def test_evidence_ids_are_validated_not_generated(cand):
    enc = mock_text_encoder(cand)
    r = enc.encode(make_text_item(evidence_id="te_0123456789abcdef"), role=role_for(cand))
    assert r.record["evidence_id"] == "te_0123456789abcdef"            # copied verbatim from upstream
    for bad in ("short", "has space in it", "ctrl\x00chars!!", "upstream-id/with:chars#1"):
        with pytest.raises(InvalidInputError):
            enc.encode(make_text_item(evidence_id=bad), role=role_for(cand))


def test_accepts_a_prebuilt_item_and_a_dict_equally(cand):
    enc = mock_text_encoder(cand)
    d = make_text_item("same")
    a = enc.encode(d, role=role_for(cand))
    b = enc.encode(TextEvidenceItem.from_dict(d), role=role_for(cand))
    assert np.array_equal(a.vector, b.vector)


# ------------------------------------------------------------------ lazy loading / backend identity
def test_adapters_are_lazy_and_fail_cleanly_without_weights():
    for cls in (E5TextEncoder, MiniLMTextEncoder):
        enc = cls(EncoderConfig())
        assert not enc.loaded
        with pytest.raises(EncoderLoadError, match="weights_dir is not configured"):
            enc.encode(make_text_item(), role="query")
        assert not enc.loaded


def test_backend_with_the_wrong_dimension_is_refused():
    with pytest.raises(EncoderIntegrityError, match="backend dim"):
        E5TextEncoder(backend=MockTextBackend(512))
    with pytest.raises(EncoderIntegrityError, match="backend dim"):
        MiniLMTextEncoder(backend=MockTextBackend(768))


def test_space_id_depends_on_encoder_weights_and_not_on_role_or_text():
    e5, mini = mock_text_encoder("e5"), mock_text_encoder("minilm")
    assert e5.space_id != mini.space_id
    assert mock_text_encoder("e5", seed=1).space_id != e5.space_id        # different weights hash
    a = e5.encode(make_text_item("x"), role="query")
    b = e5.encode(make_text_item("y y"), role="passage")
    assert a.space_id == b.space_id == e5.space_id
    again = mock_text_encoder("e5")
    assert again.space_id == e5.space_id                                   # deterministic


# ------------------------------------------------------------------ record vs Phase 1 EmbeddingRecord
@pytest.fixture
def emb_v(schemas):
    return Draft202012Validator(schemas["embedding_record.schema.json"])


@pytest.mark.parametrize("variant", ["original_text", "translation"])
def test_record_conforms_to_the_phase1_embedding_record_schema(cand, variant, emb_v):
    enc = mock_text_encoder(cand, run_id="run-1")
    item = make_text_item("Distinct-Marker-Text ✓", translation_text="Distinct-Translation-Marker",
                          nlp_context={"nlp_model_version": "nlp-3"},
                          sensitivity={"may_contain_personal_data": True, "retain_text": False})
    rec = enc.encode(item, role=role_for(cand), text_variant=variant).record
    assert not list(emb_v.iter_errors(rec)), list(emb_v.iter_errors(rec))
    assert rec["modality"] == "text" and rec["text_variant"] == variant
    assert rec["evidence_id"] == item["evidence_id"] and rec["created_by_run_id"] == "run-1"
    assert rec["sensitivity"] == {"may_contain_personal_data": True, "retain_text": False}
    assert rec["quality"]["char_count"] == len(item["original_text"] if variant == "original_text"
                                               else item["translation"]["text"])
    assert rec["quality"]["encoder_truncated"] is False
    blob = json.dumps(rec, ensure_ascii=False)
    assert "Distinct-Marker-Text" not in blob and "Distinct-Translation-Marker" not in blob  # pointer, not copy
    assert rec["source_ref"]["original_text_sha256"] == item["original_text_sha256"]
    assert "mock_backend_not_a_real_model" in rec["warnings"]
    assert rec["upstream"]["nlp_model_version"] is None


def test_upstream_truncation_flag_is_carried_not_hidden(cand, emb_v):
    rec = mock_text_encoder(cand).encode(make_text_item(truncated=True), role=role_for(cand)).record
    assert rec["quality"]["truncated"] is True and rec["quality"]["encoder_truncated"] is False
    assert "upstream_text_truncated" in rec["warnings"]
    assert not list(emb_v.iter_errors(rec))


def test_record_preprocessing_documents_the_candidate_specific_handling():
    e5 = mock_text_encoder("e5").encode(make_text_item(), role="query").record["preprocessing"]
    mini = mock_text_encoder("minilm").encode(make_text_item()).record["preprocessing"]
    assert e5["role_prefixes"] == {"query": "query: ", "passage": "passage: "} and e5["max_input_tokens"] == 512
    assert mini["role_prefixes"] is None and mini["max_input_tokens"] == 128
    for p in (e5, mini):
        assert p["truncation"] == "never" and p["token_limit_policy"] == "reject_over_limit"
        assert p["pooling"] == "mean" and p["output_normalization"] == "l2"
        assert "metadata only" in p["language_handling"]

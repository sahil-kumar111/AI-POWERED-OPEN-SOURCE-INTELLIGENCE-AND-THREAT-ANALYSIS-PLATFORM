# Phase 3 source-adapter integration and E5 candidate note

## Scope and status

This change adds source adapters only. It does not change `TextEvidenceItem`, the encoder
implementations, the model manifests, the frozen Phase 1/2 files, or checksum guards. The
existing `text/adapters.py` remains the E5/MiniLM encoder-adapter module.

The supplied review bundle contains the `threat_detector` location-news module, but does not
contain the local NLP Intelligence implementation. `nlp_source_adapter.py` therefore targets the
previously described `process_record` output shape defensively; it must be checked against the
actual local NLP output before being treated as fully integrated.

## Location-news adapter

`location_source_adapter.py` maps one result from `threat_detector.analysis_engine.analyze_threats`
to one `TextEvidenceItem`. It embeds article title and description only. Category, severity,
distance, geographic classification, and incident coordinates remain metadata. `retrieved_at` is
mandatory and supplied by the caller. Language defaults to `und`, because the result shape does not
provide a trustworthy detected-language field.

The adapter intentionally does not deduplicate rows. Its deterministic evidence ID follows the
documented `(source_id, original_text_sha256)` inputs, with source identity including URL (when
available), publisher, publication timestamp, title, category, and incident location/coordinates.
This distinguishes different incident contexts that share an article URL. If a URL is absent, a
stable source identity is derived from the available article/incident metadata.

Text over the contract limit is rejected, never truncated. Missing article text or publisher name is
rejected rather than replaced with fabricated provenance.

## NLP adapter

`nlp_source_adapter.py` embeds only the caller-supplied `original_description`, never the generated
`analysis_text`. It requires that the exact description occurs in `analysis_text`; otherwise it
rejects the record. The caller supplies `retrieved_at` and, if known, `language`; absent a trusted
language value, `und` is used. NLP entities, threats, events, and model metadata are retained as
metadata, not appended to the embedding text.

**Integration caveat:** the source implementation was not present in the supplied ZIP, so its local
output contract has not been verified here. Confirm the exact `process_record` field names and
description inclusion behavior before wiring this adapter into the application.

## E5 candidate evidence (preliminary)

Previously recorded local validation reports the `intfloat/multilingual-e5-small` checkpoint loaded
from local files, produced finite normalized 384-dimensional embeddings for English, Hindi, and Tamil
examples, behaved deterministically in the repeated-input check, and showed matching single/batch
vectors for the tested short passage. A deliberately over-limit input was rejected instead of
truncated. A three-case retrieval sanity check returned the expected passage at rank 1 in all three
cases (3/3).

These are implementation smoke checks, not a representative retrieval benchmark. The three examples
are too few to establish multilingual or threat-domain retrieval quality, and MiniLM was not evaluated
against a real checkpoint in this run. E5 is therefore a **candidate recommendation**, not a final
selection. The manifest must remain `status: candidate` and `final_encoder: false`.

## Validation limits

The uploaded review ZIP omits `representation_learning/encoders/`, which the text error module imports.
Consequently, the focused tests could not be collected in this isolated review environment without
the omitted project dependencies. Run the focused tests in the complete local repository after
copying the new source files and test file. No full-suite pass is claimed here.

"""Regression: `source.retrieved_at` / `published_at` accept a single trailing uppercase 'Z' on every
supported Python (datetime.fromisoformat only understands 'Z' from 3.11), while naive and malformed
values stay rejected. Offline; no model."""

import pytest

from representation_learning.testing.text_mocks import make_text_item
from representation_learning.text.contracts import (TextEvidenceItem, _datetime_problem,
                                                    text_evidence_problems)
from representation_learning.text.errors import InvalidInputError

FIELDS = ["retrieved_at", "published_at"]


def _item(field, value):
    d = make_text_item()
    d["source"][field] = value
    return d


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("value", ["2026-10-02T00:00:00Z", "2026-10-02T00:00:00+00:00",
                                   "2026-10-02T05:30:00+05:30", "2026-10-02T00:00:00.123456Z",
                                   "2026-10-02T00:00:00-07:00"])
def test_utc_and_offset_timestamps_are_accepted(field, value):
    d = _item(field, value)
    assert text_evidence_problems(d) == []
    assert TextEvidenceItem.from_dict(d).to_dict() == d          # value preserved verbatim, 'Z' included


@pytest.mark.parametrize("value", ["2026-10-02T00:00:00", "2026-10-02"])
def test_naive_timestamps_still_need_an_offset(value):
    assert "UTC offset" in (_datetime_problem("x", value) or "")
    assert any("UTC offset" in p for p in text_evidence_problems(_item("retrieved_at", value)))


@pytest.mark.parametrize("value", ["2026-10-02T00:00:00ZZ", "2026-10-02T00:00:00+00:00Z",
                                   "2026-10-02T00:00:00Z ", " 2026-10-02T00:00:00Z", "Z", "ZZ", "+00:00Z",
                                   "yesterday", "2026-13-02T00:00:00Z", "2026-10-02T25:00:00Z",
                                   "2026-10-02T00:00:00Z\n"])
def test_malformed_values_are_rejected(value):
    problem = _datetime_problem("x", value)
    assert problem and "date-time" in problem
    assert any("date-time" in p for p in text_evidence_problems(_item("retrieved_at", value)))


@pytest.mark.parametrize("value", ["", None, 5, 1.5, True, ["2026-10-02T00:00:00Z"], b"2026-10-02T00:00:00Z"])
def test_empty_and_non_string_values_are_rejected(value):
    assert "ISO-8601 date-time string" in (_datetime_problem("x", value) or "")
    problems = text_evidence_problems(_item("retrieved_at", value))
    assert problems and any("retrieved_at" in p for p in problems)
    with pytest.raises(InvalidInputError):
        TextEvidenceItem.from_dict(_item("retrieved_at", value))


def test_published_at_may_still_be_null_but_not_malformed():
    assert text_evidence_problems(_item("published_at", None)) == []
    assert text_evidence_problems(_item("published_at", "soon"))
    assert text_evidence_problems(_item("published_at", "2026-10-02T00:00:00ZZ"))


def test_only_the_trailing_z_is_normalised(monkeypatch):
    """The parser sees '+00:00' in place of exactly one trailing 'Z', and the original string otherwise."""
    from representation_learning.text import contracts
    seen = []
    real = contracts.datetime

    class Spy:
        @staticmethod
        def fromisoformat(v):
            seen.append(v)
            return real.fromisoformat(v)
    monkeypatch.setattr(contracts, "datetime", Spy)
    assert _datetime_problem("x", "2026-10-02T00:00:00Z") is None
    assert _datetime_problem("x", "2026-10-02T00:00:00+05:30") is None
    assert seen == ["2026-10-02T00:00:00+00:00", "2026-10-02T00:00:00+05:30"]

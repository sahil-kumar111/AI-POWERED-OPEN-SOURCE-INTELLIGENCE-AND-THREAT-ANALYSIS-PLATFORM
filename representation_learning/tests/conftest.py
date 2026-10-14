import socket

import numpy as np
import pytest
from PIL import Image

from representation_learning.testing.contracts import load_phase1_schemas


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any attempt to open a network connection fails the test (Phase 2 rule)."""
    def _blocked(*a, **k):
        raise AssertionError("network access attempted during a unit test")
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


@pytest.fixture(scope="session")
def schemas():
    return load_phase1_schemas()


@pytest.fixture
def rgb_img():
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)


@pytest.fixture
def pil_img(rgb_img):
    return Image.fromarray(rgb_img, "RGB")


def make_evidence(**over):
    ev = {
        "schema_version": "0.1", "evidence_id": "ve_test000001", "source_kind": "frame",
        "video_uri": "input/example.mp4", "video_sha256": "ab" * 32, "frame_index": 3,
        "timestamp_sec": 0.1, "fps": 30.0, "resolution": [160, 120],
        "sensitivity": {"may_contain_persons": True, "retain_pixels": False},
    }
    ev.update(over)
    return ev


@pytest.fixture
def evidence_factory():
    return make_evidence

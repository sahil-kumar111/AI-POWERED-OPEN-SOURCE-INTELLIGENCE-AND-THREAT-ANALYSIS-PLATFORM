"""Weights manifest: pins source, revision, licence, local path and SHA-256 per checkpoint.

The manifest (`representation_learning/models.manifest.json`) is committed with
placeholders; weights themselves stay local and git-ignored. Nothing here downloads.

CLI (run from the repository root):
  python -m representation_learning.encoders.manifest status
  python -m representation_learning.encoders.manifest record dinov2-small \
         --revision <40-hex commit> --license "<licence text from the model card>"
  python -m representation_learning.encoders.manifest verify [name ...]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .datatypes import EncoderConfig
from .errors import EncoderIntegrityError, EncoderLoadError
from .integrity import sha256_file

PLACEHOLDER = "TO_RECORD"
DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "models.manifest.json"
REQUIRED_FILES = ("model.safetensors", "config.json", "preprocessor_config.json")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def load_manifest(path: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(path or DEFAULT_MANIFEST)
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError) as e:
        raise EncoderLoadError(f"cannot read manifest {p}: {e}") from e


def _entry(manifest: Dict[str, Any], name: str) -> Dict[str, Any]:
    try:
        return manifest["models"][name]
    except KeyError:
        raise EncoderLoadError(
            f"unknown manifest entry {name!r}; available: {sorted(manifest.get('models', {}))}")


def pin_problems(entry: Dict[str, Any]) -> List[str]:
    """What is still missing before this entry may be used (empty list = fully pinned)."""
    out = []
    if not _HEX40.match(str(entry.get("revision", ""))):
        out.append("revision is not a 40-hex commit")
    lic = str(entry.get("license", PLACEHOLDER))
    if lic in ("", PLACEHOLDER):
        out.append("license not recorded")
    files = entry.get("files", {})
    for f in REQUIRED_FILES:
        if not _HEX64.match(str(files.get(f, ""))):
            out.append(f"sha256 of {f} not recorded")
    return out


def local_dir(entry: Dict[str, Any], manifest_path: Optional[Path] = None) -> Path:
    base = Path(manifest_path or DEFAULT_MANIFEST).resolve().parent
    return (base / entry["local_dir"]).resolve()


def record_entry(manifest_path: Path, name: str, revision: str, license_text: str) -> Dict[str, Any]:
    """Hash the local files and write revision/licence/hashes into the manifest."""
    if not _HEX40.match(revision):
        raise EncoderIntegrityError("revision must be the full 40-character commit hash")
    if not license_text.strip() or license_text == PLACEHOLDER:
        raise EncoderIntegrityError("licence text must be provided (copy it from the model card)")
    m = load_manifest(manifest_path)
    e = _entry(m, name)
    d = local_dir(e, manifest_path)
    files = {}
    for f in REQUIRED_FILES:
        fp = d / f
        if not fp.is_file():
            raise EncoderLoadError(f"{fp} not found; download the checkpoint first")
        files[f] = sha256_file(fp)
    e.update({"revision": revision, "license": license_text, "files": files,
              "recorded_on": datetime.now(timezone.utc).strftime("%Y-%m-%d")})
    Path(manifest_path).write_text(json.dumps(m, indent=2) + "\n")
    return e


def verify_entry(manifest: Dict[str, Any], name: str, manifest_path: Optional[Path] = None) -> Path:
    """Refuse unless the entry is fully pinned and every local file matches its SHA-256."""
    e = _entry(manifest, name)
    probs = pin_problems(e)
    if probs:
        raise EncoderLoadError(f"manifest entry {name!r} is not pinned yet: " + "; ".join(probs))
    d = local_dir(e, manifest_path)
    bad = []
    for f, want in e["files"].items():
        fp = d / f
        if not fp.is_file():
            bad.append(f"{f}: missing")
        elif sha256_file(fp) != want:
            bad.append(f"{f}: SHA-256 mismatch")
    if bad:
        raise EncoderIntegrityError(f"{name}: " + "; ".join(bad))
    return d


def encoder_from_manifest(name: str, manifest_path: Optional[Path] = None, device: str = "cpu"):
    """Build a (lazy) encoder from a fully pinned, verified manifest entry."""
    from .clip import ClipEncoder
    from .dinov2 import Dinov2Encoder

    mp = Path(manifest_path or DEFAULT_MANIFEST)
    m = load_manifest(mp)
    e = _entry(m, name)
    d = verify_entry(m, name, mp)
    cfg = EncoderConfig(weights_dir=d, expected_weights_sha256=e["files"]["model.safetensors"],
                        checkpoint_id=f"{e['hf_id']}@{e['revision']}", device=device,
                        license=e["license"], revision=e["revision"])
    cls = {"DINOv2": Dinov2Encoder, "CLIP": ClipEncoder}[e["family"]]
    return cls(e["variant"], cfg)


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    r = sub.add_parser("record")
    r.add_argument("name")
    r.add_argument("--revision", required=True)
    r.add_argument("--license", required=True, dest="license_text")
    v = sub.add_parser("verify")
    v.add_argument("names", nargs="*")
    a = ap.parse_args(argv)
    m = load_manifest(a.manifest)
    try:
        if a.cmd == "status":
            for n, e in m["models"].items():
                p = pin_problems(e)
                print(f"{n}: {'PINNED' if not p else 'NOT PINNED - ' + '; '.join(p)}  "
                      f"[{e['hf_id']} -> {e['local_dir']}]")
        elif a.cmd == "record":
            e = record_entry(a.manifest, a.name, a.revision, a.license_text)
            print(f"recorded {a.name}: revision {e['revision']}, "
                  f"model.safetensors sha256 {e['files']['model.safetensors']}")
        else:
            names = a.names or list(m["models"])
            for n in names:
                verify_entry(m, n, a.manifest)
                print(f"{n}: OK (all files match recorded SHA-256)")
    except (EncoderLoadError, EncoderIntegrityError) as ex:
        print(f"error: {type(ex).__name__}: {ex}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(_main())

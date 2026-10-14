"""Timestamp-only patch tool (stdlib only; Python 3.10+).  Modes: preflight (read-only) | apply | rollback.
Touches ONLY text/contracts.py (backed up first) and adds tests/test_text_timestamp_compat.py."""
import argparse, ast, hashlib, re, shutil, subprocess, sys, zipfile
from pathlib import Path

ORIG_CONTRACTS = "e7749ca7aed6f3eea20d23e59b376d9034cc81594f38abef39c9535ce650928a"   # G2.2 original
ZIP_SHA = "3026745e023a22426a55cd1588fae81fe88ff9e98b6382c8fa0a39b5166812b5"
NEW = {"text/contracts.py": "206bf54516b50edbd00e2276b5f427729ae2dadc8d59f0eb035538bac6e7f1ac",
       "tests/test_text_timestamp_compat.py": "fc515283b54cbfef9e0e5452697250b35bcc63a1a8b604e0b1e0dcbeb013d4d7"}
FROZEN = ["tests/phase2_baseline.sha256", "tests/test_phase2_unchanged.py", "tests/test_text_manifest.py",
          "text/manifest.py", "text/__init__.py", "text_models.manifest.json",
          "docs/phase3_text_manifest.md", "models.manifest.json", "evaluation/measure.py"]


def sha(b): return hashlib.sha256(b).hexdigest()
def say(tag, msg): print(f"[{tag}] {msg}")


def preflight(pkg, zpath, quiet=False):
    ok = True
    def chk(cond, msg, fail_msg=None):
        nonlocal ok
        say("PASS" if cond else "FAIL", msg if cond else (fail_msg or msg))
        ok &= bool(cond)
        return cond
    say("INFO", f"python {sys.version.split()[0]}  package dir: {pkg}")
    chk(pkg.is_dir() and (pkg / "text" / "contracts.py").is_file(), "package dir has text/contracts.py")
    if not ok: return False
    zb = zpath.read_bytes()
    chk(sha(zb) == ZIP_SHA, f"zip SHA-256 matches {ZIP_SHA[:12]}…", f"zip SHA-256 is {sha(zb)}, expected {ZIP_SHA}")
    side = zpath.with_name(zpath.name + ".sha256")
    if side.is_file():
        chk(side.read_text().split()[0].lower() == sha(zb), "zip matches its .sha256 file")
    z = zipfile.ZipFile(zpath)
    names = sorted(z.namelist())
    chk(names == sorted(NEW), f"archive has exactly the 2 expected files {names}", f"UNEXPECTED archive contents: {names}")
    for n in names:
        if n in NEW:
            chk(sha(z.read(n)) == NEW[n], f"{n} hash matches manifest")
    cur = (pkg / "text" / "contracts.py").read_bytes()
    cs = sha(cur)
    say("INFO", f"current text/contracts.py SHA-256 = {cs}")
    if cs == ORIG_CONTRACTS:
        chk(True, "current contracts.py is byte-identical to the patch's expected original (applies verbatim)")
    else:
        norm = sha(cur.replace(b"\r\n", b"\n"))
        chk(False, "", "current contracts.py DIFFERS from the patch's expected original"
            + (" (differs only by CRLF line endings)" if norm == ORIG_CONTRACTS else " (content differs: G2.3/G2.4 edits?)")
            + " -> STOP, do not force; send me the file")
    try:
        tree = ast.parse(cur.decode("utf-8"))
        fn = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_datetime_problem"]
        if fn:
            say("INFO", "current _datetime_problem:\n" + ast.get_source_segment(cur.decode("utf-8"), fn[0]))
        chk(len(fn) == 1, "exactly one _datetime_problem found", "_datetime_problem not found exactly once")
    except Exception as e:
        chk(False, f"could not parse contracts.py: {e}")
    chk("import re" in cur.decode("utf-8", "replace") or re.search(rb"^import re\b", cur, re.M) is not None,
        "contracts.py already imports re (the patch uses re.sub)")
    t = pkg / "tests" / "test_text_timestamp_compat.py"
    chk(not t.exists(), "tests/test_text_timestamp_compat.py does not exist yet")
    new_tests = set(re.findall(rb"^def (test_\w+)", z.read("tests/test_text_timestamp_compat.py"), re.M))
    clash = []
    for f in (pkg / "tests").glob("test_*.py"):
        if f.name != t.name:
            clash += [(f.name, n.decode()) for n in new_tests if re.search(rb"^def " + n + rb"\b", f.read_bytes(), re.M)]
    chk(not clash, "no test-name clashes with existing tests", f"test-name clashes: {clash}")
    for rel in FROZEN:
        p = pkg / rel
        say("INFO", f"{rel}: {sha(p.read_bytes()) if p.is_file() else 'MISSING'}  (read-only, never modified)")
    g = shutil.which("git")
    if g:
        r = subprocess.run([g, "-C", str(pkg), "status", "--short"], capture_output=True, text=True)
        say("INFO", "git status --short:\n" + (r.stdout.strip() or "(clean or not a repo) " + r.stderr.strip()))
    else:
        say("INFO", "git not available")
    say("RESULT", "PREFLIGHT PASSED - safe to apply" if ok else "PREFLIGHT FAILED - do not apply")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["preflight", "apply", "rollback"])
    ap.add_argument("--pkg", required=True, type=Path, help="the representation_learning directory")
    ap.add_argument("--zip", required=True, type=Path)
    ap.add_argument("--backup", type=Path, default=Path("timestamp_patch_backup"))
    a = ap.parse_args()
    pkg, bak = a.pkg.resolve(), a.backup.resolve()
    if a.mode == "preflight":
        sys.exit(0 if preflight(pkg, a.zip) else 1)
    if a.mode == "apply":
        if not preflight(pkg, a.zip):
            sys.exit(1)
        bak.mkdir(parents=True, exist_ok=False)
        src = pkg / "text" / "contracts.py"
        shutil.copy2(src, bak / "contracts.py")
        assert sha((bak / "contracts.py").read_bytes()) == ORIG_CONTRACTS
        (bak / "contracts.py.sha256").write_text(f"{ORIG_CONTRACTS}  contracts.py\n")
        z = zipfile.ZipFile(a.zip)
        for n, h in NEW.items():
            data = z.read(n)
            (pkg / n).write_bytes(data)
            assert sha((pkg / n).read_bytes()) == h, n
        say("DONE", f"applied. backup: {bak} (SHA-256 verified = {ORIG_CONTRACTS[:12]}…)")
        for n, h in NEW.items(): say("HASH", f"{h}  {n}")
        say("ROLLBACK", f"python {Path(__file__).name} rollback --pkg {pkg} --zip {a.zip} --backup {bak}")
    if a.mode == "rollback":
        b = bak / "contracts.py"
        assert sha(b.read_bytes()) == ORIG_CONTRACTS, "backup hash mismatch - refusing"
        shutil.copy2(b, pkg / "text" / "contracts.py")
        t = pkg / "tests" / "test_text_timestamp_compat.py"
        if t.exists(): t.unlink()
        say("DONE", f"restored contracts.py = {sha((pkg / 'text' / 'contracts.py').read_bytes())}; removed new test file")

main()
